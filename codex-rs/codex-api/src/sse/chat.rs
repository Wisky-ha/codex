use crate::common::ResponseEvent;
use crate::common::ResponseStream;
use crate::error::ApiError;
use crate::telemetry::SseTelemetry;
use codex_client::ByteStream;
use codex_client::StreamResponse;
use codex_protocol::models::ContentItem;
use codex_protocol::models::ResponseItem;
use codex_protocol::models::MessagePhase;
use codex_protocol::protocol::TokenUsage;
use eventsource_stream::Eventsource;
use futures::StreamExt;
use serde::Deserialize;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::time::Instant;
use tokio::time::timeout;
use tracing::debug;
use tracing::trace;

// ── DeepSeek SSE chunk types ──

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct ChatStreamChunk {
    #[serde(default)]
    choices: Vec<ChatChoice>,
    #[serde(default)]
    usage: Option<ChatUsage>,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    model: Option<String>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct ChatChoice {
    delta: ChatDelta,
    #[serde(default)]
    finish_reason: Option<String>,
    #[serde(default)]
    index: Option<usize>,
}

#[derive(Debug, Deserialize, Default)]
#[allow(dead_code)]
struct ChatDelta {
    #[serde(default)]
    role: Option<String>,
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<ChatDeltaToolCall>>,
    #[serde(default)]
    reasoning_content: Option<String>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct ChatDeltaToolCall {
    index: usize,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<ChatDeltaFn>,
}

#[derive(Debug, Deserialize, Default)]
#[allow(dead_code)]
struct ChatDeltaFn {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct ChatUsage {
    #[serde(default)]
    prompt_tokens: i64,
    #[serde(default)]
    completion_tokens: i64,
    #[serde(default)]
    total_tokens: i64,
}

// ── Tool call accumulator ──

#[derive(Debug, Default)]
struct ToolCallAccum {
    id: String,
    name: String,
    arguments: String,
}

// ── ChatStreamConverter ──

#[allow(dead_code)]
struct ChatStreamConverter {
    response_id: String,
    item_id_counter: u64,
    model: String,
    content_started: bool,
    tool_calls_started: bool,
    accumulated_text: String,
    accumulated_reasoning: String,
    reasoning_item_id: Option<String>,
    reasoning_summary: String,
    tool_call_accum: Vec<ToolCallAccum>,
}

impl ChatStreamConverter {
    fn new(response_id: String, model: String) -> Self {
        Self {
            response_id,
            item_id_counter: 0,
            model,
            content_started: false,
            tool_calls_started: false,
            accumulated_text: String::new(),
            accumulated_reasoning: String::new(),
            reasoning_item_id: None,
            reasoning_summary: String::new(),
            tool_call_accum: Vec::new(),
        }
    }

    fn next_item_id(&mut self) -> String {
        self.item_id_counter += 1;
        format!("msg_{}", self.item_id_counter)
    }

    fn process(&mut self, chunk: ChatStreamChunk) -> Vec<ResponseEvent> {
        let mut events = Vec::new();

        for choice in &chunk.choices {
            let delta = &choice.delta;
            let finish_reason = &choice.finish_reason;

            // Reasoning content
            if let Some(ref reasoning) = delta.reasoning_content {
                if !reasoning.is_empty() {
                    self.accumulated_reasoning.push_str(reasoning);
                    self.reasoning_summary.push_str(reasoning);
                    events.push(ResponseEvent::ReasoningSummaryDelta {
                        delta: reasoning.clone(),
                        summary_index: 0,
                    });
                }
            }

            // Tool calls (delta)
            if let Some(ref tool_calls) = delta.tool_calls {
                for tc in tool_calls {
                    // Ensure the accumulator vector is large enough
                    while self.tool_call_accum.len() <= tc.index {
                        self.tool_call_accum.push(ToolCallAccum::default());
                    }

                    let accum = &mut self.tool_call_accum[tc.index];

                    // First chunk: has id and name
                    if let Some(ref id) = tc.id {
                        accum.id = id.clone();
                    }
                    if let Some(ref function) = tc.function {
                        if let Some(ref name) = function.name {
                            accum.name = name.clone();
                        }
                        if let Some(ref args) = function.arguments {
                            accum.arguments.push_str(args);
                        }
                    }

                    // Emit ToolCallInputDelta
                    events.push(ResponseEvent::ToolCallInputDelta {
                        item_id: accum.id.clone(),
                        call_id: Some(accum.id.clone()),
                        delta: tc.function.as_ref()
                            .and_then(|f| f.arguments.clone())
                            .unwrap_or_default(),
                    });

                    self.tool_calls_started = true;
                }
            }

            // Text content (delta)
            if let Some(ref content) = delta.content {
                if !content.is_empty() {
                    self.accumulated_text.push_str(content);
                    events.push(ResponseEvent::OutputTextDelta(content.clone()));
                    self.content_started = true;
                }
            }

            // Finish reason → emit output item done
            if let Some(reason) = finish_reason {
                match reason.as_str() {
                    "stop" => {
                        // Reasoning OutputItemDone comes before Message (Responses API protocol order)
                        let reasoning_item = if !self.accumulated_reasoning.is_empty() {
                            let reasoning = std::mem::take(&mut self.accumulated_reasoning);
                            let _summary = std::mem::take(&mut self.reasoning_summary);
                            Some(ResponseEvent::OutputItemDone(ResponseItem::Reasoning {
                                id: Some(self.next_item_id()),
                                summary: vec![],
                                content: None,
                                encrypted_content: Some(reasoning),
                                internal_chat_message_metadata_passthrough: None,
                            }))
                        } else {
                            None
                        };

                        // Flush accumulated text as a Message item
                        let message_item = if !self.accumulated_text.is_empty() || self.content_started {
                            let text = std::mem::take(&mut self.accumulated_text);
                            self.content_started = false;
                            Some(ResponseEvent::OutputItemDone(ResponseItem::Message {
                                id: Some(self.next_item_id()),
                                role: "assistant".to_string(),
                                content: vec![ContentItem::OutputText { text }],
                                phase: Some(MessagePhase::FinalAnswer),
                                internal_chat_message_metadata_passthrough: None,
                            }))
                        } else {
                            None
                        };

                        events.extend(reasoning_item);
                        events.extend(message_item);

                        // Emit Completed
                        let usage = chunk.usage.as_ref().map(map_usage);
                        events.push(ResponseEvent::Completed {
                            response_id: self.response_id.clone(),
                            token_usage: usage,
                            end_turn: Some(true),
                        });
                    }
                    "tool_calls" => {
                        // Flush accumulated reasoning into a Reasoning item first
                        if !self.accumulated_reasoning.is_empty() {
                            let reasoning = std::mem::take(&mut self.accumulated_reasoning);
                            let _summary = std::mem::take(&mut self.reasoning_summary);
                            events.push(ResponseEvent::OutputItemDone(ResponseItem::Reasoning {
                                id: Some(self.next_item_id()),
                                summary: vec![],
                                content: None,
                                encrypted_content: Some(reasoning),
                                internal_chat_message_metadata_passthrough: None,
                            }));
                        }

                        // Flush accumulated tool calls as FunctionCall items
                        let tool_calls = std::mem::take(&mut self.tool_call_accum);
                        for tc in tool_calls {
                            events.push(ResponseEvent::OutputItemDone(ResponseItem::FunctionCall {
                                id: Some(tc.id.clone()),
                                name: tc.name,
                                namespace: None,
                                arguments: tc.arguments,
                                call_id: tc.id,
                                internal_chat_message_metadata_passthrough: None,
                            }));
                        }

                        // Also flush accumulated text if any (before tool calls)
                        if !self.accumulated_text.is_empty() {
                            let text = std::mem::take(&mut self.accumulated_text);
                            events.push(ResponseEvent::OutputItemDone(ResponseItem::Message {
                                id: Some(self.next_item_id()),
                                role: "assistant".to_string(),
                                content: vec![ContentItem::OutputText { text }],
                                phase: Some(MessagePhase::Commentary),
                                internal_chat_message_metadata_passthrough: None,
                            }));
                        }

                        self.content_started = false;
                        self.tool_calls_started = false;

                        // Emit Completed (no end_turn since tool calls mean more turns)
                        let usage = chunk.usage.as_ref().map(map_usage);
                        events.push(ResponseEvent::Completed {
                            response_id: self.response_id.clone(),
                            token_usage: usage,
                            end_turn: Some(false),
                        });
                    }
                    "length" => {
                        // Truncated - flush what we have
                        let text = std::mem::take(&mut self.accumulated_text);
                        if !text.is_empty() {
                            events.push(ResponseEvent::OutputItemDone(ResponseItem::Message {
                                id: Some(self.next_item_id()),
                                role: "assistant".to_string(),
                                content: vec![ContentItem::OutputText { text }],
                                phase: Some(MessagePhase::FinalAnswer),
                                internal_chat_message_metadata_passthrough: None,
                            }));
                        }
                        let usage = chunk.usage.as_ref().map(map_usage);
                        events.push(ResponseEvent::Completed {
                            response_id: self.response_id.clone(),
                            token_usage: usage,
                            end_turn: Some(true),
                        });
                    }
                    _ => {
                        // Unknown finish reason - best effort
                        debug!("Unknown finish_reason: {reason}");
                    }
                }
            }
        }

        events
    }
}

fn map_usage(usage: &ChatUsage) -> TokenUsage {
    TokenUsage {
        input_tokens: usage.prompt_tokens,
        output_tokens: usage.completion_tokens,
        total_tokens: usage.total_tokens,
        cached_input_tokens: 0,
        reasoning_output_tokens: 0,
    }
}

// ── SSE stream processing ──

pub fn spawn_chat_response_stream(
    stream_response: StreamResponse,
    idle_timeout: Duration,
    telemetry: Option<Arc<dyn SseTelemetry>>,
) -> ResponseStream {
    let upstream_request_id = stream_response
        .headers
        .get("x-request-id")
        .and_then(|value| value.to_str().ok())
        .map(str::to_string);

    let (tx_event, rx_event) = mpsc::channel::<Result<ResponseEvent, ApiError>>(1600);

    tokio::spawn(async move {
        process_chat_sse(stream_response.bytes, tx_event, idle_timeout, telemetry).await;
    });

    ResponseStream {
        rx_event,
        upstream_request_id,
    }
}

async fn process_chat_sse(
    stream: ByteStream,
    tx_event: mpsc::Sender<Result<ResponseEvent, ApiError>>,
    idle_timeout: Duration,
    telemetry: Option<Arc<dyn SseTelemetry>>,
) {
    let mut stream = stream.eventsource();
    let mut converter: Option<ChatStreamConverter> = None;
    let mut response_error: Option<ApiError> = None;

    loop {
        let start = Instant::now();
        let response = timeout(idle_timeout, stream.next()).await;
        if let Some(t) = telemetry.as_ref() {
            t.on_sse_poll(&response, start.elapsed());
        }

        let sse = match response {
            Ok(Some(Ok(sse))) => sse,
            Ok(Some(Err(e))) => {
                debug!("Chat SSE Error: {e:#}");
                let _ = tx_event.send(Err(ApiError::Stream(e.to_string()))).await;
                return;
            }
            Ok(None) => {
                let error = response_error.unwrap_or(ApiError::Stream(
                    "stream closed before completion".into(),
                ));
                let _ = tx_event.send(Err(error)).await;
                return;
            }
            Err(_) => {
                let _ = tx_event
                    .send(Err(ApiError::Stream("idle timeout waiting for SSE".into())))
                    .await;
                return;
            }
        };

        trace!("Chat SSE data: {}", &sse.data);

        // Check for [DONE] signal
        if sse.data.trim() == "[DONE]" {
            // If no Completed was emitted, send an error
            if converter.is_some() {
                let _ = tx_event
                    .send(Err(ApiError::Stream("stream ended without completion".into())))
                    .await;
            }
            return;
        }

        // Try to parse the error
        if let Some(error) = try_parse_chat_error(&sse.data) {
            response_error = Some(error);
            continue;
        }

        // Parse the chunk
        let chunk: ChatStreamChunk = match serde_json::from_str(&sse.data) {
            Ok(chunk) => chunk,
            Err(e) => {
                debug!("Failed to parse Chat SSE event: {e}, data: {}", &sse.data);
                continue;
            }
        };

        // Initialize converter on first chunk with id/model info
        if converter.is_none() {
            let response_id = chunk.id.clone().unwrap_or_else(|| "chat_resp".to_string());
            let model = chunk.model.clone().unwrap_or_default();
            converter = Some(ChatStreamConverter::new(response_id, model));

            // Emit Created
            let _ = tx_event.send(Ok(ResponseEvent::Created)).await;
        }

        // BUG FIX: Use `ref mut` instead of `ref` so that conv.process() can mutate
        if let Some(ref mut conv) = converter {
            let events = conv.process(chunk);
            for event in events {
                let is_completed = matches!(event, ResponseEvent::Completed { .. });
                if tx_event.send(Ok(event)).await.is_err() {
                    return;
                }
                if is_completed {
                    return;
                }
            }
        }
    }
}

// ── Error parsing for Chat API ──

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct ChatApiError {
    #[serde(default)]
    error: Option<ChatApiErrorDetail>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct ChatApiErrorDetail {
    #[serde(default)]
    code: Option<String>,
    #[serde(default)]
    message: Option<String>,
    #[serde(rename = "type")]
    #[serde(default)]
    error_type: Option<String>,
}

fn try_parse_chat_error(data: &str) -> Option<ApiError> {
    let parsed: ChatApiError = serde_json::from_str(data).ok()?;
    let err = parsed.error?;
    match err.code.as_deref() {
        Some("context_length_exceeded") => Some(ApiError::ContextWindowExceeded),
        Some("insufficient_quota") => Some(ApiError::QuotaExceeded),
        Some("invalid_api_key" | "invalid_authentication") => {
            Some(ApiError::Transport(codex_client::TransportError::Http {
                status: http::StatusCode::UNAUTHORIZED,
                url: None,
                headers: None,
                body: err.message.clone(),
            }))
        }
        Some("rate_limit_exceeded") => {
            Some(ApiError::Retryable {
                message: err.message.clone().unwrap_or_default(),
                delay: None,
            })
        }
        Some("server_error" | "internal_error") => Some(ApiError::ServerOverloaded),
        _ => {
            // Unknown error, return as Stream error
            None
        }
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;
    use assert_matches::assert_matches;
    use codex_client::TransportError;
    use futures::TryStreamExt;
    use serde_json::json;
    use tokio::sync::mpsc;
    use tokio_util::io::ReaderStream;

    async fn run_chat_sse(chunks: Vec<&str>) -> Vec<Result<ResponseEvent, ApiError>> {
        let mut body = String::new();
        for chunk in chunks {
            body.push_str(&format!("data: {chunk}\n\n"));
        }

        let (tx, mut rx) = mpsc::channel::<Result<ResponseEvent, ApiError>>(16);
        let stream = ReaderStream::new(std::io::Cursor::new(body))
            .map_err(|err| TransportError::Network(err.to_string()));
        tokio::spawn(process_chat_sse(
            Box::pin(stream),
            tx,
            Duration::from_millis(1000),
            None,
        ));

        let mut events = Vec::new();
        while let Some(ev) = rx.recv().await {
            events.push(ev);
        }
        events
    }

    #[tokio::test]
    async fn parses_text_stream() {
        let chunk1 = json!({
            "id": "chatcmpl-1",
            "model": "deepseek-v4-flash",
            "choices": [{
                "index": 0,
                "delta": { "role": "assistant", "content": "Hello" },
                "finish_reason": null
            }]
        }).to_string();

        let chunk2 = json!({
            "id": "chatcmpl-1",
            "choices": [{
                "index": 0,
                "delta": { "content": " world" },
                "finish_reason": null
            }]
        }).to_string();

        let chunk3 = json!({
            "id": "chatcmpl-1",
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }],
            "usage": { "prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12 }
        }).to_string();

        let events = run_chat_sse(vec![&chunk1, &chunk2, &chunk3]).await;

        assert_matches!(events[0], Ok(ResponseEvent::Created));
        assert_matches!(&events[1], Ok(ResponseEvent::OutputTextDelta(d)) if d == "Hello");
        assert_matches!(&events[2], Ok(ResponseEvent::OutputTextDelta(d)) if d == " world");
        assert_matches!(&events[3], Ok(ResponseEvent::OutputItemDone(ResponseItem::Message { .. })));
        assert_matches!(&events[4], Ok(ResponseEvent::Completed { .. }));
    }

    #[tokio::test]
    async fn parses_reasoning_stream() {
        let chunk1 = json!({
            "id": "chatcmpl-1",
            "model": "deepseek-v4-pro",
            "choices": [{
                "index": 0,
                "delta": { "role": "assistant", "reasoning_content": "Thinking step 1..." },
                "finish_reason": null
            }]
        }).to_string();

        let chunk2 = json!({
            "id": "chatcmpl-1",
            "choices": [{
                "index": 0,
                "delta": { "content": "Answer text" },
                "finish_reason": null
            }]
        }).to_string();

        let chunk3 = json!({
            "id": "chatcmpl-1",
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }],
            "usage": { "prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25 }
        }).to_string();

        let events = run_chat_sse(vec![&chunk1, &chunk2, &chunk3]).await;

        assert_matches!(events[0], Ok(ResponseEvent::Created));
        // reasoning delta
        assert_matches!(&events[1], Ok(ResponseEvent::ReasoningSummaryDelta { delta, .. }) if delta == "Thinking step 1...");
        // text delta
        assert_matches!(&events[2], Ok(ResponseEvent::OutputTextDelta(d)) if d == "Answer text");
        // OutputItemDone: reasoning
        assert_matches!(&events[3], Ok(ResponseEvent::OutputItemDone(ResponseItem::Reasoning { .. })));
        // OutputItemDone: message
        assert_matches!(&events[4], Ok(ResponseEvent::OutputItemDone(ResponseItem::Message { .. })));
        // Completed
        assert_matches!(&events[5], Ok(ResponseEvent::Completed { end_turn: Some(true), .. }));
    }

    #[tokio::test]
    async fn parses_tool_call_stream() {
        let chunk1 = json!({
            "id": "chatcmpl-2",
            "model": "deepseek-v4-flash",
            "choices": [{
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_1",
                        "function": { "name": "read_file", "arguments": "{\"path\"" }
                    }]
                },
                "finish_reason": null
            }]
        }).to_string();

        let chunk2 = json!({
            "id": "chatcmpl-2",
            "choices": [{
                "index": 0,
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "function": { "arguments": ": \"/etc/hosts\"}" }
                    }]
                },
                "finish_reason": null
            }]
        }).to_string();

        let chunk3 = json!({
            "id": "chatcmpl-2",
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "tool_calls"
            }],
            "usage": { "prompt_tokens": 15, "completion_tokens": 8, "total_tokens": 23 }
        }).to_string();

        let events = run_chat_sse(vec![&chunk1, &chunk2, &chunk3]).await;

        assert_matches!(events[0], Ok(ResponseEvent::Created));
        // tool call delta
        assert_matches!(&events[1], Ok(ResponseEvent::ToolCallInputDelta { .. }));
        assert_matches!(&events[2], Ok(ResponseEvent::ToolCallInputDelta { .. }));
        // function call output items
        assert_matches!(&events[3], Ok(ResponseEvent::OutputItemDone(ResponseItem::FunctionCall { name, .. })) if name == "read_file");
        // completed
        assert_matches!(&events[4], Ok(ResponseEvent::Completed { end_turn: Some(false), .. }));
    }

    #[tokio::test]
    async fn error_on_done_before_complete() {
        let chunk1 = json!({
            "id": "chatcmpl-1",
            "model": "deepseek-v4-flash",
            "choices": [{
                "index": 0,
                "delta": { "role": "assistant", "content": "Hello" },
                "finish_reason": null
            }]
        }).to_string();

        let events = run_chat_sse(vec![&chunk1, "[DONE]"]).await;

        assert_matches!(events[0], Ok(ResponseEvent::Created));
        assert_matches!(&events[1], Ok(ResponseEvent::OutputTextDelta(d)) if d == "Hello");
        // Should be an error since no completion was received
        assert!(events[2].is_err());
    }

    #[tokio::test]
    async fn handles_context_window_error() {
        let error_chunk = json!({
            "error": {
                "code": "context_length_exceeded",
                "message": "Token limit exceeded"
            }
        }).to_string();

        let events = run_chat_sse(vec![&error_chunk]).await;

        assert_eq!(events.len(), 1);
        match &events[0] {
            Err(ApiError::ContextWindowExceeded) => {}
            other => panic!("Expected ContextWindowExceeded, got {other:?}"),
        }
    }
}
