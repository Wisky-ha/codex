use crate::auth::SharedAuthProvider;
use crate::common::ResponseStream;
use crate::common::ResponsesApiRequest;
use crate::endpoint::session::EndpointSession;
use crate::error::ApiError;
use crate::provider::Provider;
use crate::requests::Compression;
use crate::requests::headers::build_session_headers;
use crate::requests::headers::insert_header;
use crate::requests::headers::subagent_header;
use crate::sse::chat::spawn_chat_response_stream;
use crate::telemetry::SseTelemetry;
use codex_client::EncodedJsonBody;
use codex_client::HttpTransport;
use codex_client::RequestTelemetry;
use codex_protocol::models::ContentItem;
use codex_protocol::models::ResponseItem;
use codex_protocol::protocol::SessionSource;
use http::HeaderMap;
use http::Method;
use serde::Serialize;
use std::sync::Arc;
use std::sync::OnceLock;

// ── Chat API request types ──

#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<ChatTool>>,
    tool_choice: String,  // "auto"
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<StreamOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    // DeepSeek-specific thinking mode
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<Thinking>,
    // temperature/top_p thinking 模式下不生效，MVP 省略
    // response_format (json_schema) → MVP 跳过
}

#[derive(Debug, Serialize)]
pub(crate) struct Thinking {
    r#type: String,  // "enabled"
}

#[derive(Debug, Serialize)]
pub(crate) struct StreamOptions {
    include_usage: bool,
}

#[derive(Debug, Serialize)]
pub(crate) struct ChatMessage {
    role: String,  // system/user/assistant/tool
    content: ChatContent,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ChatToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
pub(crate) enum ChatContent {
    Text(String),
    Multimodal(Vec<ChatMultimodalPart>),
}

#[derive(Debug, Serialize)]
pub(crate) struct ChatMultimodalPart {
    r#type: String,
    text: Option<String>,
    image_url: Option<ChatImageUrl>,
}

#[derive(Debug, Serialize)]
pub(crate) struct ChatImageUrl {
    url: String,
    detail: Option<String>,
}

#[derive(Debug, Serialize)]
pub(crate) struct ChatTool {
    r#type: String,  // "function"
    function: ChatToolFunction,
}

#[derive(Debug, Serialize)]
pub(crate) struct ChatToolFunction {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

#[derive(Debug, Serialize)]
pub(crate) struct ChatToolCall {
    id: String,
    r#type: String,  // "function"
    function: ChatToolCallFunction,
}

#[derive(Debug, Serialize)]
pub(crate) struct ChatToolCallFunction {
    name: String,
    arguments: String,  // JSON string
}

// ── Conversion ──

pub fn from_responses_request(req: &ResponsesApiRequest) -> ChatCompletionRequest {
    let mut messages = Vec::new();

    // instructions → system message
    if !req.instructions.is_empty() {
        messages.push(ChatMessage {
            role: "system".to_string(),
            content: ChatContent::Text(req.instructions.clone()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        });
    }

    // Track pending reasoning for next assistant message
    let mut pending_reasoning: Option<String> = None;

    for item in &req.input {
        match item {
            ResponseItem::Message { role, content, .. } => {
                // Reasoning only merges into assistant messages;
                // non-assistant messages clear pending reasoning (orphaned)
                let role_is_assistant = role == "assistant";
                let reasoning = if role_is_assistant { pending_reasoning.take() } else { pending_reasoning.take(); None };
                let chat_msg = ChatMessage {
                    role: role.clone(),
                    content: flatten_content(content),
                    tool_calls: None,
                    tool_call_id: None,
                    reasoning_content: reasoning,
                };
                messages.push(chat_msg);
            }
            ResponseItem::FunctionCall { name, arguments, call_id, .. } => {
                // Merge into last assistant message or create a new one
                if let Some(last) = messages.last_mut() {
                    if last.role == "assistant" {
                        last.tool_calls.get_or_insert_with(Vec::new).push(ChatToolCall {
                            id: call_id.clone(),
                            r#type: "function".to_string(),
                            function: ChatToolCallFunction {
                                name: name.clone(),
                                arguments: arguments.clone(),
                            },
                        });
                        // If we have pending reasoning, attach it to this assistant message
                        if let Some(reasoning) = pending_reasoning.take() {
                            last.reasoning_content = Some(reasoning);
                        }
                        continue;
                    }
                }
                // No previous assistant → create one
                messages.push(ChatMessage {
                    role: "assistant".to_string(),
                    content: ChatContent::Text(String::new()),
                    tool_calls: Some(vec![ChatToolCall {
                        id: call_id.clone(),
                        r#type: "function".to_string(),
                        function: ChatToolCallFunction {
                            name: name.clone(),
                            arguments: arguments.clone(),
                        },
                    }]),
                    tool_call_id: None,
                    reasoning_content: pending_reasoning.take(),
                });
            }
            ResponseItem::CustomToolCall { call_id, name, input, .. } => {
                // Same as FunctionCall but with input as arguments
                if let Some(last) = messages.last_mut() {
                    if last.role == "assistant" {
                        last.tool_calls.get_or_insert_with(Vec::new).push(ChatToolCall {
                            id: call_id.clone(),
                            r#type: "function".to_string(),
                            function: ChatToolCallFunction {
                                name: name.clone(),
                                arguments: input.clone(),
                            },
                        });
                        if let Some(reasoning) = pending_reasoning.take() {
                            last.reasoning_content = Some(reasoning);
                        }
                        continue;
                    }
                }
                messages.push(ChatMessage {
                    role: "assistant".to_string(),
                    content: ChatContent::Text(String::new()),
                    tool_calls: Some(vec![ChatToolCall {
                        id: call_id.clone(),
                        r#type: "function".to_string(),
                        function: ChatToolCallFunction {
                            name: name.clone(),
                            arguments: input.clone(),
                        },
                    }]),
                    tool_call_id: None,
                    reasoning_content: pending_reasoning.take(),
                });
            }
            ResponseItem::FunctionCallOutput { call_id, output, .. } |
            ResponseItem::CustomToolCallOutput { call_id, output, .. } => {
                messages.push(ChatMessage {
                    role: "tool".to_string(),
                    content: ChatContent::Text(output.body.to_text().unwrap_or_default()),
                    tool_calls: None,
                    tool_call_id: Some(call_id.clone()),
                    reasoning_content: None,
                });
            }
            ResponseItem::Reasoning { encrypted_content, .. } => {
                // Cache reasoning content to merge into next assistant message
                if let Some(content) = encrypted_content {
                    pending_reasoning = Some(content.clone());
                }
            }
            _ => {
                // Other item types (LocalShellCall, WebSearchCall, etc.) → skip with warn
                tracing::warn!("Skipping unsupported ResponseItem variant in chat conversion: {:?}",
                    std::mem::discriminant(item));
            }
        }
    }

    // Convert tools: Responses flat format to Chat nested format
    let tools = req.tools.as_ref().map(|tools| {
        tools.iter().map(|t| {
            let name = t.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let description = t.get("description").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let parameters = t.get("parameters").cloned().unwrap_or(serde_json::Value::Null);
            ChatTool {
                r#type: "function".to_string(),
                function: ChatToolFunction { name, description, parameters },
            }
        }).collect()
    });

    ChatCompletionRequest {
        model: req.model.clone(),
        messages,
        tools,
        tool_choice: req.tool_choice.clone(),
        stream: true,
        stream_options: Some(StreamOptions { include_usage: true }),
        max_tokens: None,  // MVP: no max_tokens override
        thinking: Some(Thinking { r#type: "enabled".to_string() }),
    }
}

fn flatten_content(content: &[ContentItem]) -> ChatContent {
    let mut texts = Vec::new();
    let mut multimodal = Vec::new();
    let mut has_image = false;

    for item in content {
        match item {
            ContentItem::InputText { text } | ContentItem::OutputText { text } => {
                texts.push(text.clone());
            }
            ContentItem::InputImage { image_url, detail } => {
                has_image = true;
                multimodal.push(ChatMultimodalPart {
                    r#type: "image_url".to_string(),
                    text: None,
                    image_url: Some(ChatImageUrl {
                        url: image_url.clone(),
                        detail: detail.map(|d| format!("{d:?}")),
                    }),
                });
            }
        }
    }

    if has_image {
        // Add text parts as text items for multimodal
        if !texts.is_empty() {
            multimodal.push(ChatMultimodalPart {
                r#type: "text".to_string(),
                text: Some(texts.join("\n")),
                image_url: None,
            });
        }
        // Note: DeepSeek models may not support images; this will fail on the API side
        ChatContent::Multimodal(multimodal)
    } else {
        ChatContent::Text(texts.join("\n"))
    }
}

// ── ChatClient (Phase 4) ──

#[derive(Default)]
pub struct ChatOptions {
    pub session_id: Option<String>,
    pub thread_id: Option<String>,
    pub session_source: Option<SessionSource>,
    pub extra_headers: HeaderMap,
    pub compression: Compression,
    pub turn_state: Option<Arc<OnceLock<String>>>,
}

pub struct ChatClient<T: HttpTransport> {
    session: EndpointSession<T>,
    sse_telemetry: Option<Arc<dyn SseTelemetry>>,
}

impl<T: HttpTransport> ChatClient<T> {
    pub fn new(transport: T, provider: Provider, auth: SharedAuthProvider) -> Self {
        Self {
            session: EndpointSession::new(transport, provider, auth),
            sse_telemetry: None,
        }
    }

    pub fn with_telemetry(
        self,
        request: Option<Arc<dyn RequestTelemetry>>,
        sse: Option<Arc<dyn SseTelemetry>>,
    ) -> Self {
        Self {
            session: self.session.with_request_telemetry(request),
            sse_telemetry: sse,
        }
    }

    pub async fn stream_request(
        &self,
        request: ResponsesApiRequest,
        options: ChatOptions,
    ) -> Result<ResponseStream, ApiError> {
        let ChatOptions {
            session_id,
            thread_id,
            session_source,
            extra_headers,
            compression: _compression,
            turn_state: _turn_state,
        } = options;

        // Translate Responses request → Chat request
        let chat_req = from_responses_request(&request);

        let encoded = EncodedJsonBody::encode(&chat_req);
        let body = match encoded {
            Ok(b) => Some(b),
            Err(e) => {
                return Err(ApiError::Stream(format!("failed to encode chat request: {e}")));
            }
        };

        let mut headers = extra_headers;
        if let Some(ref thread_id) = thread_id {
            insert_header(&mut headers, "x-client-request-id", thread_id);
        }
        headers.extend(build_session_headers(session_id, thread_id));
        if let Some(subagent) = subagent_header(&session_source) {
            insert_header(&mut headers, "x-openai-subagent", &subagent);
        }

        // Use stream_encoded_json (non-generic) to avoid rustc 1.95.0 ICE
        let stream = self
            .session
            .stream_encoded_json(Method::POST, "chat/completions", headers, body)
            .await?;

        Ok(spawn_chat_response_stream(
            stream,
            self.session.provider().stream_idle_timeout,
            self.sse_telemetry.clone(),
        ))
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;
    use codex_protocol::models::FunctionCallOutputBody;
    use codex_protocol::models::FunctionCallOutputPayload;
    use pretty_assertions::assert_eq;
    use serde_json::json;

    /// Helper: build a ResponsesApiRequest with given instructions + input items.
    fn make_request(instructions: &str, input: Vec<ResponseItem>, tools: Option<Vec<serde_json::Value>>) -> ResponsesApiRequest {
        ResponsesApiRequest {
            model: "deepseek-v4-flash".to_string(),
            instructions: instructions.to_string(),
            input,
            tools,
            tool_choice: "auto".to_string(),
            parallel_tool_calls: true,
            reasoning: None,
            store: false,
            stream: true,
            include: vec![],
            service_tier: None,
            prompt_cache_key: None,
            text: None,
            client_metadata: None,
        }
    }

    #[test]
    fn text_instructions_becomes_system_message() {
        let req = make_request("You are a helpful assistant.", vec![], None);
        let chat = from_responses_request(&req);

        assert_eq!(chat.model, "deepseek-v4-flash");
        assert!(chat.stream);
        assert!(chat.thinking.is_some());
        assert_eq!(chat.thinking.as_ref().unwrap().r#type, "enabled");
        assert_eq!(chat.messages.len(), 1);
        assert_eq!(chat.messages[0].role, "system");
        match &chat.messages[0].content {
            ChatContent::Text(t) => assert_eq!(t, "You are a helpful assistant."),
            _ => panic!("expected Text content"),
        }
    }

    #[test]
    fn empty_instructions_skips_system_message() {
        let req = make_request("", vec![], None);
        let chat = from_responses_request(&req);
        assert_eq!(chat.messages.len(), 0);
    }

    #[test]
    fn user_message_converts_role_and_text_content() {
        let items = vec![
            ResponseItem::Message {
                id: Some("msg_1".into()),
                role: "user".into(),
                content: vec![ContentItem::InputText { text: "Hello".into() }],
                phase: None,
                internal_chat_message_metadata_passthrough: None,
            },
        ];
        let req = make_request("", items, None);
        let chat = from_responses_request(&req);

        assert_eq!(chat.messages.len(), 1);
        assert_eq!(chat.messages[0].role, "user");
        match &chat.messages[0].content {
            ChatContent::Text(t) => assert_eq!(t, "Hello"),
            _ => panic!("expected Text"),
        }
        assert!(chat.messages[0].tool_calls.is_none());
    }

    #[test]
    fn multiple_text_content_items_are_joined() {
        let items = vec![
            ResponseItem::Message {
                id: Some("msg_1".into()),
                role: "user".into(),
                content: vec![
                    ContentItem::InputText { text: "Hello".into() },
                    ContentItem::InputText { text: "World".into() },
                ],
                phase: None,
                internal_chat_message_metadata_passthrough: None,
            },
        ];
        let req = make_request("", items, None);
        let chat = from_responses_request(&req);
        match &chat.messages[0].content {
            ChatContent::Text(t) => assert_eq!(t, "Hello\nWorld"),
            _ => panic!("expected Text"),
        }
    }

    #[test]
    fn output_text_in_user_message_still_works() {
        let items = vec![
            ResponseItem::Message {
                id: Some("msg_1".into()),
                role: "user".into(),
                content: vec![ContentItem::OutputText { text: "result".into() }],
                phase: None,
                internal_chat_message_metadata_passthrough: None,
            },
        ];
        let req = make_request("", items, None);
        let chat = from_responses_request(&req);
        assert_eq!(chat.messages.len(), 1);
        match &chat.messages[0].content {
            ChatContent::Text(t) => assert_eq!(t, "result"),
            _ => panic!("expected Text"),
        }
    }

    #[test]
    fn assistant_message_converts_role_and_text() {
        let items = vec![
            ResponseItem::Message {
                id: Some("msg_2".into()),
                role: "assistant".into(),
                content: vec![ContentItem::OutputText { text: "I think...".into() }],
                phase: None,
                internal_chat_message_metadata_passthrough: None,
            },
        ];
        let req = make_request("", items, None);
        let chat = from_responses_request(&req);

        assert_eq!(chat.messages.len(), 1);
        assert_eq!(chat.messages[0].role, "assistant");
        match &chat.messages[0].content {
            ChatContent::Text(t) => assert_eq!(t, "I think..."),
            _ => panic!("expected Text"),
        }
    }

    #[test]
    fn function_call_merges_into_previous_assistant() {
        let items = vec![
            ResponseItem::Message {
                id: Some("msg_1".into()),
                role: "assistant".into(),
                content: vec![ContentItem::OutputText { text: "Let me check".into() }],
                phase: None,
                internal_chat_message_metadata_passthrough: None,
            },
            ResponseItem::FunctionCall {
                id: Some("fc_1".into()),
                name: "read_file".into(),
                namespace: None,
                arguments: r#"{"path":"/etc/hosts"}"#.into(),
                call_id: "call_abc".into(),
                internal_chat_message_metadata_passthrough: None,
            },
        ];
        let req = make_request("", items, None);
        let chat = from_responses_request(&req);

        // Should produce 1 assistant message with both text and tool_calls
        assert_eq!(chat.messages.len(), 1);
        assert_eq!(chat.messages[0].role, "assistant");
        let tool_calls = chat.messages[0].tool_calls.as_ref().expect("expected tool_calls");
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].id, "call_abc");
        assert_eq!(tool_calls[0].function.name, "read_file");
        assert_eq!(tool_calls[0].function.arguments, r#"{"path":"/etc/hosts"}"#);
    }

    #[test]
    fn function_call_without_prior_assistant_creates_new_one() {
        let items = vec![
            ResponseItem::FunctionCall {
                id: Some("fc_1".into()),
                name: "search".into(),
                namespace: None,
                arguments: r#"{"q":"weather"}"#.into(),
                call_id: "call_1".into(),
                internal_chat_message_metadata_passthrough: None,
            },
        ];
        let req = make_request("", items, None);
        let chat = from_responses_request(&req);

        assert_eq!(chat.messages.len(), 1);
        assert_eq!(chat.messages[0].role, "assistant");
        assert_eq!(chat.messages[0].tool_calls.as_ref().unwrap().len(), 1);
        assert_eq!(chat.messages[0].tool_calls.as_ref().unwrap()[0].function.name, "search");
    }

    #[test]
    fn multiple_function_calls_all_in_one_assistant() {
        let items = vec![
            ResponseItem::FunctionCall {
                id: Some("fc_1".into()),
                name: "read_file".into(),
                namespace: None,
                arguments: r#"{"path":"a"}"#.into(),
                call_id: "call_1".into(),
                internal_chat_message_metadata_passthrough: None,
            },
            ResponseItem::FunctionCall {
                id: Some("fc_2".into()),
                name: "write_file".into(),
                namespace: None,
                arguments: r#"{"path":"b","content":"c"}"#.into(),
                call_id: "call_2".into(),
                internal_chat_message_metadata_passthrough: None,
            },
        ];
        let req = make_request("", items, None);
        let chat = from_responses_request(&req);

        assert_eq!(chat.messages.len(), 1);
        let calls = chat.messages[0].tool_calls.as_ref().unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "read_file");
        assert_eq!(calls[1].function.name, "write_file");
    }

    #[test]
    fn custom_tool_call_is_treated_like_function_call() {
        let items = vec![
            ResponseItem::CustomToolCall {
                id: Some("ctc_1".into()),
                status: Some("completed".into()),
                call_id: "call_ctc".into(),
                name: "custom_tool".into(),
                input: r#"{"key":"val"}"#.into(),
                internal_chat_message_metadata_passthrough: None,
            },
        ];
        let req = make_request("", items, None);
        let chat = from_responses_request(&req);

        assert_eq!(chat.messages.len(), 1);
        assert_eq!(chat.messages[0].role, "assistant");
        let calls = chat.messages[0].tool_calls.as_ref().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "custom_tool");
        assert_eq!(calls[0].function.arguments, r#"{"key":"val"}"#);
    }

    #[test]
    fn function_call_output_becomes_tool_message() {
        let items = vec![
            ResponseItem::FunctionCallOutput {
                id: Some("fco_1".into()),
                call_id: "call_1".into(),
                output: FunctionCallOutputPayload {
                    body: FunctionCallOutputBody::Text("file content".into()),
                    success: None,
                },
                internal_chat_message_metadata_passthrough: None,
            },
        ];
        let req = make_request("", items, None);
        let chat = from_responses_request(&req);

        assert_eq!(chat.messages.len(), 1);
        assert_eq!(chat.messages[0].role, "tool");
        assert_eq!(chat.messages[0].tool_call_id, Some("call_1".into()));
        match &chat.messages[0].content {
            ChatContent::Text(t) => assert_eq!(t, "file content"),
            _ => panic!("expected Text"),
        }
    }

    #[test]
    fn custom_tool_call_output_becomes_tool_message() {
        let items = vec![
            ResponseItem::CustomToolCallOutput {
                id: Some("ctco_1".into()),
                call_id: "call_ctc".into(),
                name: Some("ct".into()),
                output: FunctionCallOutputPayload {
                    body: FunctionCallOutputBody::Text("result".into()),
                    success: None,
                },
                internal_chat_message_metadata_passthrough: None,
            },
        ];
        let req = make_request("", items, None);
        let chat = from_responses_request(&req);

        assert_eq!(chat.messages.len(), 1);
        assert_eq!(chat.messages[0].role, "tool");
        assert_eq!(chat.messages[0].tool_call_id, Some("call_ctc".into()));
        match &chat.messages[0].content {
            ChatContent::Text(t) => assert_eq!(t, "result"),
            _ => panic!("expected Text"),
        }
    }

    #[test]
    fn reasoning_content_merged_into_next_assistant() {
        let items = vec![
            ResponseItem::Reasoning {
                id: Some("r_1".into()),
                summary: vec![],
                content: None,
                encrypted_content: Some("thinking step 1...".into()),
                internal_chat_message_metadata_passthrough: None,
            },
            ResponseItem::Message {
                id: Some("msg_1".into()),
                role: "assistant".into(),
                content: vec![ContentItem::OutputText { text: "answer".into() }],
                phase: None,
                internal_chat_message_metadata_passthrough: None,
            },
        ];
        let req = make_request("", items, None);
        let chat = from_responses_request(&req);

        assert_eq!(chat.messages.len(), 1);
        assert_eq!(chat.messages[0].role, "assistant");
        assert_eq!(chat.messages[0].reasoning_content, Some("thinking step 1...".into()));
    }

    #[test]
    fn reasoning_merged_into_assistant_with_tool_call() {
        let items = vec![
            ResponseItem::Reasoning {
                id: Some("r_1".into()),
                summary: vec![],
                content: None,
                encrypted_content: Some("thinking...".into()),
                internal_chat_message_metadata_passthrough: None,
            },
            ResponseItem::FunctionCall {
                id: Some("fc_1".into()),
                name: "read_file".into(),
                namespace: None,
                arguments: r#"{}"#.into(),
                call_id: "call_1".into(),
                internal_chat_message_metadata_passthrough: None,
            },
        ];
        let req = make_request("", items, None);
        let chat = from_responses_request(&req);

        assert_eq!(chat.messages.len(), 1);
        assert_eq!(chat.messages[0].role, "assistant");
        // reasoning_content must be preserved when tool_calls follow reasoning
        assert_eq!(chat.messages[0].reasoning_content, Some("thinking...".into()));
        assert!(chat.messages[0].tool_calls.is_some());
    }

    #[test]
    fn reasoning_without_following_assistant_is_dropped() {
        let items = vec![
            ResponseItem::Reasoning {
                id: Some("r_1".into()),
                summary: vec![],
                content: None,
                encrypted_content: Some("orphan thinking".into()),
                internal_chat_message_metadata_passthrough: None,
            },
        ];
        let req = make_request("", items, None);
        let chat = from_responses_request(&req);

        // Reasoning without following assistant → dropped (pending_reasoning never consumed)
        assert_eq!(chat.messages.len(), 0);
    }

    #[test]
    fn reasoning_not_merged_into_non_assistant_message() {
        let items = vec![
            ResponseItem::Reasoning {
                id: Some("r_1".into()),
                summary: vec![],
                content: None,
                encrypted_content: Some("thinking".into()),
                internal_chat_message_metadata_passthrough: None,
            },
            ResponseItem::Message {
                id: Some("msg_1".into()),
                role: "user".into(),
                content: vec![ContentItem::InputText { text: "continue".into() }],
                phase: None,
                internal_chat_message_metadata_passthrough: None,
            },
        ];
        let req = make_request("", items, None);
        let chat = from_responses_request(&req);

        // Reasoning before a user message → reasoning dropped, user message present without reasoning
        assert_eq!(chat.messages.len(), 1);
        assert_eq!(chat.messages[0].role, "user");
        assert!(chat.messages[0].reasoning_content.is_none());
    }

    #[test]
    fn tools_are_converted_from_flat_to_nested() {
        let tools = Some(vec![
            json!({
                "name": "read_file",
                "description": "Read a file",
                "parameters": {"type": "object", "properties": {}}
            }),
            json!({
                "name": "write_file",
                "description": "Write a file",
                "parameters": {"type": "object", "properties": {}}
            }),
        ]);
        let req = make_request("Do it.", vec![], tools);
        let chat = from_responses_request(&req);

        let tools = chat.tools.as_ref().expect("expected tools");
        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0].r#type, "function");
        assert_eq!(tools[0].function.name, "read_file");
        assert_eq!(tools[1].function.name, "write_file");
    }

    #[test]
    fn full_conversation_round_trip() {
        // Simulate a complete tool-use conversation cycle
        let items = vec![
            // User asks a question
            ResponseItem::Message {
                id: Some("msg_u1".into()),
                role: "user".into(),
                content: vec![ContentItem::InputText { text: "Read the file".into() }],
                phase: None,
                internal_chat_message_metadata_passthrough: None,
            },
            // Assistant reasons and calls read_file
            ResponseItem::Reasoning {
                id: Some("r_1".into()),
                summary: vec![],
                content: None,
                encrypted_content: Some("I need to read the file...".into()),
                internal_chat_message_metadata_passthrough: None,
            },
            ResponseItem::FunctionCall {
                id: Some("fc_1".into()),
                name: "read_file".into(),
                namespace: None,
                arguments: r#"{"path":"test.txt"}"#.into(),
                call_id: "call_rf".into(),
                internal_chat_message_metadata_passthrough: None,
            },
            // Tool returns result
            ResponseItem::FunctionCallOutput {
                id: Some("fco_1".into()),
                call_id: "call_rf".into(),
                output: FunctionCallOutputPayload {
                    body: FunctionCallOutputBody::Text("file content".into()),
                    success: None,
                },
                internal_chat_message_metadata_passthrough: None,
            },
            // Assistant responds based on tool output
            ResponseItem::Message {
                id: Some("msg_a1".into()),
                role: "assistant".into(),
                content: vec![ContentItem::OutputText { text: "Here is the content".into() }],
                phase: None,
                internal_chat_message_metadata_passthrough: None,
            },
        ];

        let req = make_request("You are a helpful assistant.", items, None);
        let chat = from_responses_request(&req);

        // Expected messages:
        // [0] system
        // [1] user
        // [2] assistant (with reasoning_content + tool_calls)
        // [3] tool
        // [4] assistant (final response)
        assert_eq!(chat.messages.len(), 5);

        // [0] system
        assert_eq!(chat.messages[0].role, "system");

        // [1] user
        assert_eq!(chat.messages[1].role, "user");
        match &chat.messages[1].content {
            ChatContent::Text(t) => assert_eq!(t, "Read the file"),
            _ => panic!("expected Text"),
        }

        // [2] assistant with reasoning + tool call
        assert_eq!(chat.messages[2].role, "assistant");
        assert_eq!(chat.messages[2].reasoning_content, Some("I need to read the file...".into()));
        let calls = chat.messages[2].tool_calls.as_ref().expect("expected tool_calls");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "read_file");

        // [3] tool
        assert_eq!(chat.messages[3].role, "tool");
        assert_eq!(chat.messages[3].tool_call_id, Some("call_rf".into()));

        // [4] assistant final
        assert_eq!(chat.messages[4].role, "assistant");
        match &chat.messages[4].content {
            ChatContent::Text(t) => assert_eq!(t, "Here is the content"),
            _ => panic!("expected Text"),
        }
        // No reasoning_content on the second assistant (no pending reasoning)
        assert!(chat.messages[4].reasoning_content.is_none());
    }

    #[test]
    fn serializes_to_valid_chat_json() {
        // Verify the output can be serialized and looks like a valid Chat API request
        let req = make_request("Be concise.", vec![
            ResponseItem::Message {
                id: Some("msg_u1".into()),
                role: "user".into(),
                content: vec![ContentItem::InputText { text: "Hello".into() }],
                phase: None,
                internal_chat_message_metadata_passthrough: None,
            },
        ], None);
        let chat = from_responses_request(&req);

        let json_str = serde_json::to_string(&chat).expect("serialization failed");
        let parsed: serde_json::Value = serde_json::from_str(&json_str).expect("valid JSON");

        assert_eq!(parsed["model"], "deepseek-v4-flash");
        assert_eq!(parsed["stream"], true);
        assert_eq!(parsed["tool_choice"], "auto");
        assert_eq!(parsed["thinking"]["type"], "enabled");
        assert_eq!(parsed["stream_options"]["include_usage"], true);
        assert_eq!(parsed["messages"][0]["role"], "system");
        assert_eq!(parsed["messages"][1]["role"], "user");
        assert_eq!(parsed["messages"][1]["content"], "Hello");
    }
}
