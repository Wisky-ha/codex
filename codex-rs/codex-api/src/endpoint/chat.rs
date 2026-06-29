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
                let chat_msg = ChatMessage {
                    role: role.clone(),
                    content: flatten_content(content),
                    tool_calls: None,
                    tool_call_id: None,
                    reasoning_content: pending_reasoning.take(),
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
