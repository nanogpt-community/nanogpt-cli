use std::io::{BufRead, BufReader};
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use reqwest::Method;
use reqwest::blocking::multipart::{Form, Part};
use reqwest::blocking::{Client, Response};
use reqwest::header::{ACCEPT, AUTHORIZATION, HeaderMap, HeaderName, HeaderValue};
use serde_json::{Map, Value, json};
use url::Url;

use crate::conversation::ConversationMessage;

#[derive(Debug, Clone)]
pub struct ClientConfig {
    pub base_url: String,
    pub api_key: Option<String>,
    pub timeout_secs: u64,
}

#[derive(Debug, Clone)]
pub struct ChatRequest {
    pub model: String,
    pub system_prompt: Option<String>,
    pub messages: Vec<ConversationMessage>,
    pub temperature: Option<f64>,
    pub max_tokens: Option<u32>,
    pub top_p: Option<f64>,
    pub service_tier: Option<String>,
    pub reasoning_effort: Option<String>,
    pub billing_mode: Option<String>,
    pub provider: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ChatResult {
    pub content: String,
}

#[derive(Clone)]
pub struct NanoGptClient {
    http: Client,
    cfg: ClientConfig,
}

impl NanoGptClient {
    pub fn new(cfg: ClientConfig) -> Result<Self> {
        let http = Client::builder()
            .timeout(Duration::from_secs(cfg.timeout_secs))
            .user_agent(format!("nanogpt-cli/{}", env!("CARGO_PKG_VERSION")))
            .build()
            .context("failed to build HTTP client")?;

        Ok(Self { http, cfg })
    }

    pub fn has_api_key(&self) -> bool {
        self.cfg
            .api_key
            .as_deref()
            .is_some_and(|v| !v.trim().is_empty())
    }

    pub fn set_api_key(&mut self, api_key: String) {
        self.cfg.api_key = Some(api_key);
    }

    pub fn request_json(
        &self,
        method: Method,
        path: &str,
        query: &[(String, String)],
        headers: &[(String, String)],
        body: Option<Value>,
    ) -> Result<Value> {
        let response = self.send(method, path, query, headers, body, None)?;
        parse_json_response(response)
    }

    pub fn request_text(
        &self,
        method: Method,
        path: &str,
        query: &[(String, String)],
        headers: &[(String, String)],
        body: Option<Value>,
    ) -> Result<(u16, String)> {
        let response = self.send(method, path, query, headers, body, None)?;
        let status = response.status().as_u16();
        let text = response.text().context("failed to read response body")?;
        Ok((status, text))
    }

    pub fn request_text_multipart(
        &self,
        method: Method,
        path: &str,
        query: &[(String, String)],
        headers: &[(String, String)],
        form_fields: &[(String, String)],
        form_files: &[(String, String)],
    ) -> Result<(u16, String)> {
        let mut form = Form::new();
        for (k, v) in form_fields {
            form = form.text(k.to_string(), v.to_string());
        }
        for (field, filepath) in form_files {
            let part = Part::file(filepath)
                .with_context(|| format!("failed to attach file {filepath} to field {field}"))?;
            form = form.part(field.to_string(), part);
        }
        let response = self.send(method, path, query, headers, None, Some(form))?;
        let status = response.status().as_u16();
        let text = response.text().context("failed to read response body")?;
        Ok((status, text))
    }

    pub fn chat_completion(&self, req: &ChatRequest) -> Result<ChatResult> {
        let body = build_chat_request_body(req, false);
        let mut headers = vec![];
        if let Some(provider) = &req.provider {
            headers.push(("X-Provider".to_string(), provider.clone()));
        }
        if let Some(mode) = &req.billing_mode {
            headers.push(("X-Billing-Mode".to_string(), mode.clone()));
        }

        let value = self.request_json(
            Method::POST,
            "/v1/chat/completions",
            &[],
            &headers,
            Some(body),
        )?;

        let content = value
            .pointer("/choices/0/message/content")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();

        Ok(ChatResult { content })
    }

    pub fn chat_completion_stream<F>(
        &self,
        req: &ChatRequest,
        mut on_delta: F,
    ) -> Result<ChatResult>
    where
        F: FnMut(&str),
    {
        let body = build_chat_request_body(req, true);
        let mut headers = vec![(ACCEPT.as_str().to_string(), "text/event-stream".to_string())];
        if let Some(provider) = &req.provider {
            headers.push(("X-Provider".to_string(), provider.clone()));
        }
        if let Some(mode) = &req.billing_mode {
            headers.push(("X-Billing-Mode".to_string(), mode.clone()));
        }

        let response = self.send(
            Method::POST,
            "/v1/chat/completions",
            &[],
            &headers,
            Some(body),
            None,
        )?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().unwrap_or_default();
            bail!("chat streaming failed ({status}): {text}");
        }

        let mut full_text = String::new();
        let reader = BufReader::new(response);

        for line in reader.lines() {
            let line = line.context("failed to read stream line")?;
            if let Some(data) = line.strip_prefix("data: ") {
                if data == "[DONE]" {
                    break;
                }

                let chunk: Value = match serde_json::from_str(data) {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                if let Some(delta) = chunk
                    .pointer("/choices/0/delta/content")
                    .and_then(Value::as_str)
                {
                    full_text.push_str(delta);
                    on_delta(delta);
                }
            }
        }

        Ok(ChatResult { content: full_text })
    }

    fn send(
        &self,
        method: Method,
        path: &str,
        query: &[(String, String)],
        headers: &[(String, String)],
        body: Option<Value>,
        form: Option<Form>,
    ) -> Result<Response> {
        let url = self.build_url(path, query)?;

        let mut req = self.http.request(method, url);
        req = self.apply_auth(req);

        let mut header_map = HeaderMap::new();
        for (k, v) in headers {
            let name = HeaderName::from_bytes(k.as_bytes())
                .with_context(|| format!("invalid header name: {k}"))?;
            let value = HeaderValue::from_str(v)
                .with_context(|| format!("invalid header value for {k}"))?;
            header_map.insert(name, value);
        }
        req = req.headers(header_map);

        if let Some(f) = form {
            req = req.multipart(f);
        } else if let Some(json_body) = body {
            req = req.json(&json_body);
        }

        req.send().context("request failed")
    }

    fn build_url(&self, path: &str, query: &[(String, String)]) -> Result<Url> {
        let mut url = if path.starts_with("http://") || path.starts_with("https://") {
            Url::parse(path).with_context(|| format!("invalid URL: {path}"))?
        } else {
            let base = self.cfg.base_url.trim_end_matches('/');
            let path = path.trim_start_matches('/');
            Url::parse(&format!("{base}/{path}")).with_context(|| {
                format!(
                    "invalid base URL/path combination: {} + {}",
                    self.cfg.base_url, path
                )
            })?
        };

        if !query.is_empty() {
            {
                let mut pairs = url.query_pairs_mut();
                for (k, v) in query {
                    pairs.append_pair(k, v);
                }
            }
        }

        Ok(url)
    }

    fn apply_auth(
        &self,
        req: reqwest::blocking::RequestBuilder,
    ) -> reqwest::blocking::RequestBuilder {
        if let Some(key) = &self.cfg.api_key {
            req.header("x-api-key", key)
                .header(AUTHORIZATION, format!("Bearer {key}"))
        } else {
            req
        }
    }
}

pub fn build_chat_request_body(req: &ChatRequest, stream: bool) -> Value {
    let mut messages = Vec::new();
    if let Some(system) = &req.system_prompt {
        messages.push(json!({
            "role": "system",
            "content": system,
        }));
    }

    for msg in &req.messages {
        messages.push(json!({
            "role": msg.role,
            "content": msg.content,
        }));
    }

    let mut obj = Map::new();
    obj.insert("model".to_string(), Value::String(req.model.clone()));
    obj.insert("messages".to_string(), Value::Array(messages));
    obj.insert("stream".to_string(), Value::Bool(stream));

    if let Some(v) = req.temperature {
        obj.insert("temperature".to_string(), json!(v));
    }
    if let Some(v) = req.max_tokens {
        obj.insert("max_tokens".to_string(), json!(v));
    }
    if let Some(v) = req.top_p {
        obj.insert("top_p".to_string(), json!(v));
    }
    if let Some(v) = &req.service_tier {
        obj.insert("service_tier".to_string(), json!(v));
    }
    if let Some(v) = &req.reasoning_effort {
        obj.insert("reasoning_effort".to_string(), json!(v));
    }
    if let Some(v) = &req.billing_mode {
        obj.insert("billing_mode".to_string(), json!(v));
    }

    Value::Object(obj)
}

pub fn parse_key_value(pair: &str, delimiter: char) -> Result<(String, String)> {
    let mut split = pair.splitn(2, delimiter);
    let key = split
        .next()
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .ok_or_else(|| anyhow!("invalid pair: {pair}"))?;
    let value = split
        .next()
        .map(str::trim)
        .ok_or_else(|| anyhow!("invalid pair: {pair}"))?;

    Ok((key.to_string(), value.to_string()))
}

pub fn parse_json_response(response: Response) -> Result<Value> {
    let status = response.status();
    let text = response.text().context("failed to read response body")?;

    if status.is_success() {
        if text.trim().is_empty() {
            return Ok(json!({}));
        }
        serde_json::from_str(&text).context("response is not valid JSON")
    } else {
        let maybe_json: Result<Value, _> = serde_json::from_str(&text);
        if let Ok(value) = maybe_json {
            bail!(
                "request failed ({status}): {}",
                serde_json::to_string_pretty(&value)?
            );
        }
        bail!("request failed ({status}): {text}");
    }
}
