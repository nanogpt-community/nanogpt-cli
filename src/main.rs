mod app_config;
mod chat;
mod cli;
mod client;
mod conversation;
mod model_mode;
mod tui;

use std::fs;

use anyhow::{Context, Result, anyhow, bail};
use clap::Parser;
use reqwest::Method;
use serde_json::{Map, Value, json};
use url::form_urlencoded::byte_serialize;

use crate::app_config::{AppConfig, load_config};
use crate::chat::run_chat_repl;
use crate::cli::{
    ApiArgs, Cli, Command, ConversationsCommand, ModelScope, ModelsArgs, ResponsesCommand,
    WebOutput, WebProvider,
};
use crate::client::{ClientConfig, NanoGptClient, parse_key_value};
use crate::conversation as conv_store;
use crate::tui::run_tui;

fn main() -> Result<()> {
    let cli = Cli::parse();
    let disk_cfg = load_config().unwrap_or_else(|_| AppConfig::default());
    let resolved_api_key = cli.api_key.clone().or(disk_cfg.api_key.clone());

    let client = NanoGptClient::new(ClientConfig {
        base_url: cli.base_url.clone(),
        api_key: resolved_api_key,
        timeout_secs: cli.timeout_secs,
    })?;

    let command = if let Some(command) = cli.command {
        command
    } else {
        let mut default_tui = crate::cli::TuiArgs::default();
        if let Some(model) = disk_cfg.default_model {
            default_tui.model = model;
        }
        Command::Tui(default_tui)
    };

    match command {
        Command::Chat(args) => run_chat_repl(&client, args),
        Command::Tui(args) => run_tui(client, args),
        Command::Models(args) => cmd_models(&client, args),
        Command::WebSearch(args) => cmd_web_search(&client, args),
        Command::Balance => cmd_balance(&client),
        Command::Responses(args) => cmd_responses(&client, args.command),
        Command::Conversations(args) => cmd_conversations(args.command),
        Command::Api(args) => cmd_api(&client, args),
    }
}

fn cmd_models(client: &NanoGptClient, args: ModelsArgs) -> Result<()> {
    if let Some(model) = args.show_providers_for.as_ref() {
        let encoded_id = encode_path_segment(model);
        let path = format!("/models/{encoded_id}/providers");
        let value = client.request_json(Method::GET, &path, &[], &[], None)?;
        print_json(&value)?;
        return Ok(());
    }

    let path = match args.scope {
        ModelScope::Canonical => "/v1/models",
        ModelScope::Subscription => "/subscription/v1/models",
        ModelScope::Paid => "/paid/v1/models",
    };

    let query = if args.detailed {
        vec![("detailed".to_string(), "true".to_string())]
    } else {
        vec![]
    };

    let value = client.request_json(Method::GET, path, &query, &[], None)?;

    if args.json {
        print_json(&value)?;
        return Ok(());
    }

    let models = value
        .get("data")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow!("unexpected models response format"))?;

    let filter = args.filter.as_ref().map(|v| v.to_lowercase());

    for model in models {
        let id = model
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or("<unknown>");

        if let Some(f) = &filter {
            if !id.to_lowercase().contains(f) {
                continue;
            }
        }

        if args.detailed {
            let context = model
                .get("context_length")
                .and_then(Value::as_u64)
                .map(|v| v.to_string())
                .unwrap_or_else(|| "-".to_string());
            let desc = model
                .get("description")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let owned_by = model
                .get("owned_by")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let sub_included = model
                .pointer("/subscription/included")
                .and_then(Value::as_bool)
                .map(|v| if v { "yes" } else { "no" })
                .unwrap_or("-");
            println!(
                "{id}\n  owned_by: {owned_by}\n  subscription: {sub_included}\n  context: {context}\n  description: {desc}\n"
            );
        } else {
            println!("{id}");
        }
    }

    Ok(())
}

fn cmd_web_search(client: &NanoGptClient, args: crate::cli::WebSearchArgs) -> Result<()> {
    let provider = match args.provider {
        WebProvider::Linkup => "linkup",
        WebProvider::Tavily => "tavily",
        WebProvider::Exa => "exa",
        WebProvider::Kagi => "kagi",
        WebProvider::Perplexity => "perplexity",
        WebProvider::Valyu => "valyu",
        WebProvider::Brave => "brave",
    };

    let output = match args.output {
        WebOutput::SearchResults => "searchResults",
        WebOutput::SourcedAnswer => "sourcedAnswer",
        WebOutput::Structured => "structured",
    };

    if output == "structured" && args.structured_schema.is_none() {
        bail!("--structured-schema is required when --output structured");
    }

    let mut body = Map::new();
    body.insert("query".to_string(), json!(args.query));
    body.insert("provider".to_string(), json!(provider));
    body.insert("depth".to_string(), json!(args.depth));
    body.insert("outputType".to_string(), json!(output));
    body.insert("includeImages".to_string(), json!(args.include_images));

    if let Some(v) = args.structured_schema {
        body.insert("structuredOutputSchema".to_string(), json!(v));
    }
    if let Some(v) = args.from_date {
        body.insert("fromDate".to_string(), json!(v));
    }
    if let Some(v) = args.to_date {
        body.insert("toDate".to_string(), json!(v));
    }
    if !args.include_domains.is_empty() {
        body.insert("includeDomains".to_string(), json!(args.include_domains));
    }
    if !args.exclude_domains.is_empty() {
        body.insert("excludeDomains".to_string(), json!(args.exclude_domains));
    }
    if let Some(extra_json) = args.extra_json {
        let extra: Value =
            serde_json::from_str(&extra_json).context("invalid JSON in --extra-json")?;
        merge_json_object(&mut body, extra)?;
    }

    let value = client.request_json(Method::POST, "/web", &[], &[], Some(Value::Object(body)))?;

    if args.json {
        print_json(&value)?;
        return Ok(());
    }

    print_json(&value)
}

fn cmd_balance(client: &NanoGptClient) -> Result<()> {
    let value = client.request_json(Method::POST, "/check-balance", &[], &[], None)?;
    print_json(&value)
}

fn cmd_responses(client: &NanoGptClient, command: ResponsesCommand) -> Result<()> {
    match command {
        ResponsesCommand::Create(args) => {
            let input = if let Some(v) = args.input {
                v
            } else if let Some(path) = args.input_file {
                fs::read_to_string(&path)
                    .with_context(|| format!("failed to read input file {}", path.display()))?
            } else {
                bail!("--input or --input-file is required");
            };

            let mut body = Map::new();
            body.insert("model".to_string(), json!(args.model));
            body.insert("input".to_string(), json!(input));

            if let Some(v) = args.instructions {
                body.insert("instructions".to_string(), json!(v));
            }
            if let Some(v) = args.previous_response_id {
                body.insert("previous_response_id".to_string(), json!(v));
            }
            if let Some(v) = args.temperature {
                body.insert("temperature".to_string(), json!(v));
            }
            if let Some(v) = args.top_p {
                body.insert("top_p".to_string(), json!(v));
            }
            if let Some(v) = args.max_output_tokens {
                body.insert("max_output_tokens".to_string(), json!(v));
            }
            if args.store {
                body.insert("store".to_string(), json!(true));
            }
            if let Some(v) = args.user {
                body.insert("user".to_string(), json!(v));
            }
            if let Some(v) = args.service_tier {
                body.insert("service_tier".to_string(), json!(v));
            }
            if let Some(v) = args.billing_mode {
                body.insert("billing_mode".to_string(), json!(v));
            }
            if let Some(v) = args.retention_days {
                body.insert("retention_days".to_string(), json!(v));
            }

            if let Some(extra) = args.extra_json {
                let extra_value: Value =
                    serde_json::from_str(&extra).context("invalid JSON in --extra-json")?;
                merge_json_object(&mut body, extra_value)?;
            }

            let value = client.request_json(
                Method::POST,
                "/v1/responses",
                &[],
                &[],
                Some(Value::Object(body)),
            )?;
            print_json(&value)
        }
        ResponsesCommand::Get(args) => {
            let path = format!("/v1/responses/{}", args.id);
            let value = client.request_json(Method::GET, &path, &[], &[], None)?;
            print_json(&value)
        }
        ResponsesCommand::Delete(args) => {
            let path = format!("/v1/responses/{}", args.id);
            let value = client.request_json(Method::DELETE, &path, &[], &[], None)?;
            print_json(&value)
        }
        ResponsesCommand::Info => {
            let value = client.request_json(Method::GET, "/v1/responses", &[], &[], None)?;
            print_json(&value)
        }
    }
}

fn cmd_conversations(command: ConversationsCommand) -> Result<()> {
    match command {
        ConversationsCommand::List => {
            let list = conv_store::list()?;
            if list.is_empty() {
                println!("No saved conversations.");
                return Ok(());
            }

            for item in list {
                println!(
                    "{}\n  model: {}\n  updated: {}\n  messages: {}\n",
                    item.id, item.model, item.updated_at, item.message_count
                );
            }
            Ok(())
        }
        ConversationsCommand::Show { id } => {
            let conv = conv_store::load(&id)?;
            print_json(&serde_json::to_value(conv)?)
        }
        ConversationsCommand::Delete { id } => {
            conv_store::delete(&id)?;
            println!("Deleted conversation: {id}");
            Ok(())
        }
    }
}

fn cmd_api(client: &NanoGptClient, args: ApiArgs) -> Result<()> {
    let method = Method::from_bytes(args.method.as_bytes())
        .with_context(|| format!("invalid HTTP method: {}", args.method))?;

    let mut query = Vec::new();
    for item in &args.query {
        query.push(parse_key_value(item, '=')?);
    }

    let mut headers = Vec::new();
    for item in &args.headers {
        headers.push(parse_key_value(item, ':')?);
    }

    let mut form_fields = Vec::new();
    for item in &args.form_fields {
        form_fields.push(parse_key_value(item, '=')?);
    }

    let mut form_files = Vec::new();
    for item in &args.form_files {
        form_files.push(parse_key_value(item, '=')?);
    }

    if !form_fields.is_empty() || !form_files.is_empty() {
        let (status, text) = client.request_text_multipart(
            method,
            &args.path,
            &query,
            &headers,
            &form_fields,
            &form_files,
        )?;

        print_api_output(&text, args.pretty);
        if status >= 400 {
            bail!("request failed with status {status}");
        }
        return Ok(());
    }

    let body = match (&args.json, &args.body_file) {
        (Some(_), Some(_)) => bail!("use either --json or --body-file, not both"),
        (Some(raw), None) => {
            let value: Value = serde_json::from_str(raw).context("invalid JSON in --json")?;
            Some(value)
        }
        (None, Some(path)) => {
            let raw = fs::read_to_string(path)
                .with_context(|| format!("failed to read body file {}", path.display()))?;
            let value: Value = serde_json::from_str(&raw).context("body file is not JSON")?;
            Some(value)
        }
        (None, None) => None,
    };

    let (status, text) = client.request_text(method, &args.path, &query, &headers, body)?;
    print_api_output(&text, args.pretty);

    if status >= 400 {
        bail!("request failed with status {status}");
    }

    Ok(())
}

fn print_json(value: &Value) -> Result<()> {
    println!("{}", serde_json::to_string_pretty(value)?);
    Ok(())
}

fn print_api_output(text: &str, pretty: bool) {
    if pretty {
        if let Ok(value) = serde_json::from_str::<Value>(text) {
            if let Ok(pretty_json) = serde_json::to_string_pretty(&value) {
                println!("{pretty_json}");
                return;
            }
        }
    }
    println!("{text}");
}

fn merge_json_object(target: &mut Map<String, Value>, extra: Value) -> Result<()> {
    let obj = extra
        .as_object()
        .ok_or_else(|| anyhow!("--extra-json must be a JSON object"))?;

    for (k, v) in obj {
        target.insert(k.to_string(), v.clone());
    }
    Ok(())
}

fn encode_path_segment(value: &str) -> String {
    byte_serialize(value.as_bytes()).collect::<String>()
}
