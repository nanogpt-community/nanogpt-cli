use std::path::PathBuf;

use clap::{ArgAction, Args, Parser, Subcommand, ValueEnum};

#[derive(Debug, Parser)]
#[command(name = "nanogpt-cli", version, about = "Rust CLI/TUI for NanoGPT")]
pub struct Cli {
    /// NanoGPT API key. If omitted, reads NANOGPT_API_KEY.
    #[arg(long, env = "NANOGPT_API_KEY", hide_env_values = true)]
    pub api_key: Option<String>,

    /// Base URL for NanoGPT API.
    #[arg(
        long,
        env = "NANOGPT_BASE_URL",
        default_value = "https://nano-gpt.com/api"
    )]
    pub base_url: String,

    /// HTTP timeout in seconds.
    #[arg(long, default_value_t = 120)]
    pub timeout_secs: u64,

    #[command(subcommand)]
    pub command: Option<Command>,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Interactive terminal chat (REPL)
    Chat(ChatArgs),

    /// Full-screen TUI chat
    Tui(TuiArgs),

    /// List available models
    Models(ModelsArgs),

    /// Perform AI-powered web search
    WebSearch(WebSearchArgs),

    /// Check NanoGPT account balance
    Balance,

    /// Responses API convenience commands
    Responses(ResponsesArgs),

    /// Manage local conversation files
    Conversations(ConversationsArgs),

    /// Generic passthrough to any NanoGPT endpoint
    Api(ApiArgs),
}

#[derive(Debug, Args, Clone)]
pub struct ChatArgs {
    /// Model ID (for example: openai/gpt-5.2)
    #[arg(long, default_value = "openai/gpt-5.2")]
    pub model: String,

    /// Optional system prompt
    #[arg(long)]
    pub system: Option<String>,

    /// Conversation name/id to load and persist
    #[arg(long)]
    pub conversation: Option<String>,

    /// Enable SSE streaming output
    #[arg(long, action = ArgAction::SetTrue)]
    pub stream: bool,

    /// Enable web search by appending :online to model (unless model already has suffix)
    #[arg(long, action = ArgAction::SetTrue)]
    pub web: bool,

    /// Enable deep web search by appending :online/linkup-deep (unless model already has suffix)
    #[arg(long, action = ArgAction::SetTrue)]
    pub deep_web: bool,

    #[arg(long)]
    pub temperature: Option<f64>,

    #[arg(long)]
    pub max_tokens: Option<u32>,

    #[arg(long)]
    pub top_p: Option<f64>,

    #[arg(long)]
    pub provider: Option<String>,

    #[arg(long)]
    pub billing_mode: Option<String>,

    #[arg(long)]
    pub service_tier: Option<String>,

    #[arg(long)]
    pub reasoning_effort: Option<String>,
}

#[derive(Debug, Args, Clone)]
pub struct TuiArgs {
    /// Model ID (for example: openai/gpt-5.2)
    #[arg(long, default_value = "openai/gpt-5.2")]
    pub model: String,

    /// Workspace root for agent tools (defaults to current directory)
    #[arg(long)]
    pub workspace: Option<PathBuf>,

    /// Optional system prompt
    #[arg(long)]
    pub system: Option<String>,

    /// Conversation name/id to load and persist
    #[arg(long)]
    pub conversation: Option<String>,

    /// Enable web search by appending :online to model (unless model already has suffix)
    #[arg(long, action = ArgAction::SetTrue)]
    pub web: bool,

    /// Enable deep web search by appending :online/linkup-deep (unless model already has suffix)
    #[arg(long, action = ArgAction::SetTrue)]
    pub deep_web: bool,

    #[arg(long)]
    pub temperature: Option<f64>,

    #[arg(long)]
    pub max_tokens: Option<u32>,

    #[arg(long)]
    pub top_p: Option<f64>,

    #[arg(long)]
    pub provider: Option<String>,

    #[arg(long)]
    pub billing_mode: Option<String>,

    #[arg(long)]
    pub service_tier: Option<String>,

    #[arg(long)]
    pub reasoning_effort: Option<String>,
}

impl Default for TuiArgs {
    fn default() -> Self {
        Self {
            model: "openai/gpt-5.2".to_string(),
            workspace: None,
            system: None,
            conversation: None,
            web: false,
            deep_web: false,
            temperature: None,
            max_tokens: None,
            top_p: None,
            provider: None,
            billing_mode: None,
            service_tier: None,
            reasoning_effort: None,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum ModelScope {
    Canonical,
    Subscription,
    Paid,
}

#[derive(Debug, Args)]
pub struct ModelsArgs {
    /// Include detailed metadata (pricing, context length)
    #[arg(long, action = ArgAction::SetTrue)]
    pub detailed: bool,

    /// Filter by model catalog scope
    #[arg(long, value_enum, default_value_t = ModelScope::Canonical)]
    pub scope: ModelScope,

    /// Filter model IDs containing this text
    #[arg(long)]
    pub filter: Option<String>,

    /// Show providers and provider pricing for a specific model ID
    #[arg(long)]
    pub show_providers_for: Option<String>,

    /// Output raw JSON
    #[arg(long, action = ArgAction::SetTrue)]
    pub json: bool,
}

#[derive(Debug, Clone, ValueEnum)]
pub enum WebOutput {
    SearchResults,
    SourcedAnswer,
    Structured,
}

#[derive(Debug, Clone, ValueEnum)]
pub enum WebProvider {
    Linkup,
    Tavily,
    Exa,
    Kagi,
    Perplexity,
    Valyu,
    Brave,
}

#[derive(Debug, Args)]
pub struct WebSearchArgs {
    /// Search query
    pub query: String,

    /// Search provider
    #[arg(long, value_enum, default_value_t = WebProvider::Linkup)]
    pub provider: WebProvider,

    /// Search depth (standard|deep, plus Exa-specific: fast|auto|neural|deep)
    #[arg(long, default_value = "standard")]
    pub depth: String,

    /// Output format
    #[arg(long, value_enum, default_value_t = WebOutput::SearchResults)]
    pub output: WebOutput,

    /// Structured schema JSON string (required with --output structured)
    #[arg(long)]
    pub structured_schema: Option<String>,

    #[arg(long)]
    pub from_date: Option<String>,

    #[arg(long)]
    pub to_date: Option<String>,

    #[arg(long, value_delimiter = ',')]
    pub include_domains: Vec<String>,

    #[arg(long, value_delimiter = ',')]
    pub exclude_domains: Vec<String>,

    #[arg(long, action = ArgAction::SetTrue)]
    pub include_images: bool,

    /// Arbitrary provider-specific JSON fields merged into the request body
    #[arg(long)]
    pub extra_json: Option<String>,

    #[arg(long, action = ArgAction::SetTrue)]
    pub json: bool,
}

#[derive(Debug, Args)]
pub struct ResponsesArgs {
    #[command(subcommand)]
    pub command: ResponsesCommand,
}

#[derive(Debug, Subcommand)]
pub enum ResponsesCommand {
    /// Create a response
    Create(ResponsesCreateArgs),

    /// Get a stored response by ID
    Get(ResponsesGetArgs),

    /// Delete a stored response by ID
    Delete(ResponsesGetArgs),

    /// Get endpoint info
    Info,
}

#[derive(Debug, Args)]
pub struct ResponsesCreateArgs {
    #[arg(long)]
    pub model: String,

    /// Input text
    #[arg(long)]
    pub input: Option<String>,

    /// Load input text from file
    #[arg(long)]
    pub input_file: Option<PathBuf>,

    #[arg(long)]
    pub instructions: Option<String>,

    #[arg(long)]
    pub previous_response_id: Option<String>,

    #[arg(long)]
    pub temperature: Option<f64>,

    #[arg(long)]
    pub top_p: Option<f64>,

    #[arg(long)]
    pub max_output_tokens: Option<u32>,

    #[arg(long, action = ArgAction::SetTrue)]
    pub store: bool,

    #[arg(long)]
    pub user: Option<String>,

    #[arg(long)]
    pub service_tier: Option<String>,

    #[arg(long)]
    pub billing_mode: Option<String>,

    #[arg(long)]
    pub retention_days: Option<u32>,

    /// Arbitrary JSON to merge into request body
    #[arg(long)]
    pub extra_json: Option<String>,
}

#[derive(Debug, Args)]
pub struct ResponsesGetArgs {
    pub id: String,
}

#[derive(Debug, Args)]
pub struct ConversationsArgs {
    #[command(subcommand)]
    pub command: ConversationsCommand,
}

#[derive(Debug, Subcommand)]
pub enum ConversationsCommand {
    /// List saved conversations
    List,

    /// Show a saved conversation in JSON
    Show { id: String },

    /// Delete a saved conversation
    Delete { id: String },
}

#[derive(Debug, Args)]
pub struct ApiArgs {
    /// HTTP method
    #[arg(long, default_value = "GET")]
    pub method: String,

    /// Path like /v1/models or full URL
    #[arg(long)]
    pub path: String,

    /// Query params key=value (repeatable)
    #[arg(long = "query", short = 'q')]
    pub query: Vec<String>,

    /// Headers key:value (repeatable)
    #[arg(long = "header", short = 'H')]
    pub headers: Vec<String>,

    /// Raw JSON string body
    #[arg(long)]
    pub json: Option<String>,

    /// Load JSON body from file
    #[arg(long)]
    pub body_file: Option<PathBuf>,

    /// Form field key=value (repeatable)
    #[arg(long = "form")]
    pub form_fields: Vec<String>,

    /// Form file field as key=/path/to/file (repeatable)
    #[arg(long = "form-file")]
    pub form_files: Vec<String>,

    /// Pretty print JSON output
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    pub pretty: bool,
}
