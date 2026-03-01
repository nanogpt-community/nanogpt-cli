use std::collections::{BTreeMap, VecDeque};
use std::ffi::OsString;
use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use glob::Pattern;
use uuid::Uuid;

use crate::client::{ChatRequest, NanoGptClient};
use crate::conversation::ConversationMessage;

const MAX_AGENT_STEPS: usize = 8;
const MAX_TOOL_CALLS_PER_STEP: usize = 8;
const MAX_TOOL_OUTPUT_CHARS: usize = 24_000;
const DEFAULT_LIST_MAX_ENTRIES: usize = 400;
const DEFAULT_GLOB_MAX_RESULTS: usize = 800;
const DEFAULT_GREP_MAX_RESULTS: usize = 400;
const DEFAULT_BASH_TIMEOUT_SECS: u64 = 20;
const MAX_BASH_TIMEOUT_SECS: u64 = 120;
const DEFAULT_SESSION_OUTPUT_MAX_CHARS: usize = 12_000;

static SHELL_SESSIONS: OnceLock<Mutex<BTreeMap<u64, ShellSession>>> = OnceLock::new();
static NEXT_SESSION_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone)]
pub struct AgentTurnRequest {
    pub model: String,
    pub system_prompt: Option<String>,
    pub messages: Vec<ConversationMessage>,
    pub user_input: String,
    pub temperature: Option<f64>,
    pub max_tokens: Option<u32>,
    pub top_p: Option<f64>,
    pub service_tier: Option<String>,
    pub reasoning_effort: Option<String>,
    pub billing_mode: Option<String>,
    pub provider: Option<String>,
    pub workspace_root: PathBuf,
}

#[derive(Debug, Clone)]
pub struct AgentTurnResult {
    pub assistant_message: String,
    pub tool_call_count: usize,
}

#[derive(Debug, Clone)]
enum ToolCall {
    ListFiles {
        path: String,
        recursive: bool,
        max_entries: usize,
    },
    GlobFiles {
        pattern: String,
        base_path: String,
        max_results: usize,
    },
    GrepFiles {
        query: String,
        base_path: String,
        include_glob: Option<String>,
        case_sensitive: bool,
        max_results: usize,
    },
    ReadFile {
        path: String,
        start_line: Option<usize>,
        end_line: Option<usize>,
    },
    WriteFile {
        path: String,
        content: String,
    },
    AppendFile {
        path: String,
        content: String,
    },
    Bash {
        command: String,
        timeout_secs: u64,
    },
    Mkdir {
        path: String,
    },
    MovePath {
        from: String,
        to: String,
    },
    DeletePath {
        path: String,
        recursive: bool,
    },
    ApplyPatch {
        path: String,
        search: String,
        replace: String,
        all: bool,
    },
    RunTest {
        command: Option<String>,
        timeout_secs: u64,
    },
    RunLint {
        command: Option<String>,
        timeout_secs: u64,
    },
    BashSessionStart {
        path: String,
        shell: String,
    },
    BashSessionRun {
        session_id: u64,
        command: String,
        timeout_secs: u64,
    },
    BashSessionOutput {
        session_id: u64,
        max_chars: usize,
    },
    BashSessionKill {
        session_id: u64,
    },
    GitStatus {
        short: bool,
    },
    GitDiff {
        staged: bool,
        path: Option<String>,
    },
    GitAdd {
        path: Option<String>,
        all: bool,
    },
    GitCommit {
        message: String,
    },
    Unknown {
        requested: String,
    },
}

#[derive(Debug, Clone)]
struct ToolResult {
    tool: String,
    status: &'static str,
    summary: String,
    output: String,
}

#[derive(Debug, Clone)]
struct Workspace {
    root: PathBuf,
}

#[derive(Debug)]
struct ShellSession {
    child: Child,
    stdin: ChildStdin,
    stdout_buf: Arc<Mutex<String>>,
    stderr_buf: Arc<Mutex<String>>,
}

#[derive(Debug)]
struct CommandOutcome {
    timed_out: bool,
    success: bool,
    exit_code: Option<i32>,
    output: String,
}

impl ShellSession {
    fn stdout_len(&self) -> usize {
        self.stdout_buf
            .lock()
            .map(|buf| buf.len())
            .unwrap_or_default()
    }

    fn stderr_len(&self) -> usize {
        self.stderr_buf
            .lock()
            .map(|buf| buf.len())
            .unwrap_or_default()
    }

    fn stdout_snapshot(&self) -> String {
        self.stdout_buf
            .lock()
            .map(|buf| buf.clone())
            .unwrap_or_default()
    }

    fn stderr_snapshot(&self) -> String {
        self.stderr_buf
            .lock()
            .map(|buf| buf.clone())
            .unwrap_or_default()
    }
}

pub fn run_agent_turn(client: NanoGptClient, request: AgentTurnRequest) -> Result<AgentTurnResult> {
    let workspace = Workspace::new(request.workspace_root)?;
    let agent_system_prompt =
        build_agent_system_prompt(request.system_prompt.as_deref(), &workspace);

    let mut messages = request.messages;
    messages.push(ConversationMessage {
        role: "user".to_string(),
        content: request.user_input,
    });

    let mut tool_call_count = 0usize;

    for _step in 0..MAX_AGENT_STEPS {
        let chat_request = ChatRequest {
            model: request.model.clone(),
            system_prompt: Some(agent_system_prompt.clone()),
            messages: messages.clone(),
            temperature: request.temperature,
            max_tokens: request.max_tokens,
            top_p: request.top_p,
            service_tier: request.service_tier.clone(),
            reasoning_effort: request.reasoning_effort.clone(),
            billing_mode: request.billing_mode.clone(),
            provider: request.provider.clone(),
        };

        let assistant_response = client.chat_completion(&chat_request)?.content;
        let mut tool_calls = parse_tool_calls(&assistant_response);

        if tool_calls.is_empty() {
            return Ok(AgentTurnResult {
                assistant_message: assistant_response,
                tool_call_count,
            });
        }

        if tool_calls.len() > MAX_TOOL_CALLS_PER_STEP {
            tool_calls.truncate(MAX_TOOL_CALLS_PER_STEP);
        }

        tool_call_count += tool_calls.len();

        let mut results = Vec::with_capacity(tool_calls.len());
        for call in tool_calls {
            results.push(execute_tool_call(&workspace, call));
        }

        messages.push(ConversationMessage {
            role: "assistant".to_string(),
            content: assistant_response,
        });

        messages.push(ConversationMessage {
            role: "user".to_string(),
            content: render_tool_results_prompt(&results),
        });
    }

    bail!("agent exceeded max tool iterations ({MAX_AGENT_STEPS}); ask for a narrower task");
}

fn build_agent_system_prompt(user_system_prompt: Option<&str>, workspace: &Workspace) -> String {
    let mut parts = Vec::new();

    if let Some(system_prompt) = user_system_prompt {
        let trimmed = system_prompt.trim();
        if !trimmed.is_empty() {
            parts.push(trimmed.to_string());
        }
    }

    parts.push(format!(
        "You are an agentic coding assistant. Workspace root: {}.",
        workspace.root.display()
    ));
    parts.push(
        "Only read/write files inside the workspace root. Never use paths outside it.".to_string(),
    );
    parts.push(
        "When you need tools, respond with one or more XML blocks and no extra text.".to_string(),
    );
    parts.push(
        "Tool call schema:\n<tool_call>\n  <tool>TOOL_NAME</tool>\n  ...args...\n</tool_call>"
            .to_string(),
    );
    parts.push("Available tools:".to_string());
    parts.push("1) list_files: <path>relative-or-absolute</path> optional <recursive>true|false</recursive> optional <max_entries>400</max_entries>".to_string());
    parts.push("2) glob_files: <pattern>src/**/*.rs</pattern> optional <base_path>.</base_path> optional <max_results>800</max_results>".to_string());
    parts.push("3) grep_files: <query>needle</query> optional <base_path>.</base_path> optional <include_glob>src/**/*.rs</include_glob> optional <case_sensitive>false</case_sensitive> optional <max_results>400</max_results>".to_string());
    parts.push("4) read_file: <path>...</path> optional <start_line>1</start_line> optional <end_line>200</end_line>".to_string());
    parts.push(
        "5) write_file: <path>...</path> <content><![CDATA[...]]></content> (overwrite/create)"
            .to_string(),
    );
    parts.push("6) append_file: <path>...</path> <content><![CDATA[...]]></content>".to_string());
    parts.push("7) mkdir: <path>relative-or-absolute</path>".to_string());
    parts.push("8) move_path: <from>old/path</from> <to>new/path</to>".to_string());
    parts.push(
        "9) delete_path: <path>...</path> optional <recursive>true|false</recursive>".to_string(),
    );
    parts.push(
        "10) bash: <command><![CDATA[...]]></command> optional <timeout_secs>20</timeout_secs>"
            .to_string(),
    );
    parts.push("11) apply_patch: <path>...</path> <search><![CDATA[...]]></search> <replace><![CDATA[...]]></replace> optional <all>true|false</all>".to_string());
    parts.push("12) run_test: optional <command><![CDATA[...]]></command> optional <timeout_secs>20</timeout_secs>".to_string());
    parts.push("13) run_lint: optional <command><![CDATA[...]]></command> optional <timeout_secs>20</timeout_secs>".to_string());
    parts.push(
        "14) bash_session_start: optional <path>.</path> optional <shell>bash</shell>".to_string(),
    );
    parts.push("15) bash_session_run: <session_id>1</session_id> <command><![CDATA[pwd]]></command> optional <timeout_secs>20</timeout_secs>".to_string());
    parts.push(
        "16) bash_session_output: <session_id>1</session_id> optional <max_chars>12000</max_chars>"
            .to_string(),
    );
    parts.push("17) bash_session_kill: <session_id>1</session_id>".to_string());
    parts.push("18) git_status: optional <short>true|false</short>".to_string());
    parts.push(
        "19) git_diff: optional <staged>true|false</staged> optional <path>src/main.rs</path>"
            .to_string(),
    );
    parts.push(
        "20) git_add: optional <path>src/main.rs</path> optional <all>true|false</all>".to_string(),
    );
    parts.push("21) git_commit: <message>your commit message</message>".to_string());
    parts.push(
        "Use relative paths when possible. Keep tool arguments minimal and valid XML.".to_string(),
    );
    parts.push("After tool results arrive, either call more tools or provide the final user-facing answer (normal text, no XML).".to_string());

    parts.join("\n\n")
}

fn parse_tool_calls(response: &str) -> Vec<ToolCall> {
    let blocks = extract_tool_call_blocks(response);
    let mut calls = Vec::with_capacity(blocks.len());

    for block in blocks {
        let name = extract_tag_value(&block, "tool")
            .or_else(|| extract_tag_value(&block, "name"))
            .unwrap_or_default();

        let tool_name = name.trim().to_lowercase();

        let call = match tool_name.as_str() {
            "list_files" => ToolCall::ListFiles {
                path: extract_tag_value(&block, "path").unwrap_or_else(|| ".".to_string()),
                recursive: parse_bool(extract_tag_value(&block, "recursive").as_deref(), true),
                max_entries: parse_usize(
                    extract_tag_value(&block, "max_entries").as_deref(),
                    DEFAULT_LIST_MAX_ENTRIES,
                )
                .max(1)
                .min(2000),
            },
            "glob_files" => ToolCall::GlobFiles {
                pattern: extract_tag_value(&block, "pattern").unwrap_or_default(),
                base_path: extract_tag_value(&block, "base_path")
                    .unwrap_or_else(|| ".".to_string()),
                max_results: parse_usize(
                    extract_tag_value(&block, "max_results").as_deref(),
                    DEFAULT_GLOB_MAX_RESULTS,
                )
                .max(1)
                .min(10_000),
            },
            "grep_files" => ToolCall::GrepFiles {
                query: extract_tag_value(&block, "query").unwrap_or_default(),
                base_path: extract_tag_value(&block, "base_path")
                    .unwrap_or_else(|| ".".to_string()),
                include_glob: extract_tag_value(&block, "include_glob")
                    .filter(|v| !v.trim().is_empty()),
                case_sensitive: parse_bool(
                    extract_tag_value(&block, "case_sensitive").as_deref(),
                    false,
                ),
                max_results: parse_usize(
                    extract_tag_value(&block, "max_results").as_deref(),
                    DEFAULT_GREP_MAX_RESULTS,
                )
                .max(1)
                .min(5_000),
            },
            "read_file" => ToolCall::ReadFile {
                path: extract_tag_value(&block, "path").unwrap_or_default(),
                start_line: parse_optional_usize(
                    extract_tag_value(&block, "start_line").as_deref(),
                ),
                end_line: parse_optional_usize(extract_tag_value(&block, "end_line").as_deref()),
            },
            "write_file" => ToolCall::WriteFile {
                path: extract_tag_value(&block, "path").unwrap_or_default(),
                content: extract_tag_value(&block, "content").unwrap_or_default(),
            },
            "append_file" => ToolCall::AppendFile {
                path: extract_tag_value(&block, "path").unwrap_or_default(),
                content: extract_tag_value(&block, "content").unwrap_or_default(),
            },
            "bash" => ToolCall::Bash {
                command: extract_tag_value(&block, "command").unwrap_or_default(),
                timeout_secs: parse_u64(
                    extract_tag_value(&block, "timeout_secs").as_deref(),
                    DEFAULT_BASH_TIMEOUT_SECS,
                )
                .clamp(1, MAX_BASH_TIMEOUT_SECS),
            },
            "mkdir" => ToolCall::Mkdir {
                path: extract_tag_value(&block, "path").unwrap_or_default(),
            },
            "move_path" => ToolCall::MovePath {
                from: extract_tag_value(&block, "from").unwrap_or_default(),
                to: extract_tag_value(&block, "to").unwrap_or_default(),
            },
            "delete_path" => ToolCall::DeletePath {
                path: extract_tag_value(&block, "path").unwrap_or_default(),
                recursive: parse_bool(extract_tag_value(&block, "recursive").as_deref(), false),
            },
            "apply_patch" => ToolCall::ApplyPatch {
                path: extract_tag_value(&block, "path").unwrap_or_default(),
                search: extract_tag_value(&block, "search").unwrap_or_default(),
                replace: extract_tag_value(&block, "replace").unwrap_or_default(),
                all: parse_bool(extract_tag_value(&block, "all").as_deref(), false),
            },
            "run_test" => ToolCall::RunTest {
                command: extract_tag_value(&block, "command").filter(|v| !v.trim().is_empty()),
                timeout_secs: parse_u64(
                    extract_tag_value(&block, "timeout_secs").as_deref(),
                    DEFAULT_BASH_TIMEOUT_SECS,
                )
                .clamp(1, MAX_BASH_TIMEOUT_SECS),
            },
            "run_lint" => ToolCall::RunLint {
                command: extract_tag_value(&block, "command").filter(|v| !v.trim().is_empty()),
                timeout_secs: parse_u64(
                    extract_tag_value(&block, "timeout_secs").as_deref(),
                    DEFAULT_BASH_TIMEOUT_SECS,
                )
                .clamp(1, MAX_BASH_TIMEOUT_SECS),
            },
            "bash_session_start" => ToolCall::BashSessionStart {
                path: extract_tag_value(&block, "path").unwrap_or_else(|| ".".to_string()),
                shell: extract_tag_value(&block, "shell").unwrap_or_else(|| "bash".to_string()),
            },
            "bash_session_run" => ToolCall::BashSessionRun {
                session_id: parse_optional_u64(extract_tag_value(&block, "session_id").as_deref())
                    .unwrap_or_default(),
                command: extract_tag_value(&block, "command").unwrap_or_default(),
                timeout_secs: parse_u64(
                    extract_tag_value(&block, "timeout_secs").as_deref(),
                    DEFAULT_BASH_TIMEOUT_SECS,
                )
                .clamp(1, MAX_BASH_TIMEOUT_SECS),
            },
            "bash_session_output" => ToolCall::BashSessionOutput {
                session_id: parse_optional_u64(extract_tag_value(&block, "session_id").as_deref())
                    .unwrap_or_default(),
                max_chars: parse_usize(
                    extract_tag_value(&block, "max_chars").as_deref(),
                    DEFAULT_SESSION_OUTPUT_MAX_CHARS,
                )
                .max(100)
                .min(100_000),
            },
            "bash_session_kill" => ToolCall::BashSessionKill {
                session_id: parse_optional_u64(extract_tag_value(&block, "session_id").as_deref())
                    .unwrap_or_default(),
            },
            "git_status" => ToolCall::GitStatus {
                short: parse_bool(extract_tag_value(&block, "short").as_deref(), true),
            },
            "git_diff" => ToolCall::GitDiff {
                staged: parse_bool(extract_tag_value(&block, "staged").as_deref(), false),
                path: extract_tag_value(&block, "path").filter(|v| !v.trim().is_empty()),
            },
            "git_add" => ToolCall::GitAdd {
                path: extract_tag_value(&block, "path").filter(|v| !v.trim().is_empty()),
                all: parse_bool(extract_tag_value(&block, "all").as_deref(), false),
            },
            "git_commit" => ToolCall::GitCommit {
                message: extract_tag_value(&block, "message").unwrap_or_default(),
            },
            _ => ToolCall::Unknown {
                requested: if tool_name.is_empty() {
                    "<missing>".to_string()
                } else {
                    tool_name
                },
            },
        };

        calls.push(call);
    }

    calls
}

fn extract_tool_call_blocks(input: &str) -> Vec<String> {
    let mut blocks = Vec::new();
    let mut cursor = 0usize;

    while let Some(open_rel) = input[cursor..].find("<tool_call") {
        let open_start = cursor + open_rel;
        let Some(open_end_rel) = input[open_start..].find('>') else {
            break;
        };
        let inner_start = open_start + open_end_rel + 1;

        let Some(close_rel) = input[inner_start..].find("</tool_call>") else {
            break;
        };
        let close_start = inner_start + close_rel;

        blocks.push(input[inner_start..close_start].to_string());
        cursor = close_start + "</tool_call>".len();
    }

    blocks
}

fn extract_tag_value(block: &str, tag: &str) -> Option<String> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");

    let start = block.find(&open)? + open.len();
    let end = start + block[start..].find(&close)?;

    let raw = block[start..end].trim();

    if let Some(cdata) = raw
        .strip_prefix("<![CDATA[")
        .and_then(|v| v.strip_suffix("]]>"))
    {
        return Some(cdata.to_string());
    }

    Some(unescape_xml(raw))
}

fn parse_bool(value: Option<&str>, default: bool) -> bool {
    let Some(raw) = value.map(str::trim).filter(|v| !v.is_empty()) else {
        return default;
    };

    matches!(
        raw.to_lowercase().as_str(),
        "true" | "1" | "yes" | "y" | "on"
    )
}

fn parse_usize(value: Option<&str>, default: usize) -> usize {
    value
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(default)
}

fn parse_optional_usize(value: Option<&str>) -> Option<usize> {
    value
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .and_then(|v| v.parse::<usize>().ok())
}

fn parse_optional_u64(value: Option<&str>) -> Option<u64> {
    value
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .and_then(|v| v.parse::<u64>().ok())
}

fn parse_u64(value: Option<&str>, default: u64) -> u64 {
    value
        .and_then(|v| v.trim().parse::<u64>().ok())
        .unwrap_or(default)
}

fn execute_tool_call(workspace: &Workspace, call: ToolCall) -> ToolResult {
    match call {
        ToolCall::ListFiles {
            path,
            recursive,
            max_entries,
        } => tool_list_files(workspace, &path, recursive, max_entries),
        ToolCall::GlobFiles {
            pattern,
            base_path,
            max_results,
        } => tool_glob_files(workspace, &pattern, &base_path, max_results),
        ToolCall::GrepFiles {
            query,
            base_path,
            include_glob,
            case_sensitive,
            max_results,
        } => tool_grep_files(
            workspace,
            &query,
            &base_path,
            include_glob.as_deref(),
            case_sensitive,
            max_results,
        ),
        ToolCall::ReadFile {
            path,
            start_line,
            end_line,
        } => tool_read_file(workspace, &path, start_line, end_line),
        ToolCall::WriteFile { path, content } => tool_write_file(workspace, &path, &content),
        ToolCall::AppendFile { path, content } => tool_append_file(workspace, &path, &content),
        ToolCall::Bash {
            command,
            timeout_secs,
        } => tool_bash(workspace, &command, timeout_secs),
        ToolCall::Mkdir { path } => tool_mkdir(workspace, &path),
        ToolCall::MovePath { from, to } => tool_move_path(workspace, &from, &to),
        ToolCall::DeletePath { path, recursive } => tool_delete_path(workspace, &path, recursive),
        ToolCall::ApplyPatch {
            path,
            search,
            replace,
            all,
        } => tool_apply_patch(workspace, &path, &search, &replace, all),
        ToolCall::RunTest {
            command,
            timeout_secs,
        } => tool_run_test(workspace, command.as_deref(), timeout_secs),
        ToolCall::RunLint {
            command,
            timeout_secs,
        } => tool_run_lint(workspace, command.as_deref(), timeout_secs),
        ToolCall::BashSessionStart { path, shell } => {
            tool_bash_session_start(workspace, &path, &shell)
        }
        ToolCall::BashSessionRun {
            session_id,
            command,
            timeout_secs,
        } => tool_bash_session_run(session_id, &command, timeout_secs),
        ToolCall::BashSessionOutput {
            session_id,
            max_chars,
        } => tool_bash_session_output(session_id, max_chars),
        ToolCall::BashSessionKill { session_id } => tool_bash_session_kill(session_id),
        ToolCall::GitStatus { short } => tool_git_status(workspace, short),
        ToolCall::GitDiff { staged, path } => tool_git_diff(workspace, staged, path.as_deref()),
        ToolCall::GitAdd { path, all } => tool_git_add(workspace, path.as_deref(), all),
        ToolCall::GitCommit { message } => tool_git_commit(workspace, &message),
        ToolCall::Unknown { requested } => ToolResult::error(
            "unknown",
            format!("unknown tool requested: {requested}"),
            "supported tools: list_files, glob_files, grep_files, read_file, write_file, append_file, mkdir, move_path, delete_path, apply_patch, bash, run_test, run_lint, bash_session_start, bash_session_run, bash_session_output, bash_session_kill, git_status, git_diff, git_add, git_commit".to_string(),
        ),
    }
}

fn tool_list_files(
    workspace: &Workspace,
    path: &str,
    recursive: bool,
    max_entries: usize,
) -> ToolResult {
    let result = (|| -> Result<(String, String)> {
        let resolved = workspace.resolve_for_read(path)?;
        let mut lines = Vec::new();

        if resolved.is_file() {
            lines.push(workspace.relative_display(&resolved));
        } else {
            let mut queue = VecDeque::new();
            queue.push_back(resolved.clone());
            let mut truncated = false;

            while let Some(dir) = queue.pop_front() {
                let mut entries = fs::read_dir(&dir)
                    .with_context(|| format!("failed to read directory {}", dir.display()))?
                    .collect::<std::result::Result<Vec<_>, _>>()
                    .with_context(|| format!("failed to list directory {}", dir.display()))?;

                entries.sort_by_key(|entry| entry.path());

                for entry in entries {
                    let entry_path = entry.path();
                    let mut display = workspace.relative_display(&entry_path);
                    if entry_path.is_dir() {
                        display.push('/');
                        if recursive {
                            queue.push_back(entry_path.clone());
                        }
                    }
                    lines.push(display);

                    if lines.len() >= max_entries {
                        truncated = true;
                        break;
                    }
                }

                if truncated {
                    break;
                }
                if !recursive {
                    break;
                }
            }

            if truncated {
                lines.push(format!("...truncated at {max_entries} entries"));
            }
        }

        let summary = format!(
            "listed {} entries from {}",
            lines.len(),
            workspace.relative_display(&resolved)
        );
        let output = if lines.is_empty() {
            "(no entries)".to_string()
        } else {
            lines.join("\n")
        };

        Ok((summary, output))
    })();

    match result {
        Ok((summary, output)) => ToolResult::ok("list_files", summary, output),
        Err(err) => ToolResult::error(
            "list_files",
            "failed to list files".to_string(),
            err.to_string(),
        ),
    }
}

fn tool_glob_files(
    workspace: &Workspace,
    pattern: &str,
    base_path: &str,
    max_results: usize,
) -> ToolResult {
    let result = (|| -> Result<(String, String)> {
        let trimmed_pattern = pattern.trim();
        if trimmed_pattern.is_empty() {
            bail!("pattern is required");
        }

        let matcher = Pattern::new(trimmed_pattern)
            .with_context(|| format!("invalid glob pattern: {trimmed_pattern}"))?;
        let root = workspace.resolve_for_read(base_path)?;

        let mut entries = Vec::new();
        collect_workspace_entries(&root, true, &mut entries)?;
        entries.sort();

        let mut matches = Vec::new();
        for entry in entries {
            let rel = workspace.relative_display(&entry);
            if rel == "." {
                continue;
            }
            if matcher.matches(&rel.replace('\\', "/")) {
                let mut shown = rel;
                if entry.is_dir() {
                    shown.push('/');
                }
                matches.push(shown);
                if matches.len() >= max_results {
                    break;
                }
            }
        }

        if matches.is_empty() {
            return Ok((
                format!("no matches for pattern {trimmed_pattern}"),
                "(no matches)".to_string(),
            ));
        }

        Ok((
            format!(
                "matched {} path(s) for pattern {}",
                matches.len(),
                trimmed_pattern
            ),
            matches.join("\n"),
        ))
    })();

    match result {
        Ok((summary, output)) => ToolResult::ok("glob_files", summary, output),
        Err(err) => ToolResult::error(
            "glob_files",
            "failed to glob files".to_string(),
            err.to_string(),
        ),
    }
}

fn tool_grep_files(
    workspace: &Workspace,
    query: &str,
    base_path: &str,
    include_glob: Option<&str>,
    case_sensitive: bool,
    max_results: usize,
) -> ToolResult {
    let result = (|| -> Result<(String, String)> {
        let needle = query.trim();
        if needle.is_empty() {
            bail!("query is required");
        }

        let matcher = if let Some(pattern) = include_glob.map(str::trim).filter(|v| !v.is_empty()) {
            Some(
                Pattern::new(pattern)
                    .with_context(|| format!("invalid include_glob pattern: {pattern}"))?,
            )
        } else {
            None
        };

        let root = workspace.resolve_for_read(base_path)?;
        let mut files = Vec::new();
        collect_workspace_files(&root, true, &mut files)?;
        files.sort();

        let needle_cmp = if case_sensitive {
            needle.to_string()
        } else {
            needle.to_lowercase()
        };

        let mut hits = Vec::new();
        for file in files {
            let rel = workspace.relative_display(&file);
            if let Some(glob_matcher) = &matcher {
                if !glob_matcher.matches(&rel.replace('\\', "/")) {
                    continue;
                }
            }

            let content = match fs::read_to_string(&file) {
                Ok(v) => v,
                Err(_) => continue,
            };

            for (idx, line) in content.lines().enumerate() {
                let matched = if case_sensitive {
                    line.contains(needle)
                } else {
                    line.to_lowercase().contains(&needle_cmp)
                };

                if matched {
                    hits.push(format!("{}:{}: {}", rel, idx + 1, truncate_line(line, 240)));
                    if hits.len() >= max_results {
                        break;
                    }
                }
            }

            if hits.len() >= max_results {
                break;
            }
        }

        if hits.is_empty() {
            return Ok((
                format!("no matches for query {:?}", needle),
                "(no matches)".to_string(),
            ));
        }

        Ok((
            format!("found {} match(es) for query {:?}", hits.len(), needle),
            hits.join("\n"),
        ))
    })();

    match result {
        Ok((summary, output)) => ToolResult::ok("grep_files", summary, output),
        Err(err) => ToolResult::error(
            "grep_files",
            "failed to grep files".to_string(),
            err.to_string(),
        ),
    }
}

fn tool_read_file(
    workspace: &Workspace,
    path: &str,
    start_line: Option<usize>,
    end_line: Option<usize>,
) -> ToolResult {
    let result = (|| -> Result<(String, String)> {
        let resolved = workspace.resolve_for_read(path)?;
        if resolved.is_dir() {
            bail!("{} is a directory", workspace.relative_display(&resolved));
        }

        let content = fs::read_to_string(&resolved)
            .with_context(|| format!("failed to read {} as UTF-8 text", resolved.display()))?;

        let lines = content.lines().collect::<Vec<_>>();
        if lines.is_empty() {
            let summary = format!(
                "read {} (empty file)",
                workspace.relative_display(&resolved)
            );
            return Ok((summary, "(empty file)\n".to_string()));
        }

        let total_lines = lines.len();
        let start = start_line.unwrap_or(1).max(1);
        let end = end_line.unwrap_or(lines.len()).max(start);

        if start > total_lines {
            bail!("start_line {} exceeds total lines {}", start, lines.len());
        }

        let end = end.min(lines.len());
        if start > end {
            bail!("invalid line range: start_line must be <= end_line");
        }

        let mut rendered = String::new();
        for (idx, line) in lines[start - 1..end].iter().enumerate() {
            rendered.push_str(&format!("{:>5} | {}\n", start + idx, line));
        }
        let summary = format!(
            "read {} (lines {}-{} of {})",
            workspace.relative_display(&resolved),
            start,
            end,
            lines.len()
        );

        Ok((summary, rendered))
    })();

    match result {
        Ok((summary, output)) => ToolResult::ok("read_file", summary, output),
        Err(err) => ToolResult::error(
            "read_file",
            "failed to read file".to_string(),
            err.to_string(),
        ),
    }
}

fn tool_write_file(workspace: &Workspace, path: &str, content: &str) -> ToolResult {
    let result = (|| -> Result<String> {
        let resolved = workspace.resolve_for_write(path)?;
        if resolved.is_dir() {
            bail!("{} is a directory", workspace.relative_display(&resolved));
        }

        let parent = resolved
            .parent()
            .ok_or_else(|| anyhow!("invalid path {}", resolved.display()))?;
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create parent directory {}", parent.display()))?;

        fs::write(&resolved, content)
            .with_context(|| format!("failed to write {}", resolved.display()))?;

        Ok(format!(
            "wrote {} bytes to {}",
            content.len(),
            workspace.relative_display(&resolved)
        ))
    })();

    match result {
        Ok(summary) => ToolResult::ok("write_file", summary, "ok".to_string()),
        Err(err) => ToolResult::error(
            "write_file",
            "failed to write file".to_string(),
            err.to_string(),
        ),
    }
}

fn tool_append_file(workspace: &Workspace, path: &str, content: &str) -> ToolResult {
    let result = (|| -> Result<String> {
        let resolved = workspace.resolve_for_write(path)?;
        if resolved.is_dir() {
            bail!("{} is a directory", workspace.relative_display(&resolved));
        }

        let parent = resolved
            .parent()
            .ok_or_else(|| anyhow!("invalid path {}", resolved.display()))?;
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create parent directory {}", parent.display()))?;

        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&resolved)
            .with_context(|| format!("failed to open {} for append", resolved.display()))?;
        file.write_all(content.as_bytes())
            .with_context(|| format!("failed to append to {}", resolved.display()))?;

        Ok(format!(
            "appended {} bytes to {}",
            content.len(),
            workspace.relative_display(&resolved)
        ))
    })();

    match result {
        Ok(summary) => ToolResult::ok("append_file", summary, "ok".to_string()),
        Err(err) => ToolResult::error(
            "append_file",
            "failed to append file".to_string(),
            err.to_string(),
        ),
    }
}

fn tool_mkdir(workspace: &Workspace, path: &str) -> ToolResult {
    let result = (|| -> Result<String> {
        let resolved = workspace.resolve_for_write(path)?;
        if resolved.exists() && !resolved.is_dir() {
            bail!("{} exists and is not a directory", resolved.display());
        }
        fs::create_dir_all(&resolved)
            .with_context(|| format!("failed to create directory {}", resolved.display()))?;
        Ok(format!(
            "created directory {}",
            workspace.relative_display(&resolved)
        ))
    })();

    match result {
        Ok(summary) => ToolResult::ok("mkdir", summary, "ok".to_string()),
        Err(err) => ToolResult::error(
            "mkdir",
            "failed to create directory".to_string(),
            err.to_string(),
        ),
    }
}

fn tool_move_path(workspace: &Workspace, from: &str, to: &str) -> ToolResult {
    let result = (|| -> Result<String> {
        let source = workspace.resolve_for_read(from)?;
        if source == workspace.root_path() {
            bail!("cannot move workspace root");
        }

        let destination = workspace.resolve_for_write(to)?;
        if destination.exists() {
            bail!("destination already exists: {}", destination.display());
        }
        let parent = destination
            .parent()
            .ok_or_else(|| anyhow!("invalid destination path {}", destination.display()))?;
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create directory {}", parent.display()))?;

        fs::rename(&source, &destination).with_context(|| {
            format!(
                "failed to move {} -> {}",
                source.display(),
                destination.display()
            )
        })?;

        Ok(format!(
            "moved {} -> {}",
            workspace.relative_display(&source),
            workspace.relative_display(&destination)
        ))
    })();

    match result {
        Ok(summary) => ToolResult::ok("move_path", summary, "ok".to_string()),
        Err(err) => ToolResult::error(
            "move_path",
            "failed to move path".to_string(),
            err.to_string(),
        ),
    }
}

fn tool_delete_path(workspace: &Workspace, path: &str, recursive: bool) -> ToolResult {
    let result = (|| -> Result<String> {
        let target = workspace.resolve_for_read(path)?;
        if target == workspace.root_path() {
            bail!("cannot delete workspace root");
        }

        if target.is_file() {
            fs::remove_file(&target)
                .with_context(|| format!("failed to delete file {}", target.display()))?;
            return Ok(format!(
                "deleted file {}",
                workspace.relative_display(&target)
            ));
        }

        if target.is_dir() {
            if recursive {
                fs::remove_dir_all(&target)
                    .with_context(|| format!("failed to delete directory {}", target.display()))?;
            } else {
                fs::remove_dir(&target).with_context(|| {
                    format!(
                        "failed to delete directory {} (set recursive=true if non-empty)",
                        target.display()
                    )
                })?;
            }
            return Ok(format!(
                "deleted directory {}",
                workspace.relative_display(&target)
            ));
        }

        bail!("unsupported path type: {}", target.display())
    })();

    match result {
        Ok(summary) => ToolResult::ok("delete_path", summary, "ok".to_string()),
        Err(err) => ToolResult::error(
            "delete_path",
            "failed to delete path".to_string(),
            err.to_string(),
        ),
    }
}

fn tool_apply_patch(
    workspace: &Workspace,
    path: &str,
    search: &str,
    replace: &str,
    replace_all: bool,
) -> ToolResult {
    let result = (|| -> Result<String> {
        let resolved = workspace.resolve_for_read(path)?;
        if resolved.is_dir() {
            bail!("{} is a directory", workspace.relative_display(&resolved));
        }
        let needle = search;
        if needle.is_empty() {
            bail!("search content is required");
        }

        let original = fs::read_to_string(&resolved)
            .with_context(|| format!("failed to read {}", resolved.display()))?;
        let occurrences = original.matches(needle).count();
        if occurrences == 0 {
            bail!(
                "search text not found in {}",
                workspace.relative_display(&resolved)
            );
        }

        let updated = if replace_all {
            original.replace(needle, replace)
        } else {
            original.replacen(needle, replace, 1)
        };

        fs::write(&resolved, updated)
            .with_context(|| format!("failed to write {}", resolved.display()))?;

        let applied = if replace_all { occurrences } else { 1 };
        Ok(format!(
            "applied patch to {} ({} replacement{})",
            workspace.relative_display(&resolved),
            applied,
            if applied == 1 { "" } else { "s" }
        ))
    })();

    match result {
        Ok(summary) => ToolResult::ok("apply_patch", summary, "ok".to_string()),
        Err(err) => ToolResult::error(
            "apply_patch",
            "failed to apply patch".to_string(),
            err.to_string(),
        ),
    }
}

fn tool_bash(workspace: &Workspace, command: &str, timeout_secs: u64) -> ToolResult {
    let result = (|| -> Result<CommandOutcome> {
        let trimmed = command.trim();
        if trimmed.is_empty() {
            bail!("command cannot be empty");
        }

        run_process_capture(
            "bash",
            &["-lc", trimmed],
            workspace.root_path(),
            timeout_secs.clamp(1, MAX_BASH_TIMEOUT_SECS),
        )
    })();

    match result {
        Ok(outcome) => {
            let code = outcome
                .exit_code
                .map(|v| v.to_string())
                .unwrap_or_else(|| "signal".to_string());
            let summary = if outcome.timed_out {
                format!(
                    "bash timed out after {}s",
                    timeout_secs.clamp(1, MAX_BASH_TIMEOUT_SECS)
                )
            } else if outcome.success {
                format!("bash exited with code {code}")
            } else {
                format!("bash failed with code {code}")
            };

            if outcome.success {
                ToolResult::ok("bash", summary, outcome.output)
            } else {
                ToolResult::error("bash", summary, outcome.output)
            }
        }
        Err(err) => ToolResult::error("bash", "failed to run bash".to_string(), err.to_string()),
    }
}

fn tool_run_test(workspace: &Workspace, command: Option<&str>, timeout_secs: u64) -> ToolResult {
    let chosen = command
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .map(|v| v.to_string())
        .or_else(|| infer_test_command(workspace));

    let Some(cmd) = chosen else {
        return ToolResult::error(
            "run_test",
            "failed to choose test command".to_string(),
            "no test command provided and no known project type found; provide <command> explicitly".to_string(),
        );
    };

    let mut result = tool_bash(workspace, &cmd, timeout_secs);
    result.tool = "run_test".to_string();
    if result.status == "ok" {
        result.summary = format!("test command succeeded: {cmd}");
    } else {
        result.summary = format!("test command failed: {cmd}");
    }
    result
}

fn tool_run_lint(workspace: &Workspace, command: Option<&str>, timeout_secs: u64) -> ToolResult {
    let chosen = command
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .map(|v| v.to_string())
        .or_else(|| infer_lint_command(workspace));

    let Some(cmd) = chosen else {
        return ToolResult::error(
            "run_lint",
            "failed to choose lint command".to_string(),
            "no lint command provided and no known project type found; provide <command> explicitly".to_string(),
        );
    };

    let mut result = tool_bash(workspace, &cmd, timeout_secs);
    result.tool = "run_lint".to_string();
    if result.status == "ok" {
        result.summary = format!("lint command succeeded: {cmd}");
    } else {
        result.summary = format!("lint command failed: {cmd}");
    }
    result
}

fn tool_bash_session_start(workspace: &Workspace, path: &str, shell: &str) -> ToolResult {
    let result = (|| -> Result<(String, String)> {
        let dir = workspace.resolve_for_read(path)?;
        if !dir.is_dir() {
            bail!("{} is not a directory", workspace.relative_display(&dir));
        }

        let shell_bin = if shell.trim().is_empty() {
            "bash"
        } else {
            shell.trim()
        };

        let mut child = Command::new(shell_bin)
            .arg("--noprofile")
            .arg("--norc")
            .current_dir(&dir)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .with_context(|| format!("failed to spawn shell {shell_bin}"))?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| anyhow!("failed to capture shell stdin"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow!("failed to capture shell stdout"))?;
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| anyhow!("failed to capture shell stderr"))?;

        let stdout_buf = Arc::new(Mutex::new(String::new()));
        let stderr_buf = Arc::new(Mutex::new(String::new()));
        spawn_stream_reader(stdout, Arc::clone(&stdout_buf));
        spawn_stream_reader(stderr, Arc::clone(&stderr_buf));

        let id = NEXT_SESSION_ID.fetch_add(1, Ordering::Relaxed);
        let mut registry = session_registry()
            .lock()
            .map_err(|_| anyhow!("failed to lock shell sessions"))?;
        registry.insert(
            id,
            ShellSession {
                child,
                stdin,
                stdout_buf,
                stderr_buf,
            },
        );

        Ok((
            format!(
                "started shell session {id} in {}",
                workspace.relative_display(&dir)
            ),
            format!(
                "session_id: {id}\nshell: {shell_bin}\ncwd: {}",
                workspace.relative_display(&dir)
            ),
        ))
    })();

    match result {
        Ok((summary, output)) => ToolResult::ok("bash_session_start", summary, output),
        Err(err) => ToolResult::error(
            "bash_session_start",
            "failed to start shell session".to_string(),
            err.to_string(),
        ),
    }
}

fn tool_bash_session_run(session_id: u64, command: &str, timeout_secs: u64) -> ToolResult {
    let result = (|| -> Result<(String, String, bool)> {
        if session_id == 0 {
            bail!("session_id is required");
        }

        let trimmed = command.trim();
        if trimmed.is_empty() {
            bail!("command cannot be empty");
        }

        let timeout = timeout_secs.clamp(1, MAX_BASH_TIMEOUT_SECS);
        let marker = Uuid::new_v4().simple().to_string();
        let start_marker = format!("__NANOGPT_START_{marker}__");
        let end_prefix = format!("__NANOGPT_END_{marker}__:");

        let mut registry = session_registry()
            .lock()
            .map_err(|_| anyhow!("failed to lock shell sessions"))?;
        let session = registry
            .get_mut(&session_id)
            .ok_or_else(|| anyhow!("unknown session_id: {session_id}"))?;

        if let Some(status) = session
            .child
            .try_wait()
            .with_context(|| format!("failed to check session {session_id}"))?
        {
            bail!(
                "session {session_id} has exited (code: {:?})",
                status.code()
            );
        }

        let stdout_before = session.stdout_len();
        let stderr_before = session.stderr_len();
        let script = format!(
            "printf '%s\\n' '{start_marker}'\n{trimmed}\n__ec=$?\nprintf '%s%s\\n' '{end_prefix}' \"$__ec\"\n"
        );
        session
            .stdin
            .write_all(script.as_bytes())
            .with_context(|| format!("failed to write to session {session_id}"))?;
        session
            .stdin
            .flush()
            .with_context(|| format!("failed to flush session {session_id}"))?;

        let started = Instant::now();
        loop {
            if started.elapsed() >= Duration::from_secs(timeout) {
                let _ = session.child.kill();
                let _ = session.child.wait();
                return Ok((
                    format!("session {session_id} command timed out after {timeout}s"),
                    "session terminated due to timeout".to_string(),
                    true,
                ));
            }

            if let Some(status) = session
                .child
                .try_wait()
                .with_context(|| format!("failed while waiting for session {session_id}"))?
            {
                return Ok((
                    format!(
                        "session {session_id} exited while running command (code: {:?})",
                        status.code()
                    ),
                    "(session exited)".to_string(),
                    true,
                ));
            }

            let stdout_full = session.stdout_snapshot();
            if stdout_full.len() >= stdout_before {
                let delta = &stdout_full[stdout_before..];
                if let Some(end_idx) = delta.find(&end_prefix) {
                    let start_idx = delta
                        .find(&start_marker)
                        .map(|idx| {
                            let mut after = idx + start_marker.len();
                            if delta[after..].starts_with('\n') {
                                after += 1;
                            }
                            after
                        })
                        .unwrap_or(0);

                    let stdout_payload = delta[start_idx..end_idx].to_string();
                    let code_start = end_idx + end_prefix.len();
                    let code_end = delta[code_start..]
                        .find('\n')
                        .map(|i| code_start + i)
                        .unwrap_or(delta.len());
                    let exit_code = delta[code_start..code_end]
                        .trim()
                        .parse::<i32>()
                        .unwrap_or(1);

                    let stderr_full = session.stderr_snapshot();
                    let stderr_payload = if stderr_full.len() >= stderr_before {
                        stderr_full[stderr_before..].to_string()
                    } else {
                        String::new()
                    };

                    let output = merge_stdio(&stdout_payload, &stderr_payload);
                    return Ok((
                        format!("session {session_id} exited with code {exit_code}"),
                        output,
                        exit_code != 0,
                    ));
                }
            }

            std::thread::sleep(Duration::from_millis(25));
        }
    })();

    match result {
        Ok((summary, output, failed)) => {
            if failed {
                ToolResult::error("bash_session_run", summary, output)
            } else {
                ToolResult::ok("bash_session_run", summary, output)
            }
        }
        Err(err) => ToolResult::error(
            "bash_session_run",
            "failed to run session command".to_string(),
            err.to_string(),
        ),
    }
}

fn tool_bash_session_output(session_id: u64, max_chars: usize) -> ToolResult {
    let result = (|| -> Result<(String, String)> {
        if session_id == 0 {
            bail!("session_id is required");
        }

        let mut registry = session_registry()
            .lock()
            .map_err(|_| anyhow!("failed to lock shell sessions"))?;
        let session = registry
            .get_mut(&session_id)
            .ok_or_else(|| anyhow!("unknown session_id: {session_id}"))?;

        let state = if let Some(status) = session
            .child
            .try_wait()
            .with_context(|| format!("failed to inspect session {session_id}"))?
        {
            format!("exited({:?})", status.code())
        } else {
            "running".to_string()
        };

        let stdout = session.stdout_snapshot();
        let stderr = session.stderr_snapshot();
        let merged = merge_stdio(&stdout, &stderr);
        let shown = tail_text(&merged, max_chars.max(100));

        Ok((
            format!("session {session_id} output ({state})"),
            if shown.trim().is_empty() {
                "(no output)".to_string()
            } else {
                shown
            },
        ))
    })();

    match result {
        Ok((summary, output)) => ToolResult::ok("bash_session_output", summary, output),
        Err(err) => ToolResult::error(
            "bash_session_output",
            "failed to get session output".to_string(),
            err.to_string(),
        ),
    }
}

fn tool_bash_session_kill(session_id: u64) -> ToolResult {
    let result = (|| -> Result<String> {
        if session_id == 0 {
            bail!("session_id is required");
        }

        let mut registry = session_registry()
            .lock()
            .map_err(|_| anyhow!("failed to lock shell sessions"))?;
        let mut session = registry
            .remove(&session_id)
            .ok_or_else(|| anyhow!("unknown session_id: {session_id}"))?;

        let _ = session.child.kill();
        let _ = session.child.wait();
        Ok(format!("killed session {session_id}"))
    })();

    match result {
        Ok(summary) => ToolResult::ok("bash_session_kill", summary, "ok".to_string()),
        Err(err) => ToolResult::error(
            "bash_session_kill",
            "failed to kill session".to_string(),
            err.to_string(),
        ),
    }
}

fn tool_git_status(workspace: &Workspace, short: bool) -> ToolResult {
    let args = if short {
        vec!["status", "--short", "--branch"]
    } else {
        vec!["status"]
    };

    tool_git_command(workspace, "git_status", &args, 30)
}

fn tool_git_diff(workspace: &Workspace, staged: bool, path: Option<&str>) -> ToolResult {
    let mut owned_args = vec!["diff".to_string()];
    if staged {
        owned_args.push("--staged".to_string());
    }
    if let Some(path_value) = path {
        let rel = match workspace.relative_git_path(path_value) {
            Ok(v) => v,
            Err(err) => {
                return ToolResult::error(
                    "git_diff",
                    "invalid git diff path".to_string(),
                    err.to_string(),
                );
            }
        };
        owned_args.push("--".to_string());
        owned_args.push(rel);
    }

    let args = owned_args.iter().map(|s| s.as_str()).collect::<Vec<_>>();
    tool_git_command(workspace, "git_diff", &args, 30)
}

fn tool_git_add(workspace: &Workspace, path: Option<&str>, all: bool) -> ToolResult {
    let mut owned_args = vec!["add".to_string()];
    if all {
        owned_args.push("-A".to_string());
    } else if let Some(path_value) = path {
        let rel = match workspace.relative_git_path(path_value) {
            Ok(v) => v,
            Err(err) => {
                return ToolResult::error(
                    "git_add",
                    "invalid git add path".to_string(),
                    err.to_string(),
                );
            }
        };
        owned_args.push(rel);
    } else {
        return ToolResult::error(
            "git_add",
            "invalid git add arguments".to_string(),
            "provide <path> or set <all>true</all>".to_string(),
        );
    }

    let args = owned_args.iter().map(|s| s.as_str()).collect::<Vec<_>>();
    tool_git_command(workspace, "git_add", &args, 30)
}

fn tool_git_commit(workspace: &Workspace, message: &str) -> ToolResult {
    let msg = message.trim();
    if msg.is_empty() {
        return ToolResult::error(
            "git_commit",
            "invalid commit message".to_string(),
            "message cannot be empty".to_string(),
        );
    }

    let args = ["commit", "-m", msg];
    tool_git_command(workspace, "git_commit", &args, 60)
}

fn tool_git_command(
    workspace: &Workspace,
    tool_name: &str,
    args: &[&str],
    timeout_secs: u64,
) -> ToolResult {
    let result = (|| -> Result<CommandOutcome> {
        run_process_capture("git", args, workspace.root_path(), timeout_secs)
    })();

    match result {
        Ok(outcome) => {
            let code = outcome
                .exit_code
                .map(|v| v.to_string())
                .unwrap_or_else(|| "signal".to_string());
            let summary = if outcome.timed_out {
                format!("git command timed out after {timeout_secs}s")
            } else if outcome.success {
                format!("git command succeeded (code {code})")
            } else {
                format!("git command failed (code {code})")
            };

            if outcome.success {
                ToolResult::ok(tool_name, summary, outcome.output)
            } else {
                ToolResult::error(tool_name, summary, outcome.output)
            }
        }
        Err(err) => ToolResult::error(
            tool_name,
            "failed to run git command".to_string(),
            err.to_string(),
        ),
    }
}

impl ToolResult {
    fn ok(tool: &str, summary: String, output: String) -> Self {
        Self {
            tool: tool.to_string(),
            status: "ok",
            summary,
            output: truncate_text(&output, MAX_TOOL_OUTPUT_CHARS),
        }
    }

    fn error(tool: &str, summary: String, output: String) -> Self {
        Self {
            tool: tool.to_string(),
            status: "error",
            summary,
            output: truncate_text(&output, MAX_TOOL_OUTPUT_CHARS),
        }
    }
}

impl Workspace {
    fn new(root: PathBuf) -> Result<Self> {
        let canonical_root = root
            .canonicalize()
            .with_context(|| format!("failed to resolve workspace root {}", root.display()))?;

        if !canonical_root.is_dir() {
            bail!(
                "workspace root is not a directory: {}",
                canonical_root.display()
            );
        }

        Ok(Self {
            root: canonical_root,
        })
    }

    fn resolve_for_read(&self, user_path: &str) -> Result<PathBuf> {
        let candidate = self.normalize_user_path(user_path)?;
        if !candidate.exists() {
            bail!("path does not exist: {}", candidate.display());
        }

        let canonical = candidate
            .canonicalize()
            .with_context(|| format!("failed to resolve {}", candidate.display()))?;
        self.ensure_within_root(&canonical)?;
        Ok(canonical)
    }

    fn resolve_for_write(&self, user_path: &str) -> Result<PathBuf> {
        let candidate = self.normalize_user_path(user_path)?;

        let (ancestor, tail) = nearest_existing_ancestor(&candidate)?;
        let canonical_ancestor = ancestor
            .canonicalize()
            .with_context(|| format!("failed to resolve {}", ancestor.display()))?;
        self.ensure_within_root(&canonical_ancestor)?;

        if tail.as_os_str().is_empty() {
            return Ok(canonical_ancestor);
        }

        Ok(canonical_ancestor.join(tail))
    }

    fn normalize_user_path(&self, user_path: &str) -> Result<PathBuf> {
        let trimmed = user_path.trim();
        if trimmed.is_empty() {
            bail!("path is required");
        }

        let base = if Path::new(trimmed).is_absolute() {
            PathBuf::from(trimmed)
        } else {
            self.root.join(trimmed)
        };

        Ok(normalize_lexical_path(base))
    }

    fn ensure_within_root(&self, candidate: &Path) -> Result<()> {
        if candidate == self.root || candidate.starts_with(&self.root) {
            Ok(())
        } else {
            bail!("path is outside workspace root {}", self.root.display())
        }
    }

    fn relative_display(&self, path: &Path) -> String {
        if path == self.root {
            return ".".to_string();
        }

        path.strip_prefix(&self.root)
            .map(|p| p.display().to_string())
            .unwrap_or_else(|_| path.display().to_string())
    }

    fn root_path(&self) -> &Path {
        &self.root
    }

    fn has_root_file(&self, filename: &str) -> bool {
        self.root.join(filename).is_file()
    }

    fn has_root_any(&self, filename: &str) -> bool {
        self.root.join(filename).exists()
    }

    fn relative_git_path(&self, user_path: &str) -> Result<String> {
        let resolved = self.resolve_for_write(user_path)?;
        Ok(self.relative_display(&resolved))
    }
}

fn nearest_existing_ancestor(path: &Path) -> Result<(PathBuf, PathBuf)> {
    let mut current = path.to_path_buf();
    let mut suffix: Vec<OsString> = Vec::new();

    while !current.exists() {
        let name = current
            .file_name()
            .ok_or_else(|| anyhow!("invalid path {}", path.display()))?;
        suffix.push(name.to_os_string());
        current = current
            .parent()
            .ok_or_else(|| anyhow!("invalid path {}", path.display()))?
            .to_path_buf();
    }

    suffix.reverse();

    let mut tail = PathBuf::new();
    for component in suffix {
        tail.push(component);
    }

    Ok((current, tail))
}

fn normalize_lexical_path(path: PathBuf) -> PathBuf {
    let mut normalized = PathBuf::new();

    for component in path.components() {
        match component {
            Component::Prefix(prefix) => normalized.push(prefix.as_os_str()),
            Component::RootDir => normalized.push(Path::new(std::path::MAIN_SEPARATOR_STR)),
            Component::CurDir => {}
            Component::ParentDir => {
                let _ = normalized.pop();
            }
            Component::Normal(segment) => normalized.push(segment),
        }
    }

    normalized
}

fn collect_workspace_entries(
    path: &Path,
    recursive: bool,
    output: &mut Vec<PathBuf>,
) -> Result<()> {
    if path.is_file() {
        output.push(path.to_path_buf());
        return Ok(());
    }

    let mut queue = VecDeque::new();
    queue.push_back(path.to_path_buf());

    while let Some(dir) = queue.pop_front() {
        let entries = fs::read_dir(&dir)
            .with_context(|| format!("failed to read directory {}", dir.display()))?;
        for entry in entries {
            let entry = entry.with_context(|| format!("failed to read in {}", dir.display()))?;
            let p = entry.path();
            output.push(p.clone());
            if recursive && p.is_dir() {
                queue.push_back(p);
            }
        }

        if !recursive {
            break;
        }
    }

    Ok(())
}

fn collect_workspace_files(path: &Path, recursive: bool, output: &mut Vec<PathBuf>) -> Result<()> {
    let mut entries = Vec::new();
    collect_workspace_entries(path, recursive, &mut entries)?;
    output.extend(entries.into_iter().filter(|p| p.is_file()));
    Ok(())
}

fn truncate_line(line: &str, max_chars: usize) -> String {
    if line.chars().count() <= max_chars {
        return line.to_string();
    }
    let mut out = line.chars().take(max_chars).collect::<String>();
    out.push_str("...");
    out
}

fn infer_test_command(workspace: &Workspace) -> Option<String> {
    if workspace.has_root_file("Cargo.toml") {
        return Some("cargo test -q".to_string());
    }
    if workspace.has_root_file("package.json") {
        return Some("npm test --silent".to_string());
    }
    if workspace.has_root_file("pyproject.toml") || workspace.has_root_any("pytest.ini") {
        return Some("pytest -q".to_string());
    }
    None
}

fn infer_lint_command(workspace: &Workspace) -> Option<String> {
    if workspace.has_root_file("Cargo.toml") {
        return Some("cargo clippy --all-targets --all-features -- -D warnings".to_string());
    }
    if workspace.has_root_file("package.json") {
        return Some("npm run lint --silent".to_string());
    }
    if workspace.has_root_file("pyproject.toml") {
        return Some("ruff check .".to_string());
    }
    None
}

fn session_registry() -> &'static Mutex<BTreeMap<u64, ShellSession>> {
    SHELL_SESSIONS.get_or_init(|| Mutex::new(BTreeMap::new()))
}

fn spawn_stream_reader<R>(mut reader: R, sink: Arc<Mutex<String>>)
where
    R: Read + Send + 'static,
{
    std::thread::spawn(move || {
        let mut chunk = [0u8; 4096];
        loop {
            match reader.read(&mut chunk) {
                Ok(0) => break,
                Ok(n) => {
                    let text = String::from_utf8_lossy(&chunk[..n]);
                    if let Ok(mut buf) = sink.lock() {
                        buf.push_str(&text);
                    } else {
                        break;
                    }
                }
                Err(_) => break,
            }
        }
    });
}

fn run_process_capture(
    program: &str,
    args: &[&str],
    cwd: &Path,
    timeout_secs: u64,
) -> Result<CommandOutcome> {
    let timeout = timeout_secs.clamp(1, MAX_BASH_TIMEOUT_SECS);
    let stamp = Uuid::new_v4();
    let stdout_path = std::env::temp_dir().join(format!("nanogpt-cli-agent-{stamp}.out"));
    let stderr_path = std::env::temp_dir().join(format!("nanogpt-cli-agent-{stamp}.err"));

    let stdout_file = fs::File::create(&stdout_path)
        .with_context(|| format!("failed to create {}", stdout_path.display()))?;
    let stderr_file = fs::File::create(&stderr_path)
        .with_context(|| format!("failed to create {}", stderr_path.display()))?;

    let mut child = Command::new(program)
        .args(args)
        .current_dir(cwd)
        .stdout(Stdio::from(stdout_file))
        .stderr(Stdio::from(stderr_file))
        .spawn()
        .with_context(|| format!("failed to spawn command: {} {:?}", program, args))?;

    let started = Instant::now();
    let mut timed_out = false;
    let status = loop {
        if let Some(status) = child
            .try_wait()
            .with_context(|| format!("failed while waiting for command: {program}"))?
        {
            break status;
        }

        if started.elapsed() >= Duration::from_secs(timeout) {
            timed_out = true;
            let _ = child.kill();
            break child
                .wait()
                .with_context(|| format!("failed to collect timed out process: {program}"))?;
        }

        std::thread::sleep(Duration::from_millis(40));
    };

    let stdout = fs::read_to_string(&stdout_path).unwrap_or_default();
    let stderr = fs::read_to_string(&stderr_path).unwrap_or_default();
    let _ = fs::remove_file(&stdout_path);
    let _ = fs::remove_file(&stderr_path);

    let output = merge_stdio(&stdout, &stderr);
    Ok(CommandOutcome {
        timed_out,
        success: !timed_out && status.success(),
        exit_code: status.code(),
        output,
    })
}

fn merge_stdio(stdout: &str, stderr: &str) -> String {
    let mut output = String::new();

    if !stdout.trim().is_empty() {
        output.push_str("stdout:\n");
        output.push_str(stdout);
        if !stdout.ends_with('\n') {
            output.push('\n');
        }
    }

    if !stderr.trim().is_empty() {
        if !output.is_empty() {
            output.push('\n');
        }
        output.push_str("stderr:\n");
        output.push_str(stderr);
        if !stderr.ends_with('\n') {
            output.push('\n');
        }
    }

    if output.is_empty() {
        "(no output)".to_string()
    } else {
        output
    }
}

fn tail_text(input: &str, max_chars: usize) -> String {
    if input.chars().count() <= max_chars {
        return input.to_string();
    }

    let tail = input
        .chars()
        .rev()
        .take(max_chars)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect::<String>();
    format!("...[truncated]\n{tail}")
}

fn render_tool_results_prompt(results: &[ToolResult]) -> String {
    let mut output = String::from("<tool_results>\n");

    for result in results {
        output.push_str("  <tool_result>\n");
        output.push_str(&format!("    <tool>{}</tool>\n", escape_xml(&result.tool)));
        output.push_str(&format!(
            "    <status>{}</status>\n",
            escape_xml(result.status)
        ));
        output.push_str(&format!(
            "    <summary>{}</summary>\n",
            escape_xml(&result.summary)
        ));
        output.push_str(&format!(
            "    <output><![CDATA[{}]]></output>\n",
            escape_cdata(&result.output)
        ));
        output.push_str("  </tool_result>\n");
    }

    output.push_str("</tool_results>\n\n");
    output.push_str("If you need more tools, respond with <tool_call> blocks only. If done, answer the user normally.");
    output
}

fn truncate_text(input: &str, max_chars: usize) -> String {
    if input.chars().count() <= max_chars {
        return input.to_string();
    }

    let mut out = input.chars().take(max_chars).collect::<String>();
    out.push_str("\n...[truncated]");
    out
}

fn escape_xml(input: &str) -> String {
    input
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

fn unescape_xml(input: &str) -> String {
    input
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
        .replace("&amp;", "&")
}

fn escape_cdata(input: &str) -> String {
    input.replace("]]>", "]]]]><![CDATA[>")
}
