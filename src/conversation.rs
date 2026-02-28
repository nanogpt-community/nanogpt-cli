use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conversation {
    pub id: String,
    pub model: String,
    pub system_prompt: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub messages: Vec<ConversationMessage>,
}

#[derive(Debug, Clone)]
pub struct ConversationSummary {
    pub id: String,
    pub model: String,
    pub updated_at: DateTime<Utc>,
    pub message_count: usize,
}

impl Conversation {
    pub fn load_or_create(
        id_or_name: Option<&str>,
        model: String,
        system_prompt: Option<String>,
    ) -> Result<Self> {
        ensure_conversation_dir()?;

        if let Some(raw) = id_or_name {
            let id = normalize_id(raw);
            let path = conversation_path(&id);
            if path.exists() {
                return load(&id);
            }
            return Ok(new_conversation(id, model, system_prompt));
        }

        let id = format!("chat-{}", Uuid::new_v4());
        Ok(new_conversation(id, model, system_prompt))
    }

    pub fn save(&mut self) -> Result<()> {
        self.updated_at = Utc::now();
        ensure_conversation_dir()?;

        let path = conversation_path(&self.id);
        let data =
            serde_json::to_string_pretty(self).context("failed to serialize conversation")?;
        fs::write(path, data).context("failed to write conversation file")
    }

    pub fn push_user_message(&mut self, content: String) {
        self.messages.push(ConversationMessage {
            role: "user".to_string(),
            content,
        });
    }

    pub fn push_assistant_message(&mut self, content: String) {
        self.messages.push(ConversationMessage {
            role: "assistant".to_string(),
            content,
        });
    }

    pub fn clear_history(&mut self) {
        self.messages.clear();
    }
}

pub fn load(id_or_name: &str) -> Result<Conversation> {
    let id = normalize_id(id_or_name);
    let path = conversation_path(&id);
    if !path.exists() {
        return Err(anyhow!("conversation not found: {id}"));
    }

    let raw = fs::read_to_string(path).context("failed to read conversation file")?;
    serde_json::from_str(&raw).context("failed to parse conversation JSON")
}

pub fn delete(id_or_name: &str) -> Result<()> {
    let id = normalize_id(id_or_name);
    let path = conversation_path(&id);
    if !path.exists() {
        return Err(anyhow!("conversation not found: {id}"));
    }
    fs::remove_file(path).context("failed to delete conversation")
}

pub fn list() -> Result<Vec<ConversationSummary>> {
    ensure_conversation_dir()?;
    let mut summaries = Vec::new();

    for entry in fs::read_dir(conversation_dir()?).context("failed to list conversations")? {
        let entry = entry.context("failed to read conversation directory entry")?;
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("json") {
            continue;
        }

        let raw = match fs::read_to_string(&path) {
            Ok(v) => v,
            Err(_) => continue,
        };

        let conversation: Conversation = match serde_json::from_str(&raw) {
            Ok(v) => v,
            Err(_) => continue,
        };

        summaries.push(ConversationSummary {
            id: conversation.id,
            model: conversation.model,
            updated_at: conversation.updated_at,
            message_count: conversation.messages.len(),
        });
    }

    summaries.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
    Ok(summaries)
}

pub fn delete_empty_conversations() -> Result<usize> {
    let summaries = list()?;
    let mut deleted = 0usize;

    for summary in summaries {
        if summary.message_count == 0 {
            delete(&summary.id)?;
            deleted += 1;
        }
    }

    Ok(deleted)
}

pub fn conversation_dir() -> Result<PathBuf> {
    let home = dirs::home_dir().ok_or_else(|| anyhow!("failed to resolve home directory"))?;
    Ok(home.join(".nanogpt-cli").join("conversations"))
}

fn ensure_conversation_dir() -> Result<()> {
    fs::create_dir_all(conversation_dir()?).context("failed to create conversation directory")
}

fn conversation_path(id: &str) -> PathBuf {
    conversation_dir()
        .unwrap_or_else(|_| Path::new(".").to_path_buf())
        .join(format!("{}.json", normalize_id(id)))
}

fn new_conversation(id: String, model: String, system_prompt: Option<String>) -> Conversation {
    let now = Utc::now();
    Conversation {
        id,
        model,
        system_prompt,
        created_at: now,
        updated_at: now,
        messages: Vec::new(),
    }
}

fn normalize_id(value: &str) -> String {
    let cleaned = value
        .trim()
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '-'
            }
        })
        .collect::<String>();

    let normalized = cleaned.trim_matches('-').to_lowercase();
    if normalized.is_empty() {
        format!("chat-{}", Uuid::new_v4())
    } else {
        normalized
    }
}
