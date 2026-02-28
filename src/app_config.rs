use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct AppConfig {
    pub api_key: Option<String>,
    pub default_model: Option<String>,
    pub provider_overrides: BTreeMap<String, String>,
}

pub fn load_config() -> Result<AppConfig> {
    let path = config_path()?;
    if !path.exists() {
        return Ok(AppConfig::default());
    }

    let raw = fs::read_to_string(&path)
        .with_context(|| format!("failed to read config file {}", path.display()))?;
    let cfg: AppConfig = serde_json::from_str(&raw)
        .with_context(|| format!("failed to parse config file {}", path.display()))?;
    Ok(cfg)
}

pub fn save_config(cfg: &AppConfig) -> Result<()> {
    ensure_config_dir()?;
    let path = config_path()?;
    let raw = serde_json::to_string_pretty(cfg).context("failed to serialize app config")?;
    fs::write(&path, raw).with_context(|| format!("failed to write config {}", path.display()))
}

pub fn config_path() -> Result<PathBuf> {
    let home = dirs::home_dir().ok_or_else(|| anyhow!("failed to resolve home directory"))?;
    Ok(home.join(".nanogpt-cli").join("config.json"))
}

fn ensure_config_dir() -> Result<()> {
    let path = config_path()?;
    let dir = path
        .parent()
        .ok_or_else(|| anyhow!("invalid config path {}", path.display()))?;
    fs::create_dir_all(Path::new(dir)).with_context(|| {
        format!(
            "failed to create config directory {}",
            Path::new(dir).display()
        )
    })
}
