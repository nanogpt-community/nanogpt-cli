#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WebMode {
    Off,
    Enabled { suffix: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WebModePreset {
    pub key: &'static str,
    pub provider: &'static str,
    pub mode: &'static str,
    pub description: &'static str,
}

const WEB_MODE_PRESETS: &[WebModePreset] = &[
    WebModePreset {
        key: "online",
        provider: "auto",
        mode: "standard",
        description: "Default web routing",
    },
    WebModePreset {
        key: "linkup",
        provider: "linkup",
        mode: "standard",
        description: "Linkup standard search",
    },
    WebModePreset {
        key: "linkup-deep",
        provider: "linkup",
        mode: "deep",
        description: "Linkup deep search",
    },
    WebModePreset {
        key: "tavily",
        provider: "tavily",
        mode: "standard",
        description: "Tavily standard search",
    },
    WebModePreset {
        key: "tavily-deep",
        provider: "tavily",
        mode: "deep",
        description: "Tavily deep search",
    },
    WebModePreset {
        key: "brave",
        provider: "brave",
        mode: "standard",
        description: "Brave standard search",
    },
    WebModePreset {
        key: "brave-deep",
        provider: "brave",
        mode: "deep",
        description: "Brave deep search",
    },
    WebModePreset {
        key: "exa-fast",
        provider: "exa",
        mode: "fast",
        description: "Exa fast search",
    },
    WebModePreset {
        key: "exa-auto",
        provider: "exa",
        mode: "auto",
        description: "Exa auto search",
    },
    WebModePreset {
        key: "exa-neural",
        provider: "exa",
        mode: "neural",
        description: "Exa neural search",
    },
    WebModePreset {
        key: "exa-deep",
        provider: "exa",
        mode: "deep",
        description: "Exa deep search",
    },
    WebModePreset {
        key: "kagi",
        provider: "kagi",
        mode: "standard/search",
        description: "Kagi standard search source",
    },
    WebModePreset {
        key: "kagi-web",
        provider: "kagi",
        mode: "standard/web",
        description: "Kagi standard web source",
    },
    WebModePreset {
        key: "kagi-news",
        provider: "kagi",
        mode: "standard/news",
        description: "Kagi standard news source",
    },
    WebModePreset {
        key: "kagi-search",
        provider: "kagi",
        mode: "deep/search",
        description: "Kagi deep search source",
    },
    WebModePreset {
        key: "perplexity",
        provider: "perplexity",
        mode: "standard",
        description: "Perplexity standard search",
    },
    WebModePreset {
        key: "perplexity-deep",
        provider: "perplexity",
        mode: "deep",
        description: "Perplexity deep search",
    },
    WebModePreset {
        key: "valyu",
        provider: "valyu",
        mode: "standard/all",
        description: "Valyu standard all sources",
    },
    WebModePreset {
        key: "valyu-deep",
        provider: "valyu",
        mode: "deep/all",
        description: "Valyu deep all sources",
    },
    WebModePreset {
        key: "valyu-web",
        provider: "valyu",
        mode: "standard/web",
        description: "Valyu standard web only",
    },
    WebModePreset {
        key: "valyu-web-deep",
        provider: "valyu",
        mode: "deep/web",
        description: "Valyu deep web only",
    },
];

pub fn web_mode_presets() -> &'static [WebModePreset] {
    WEB_MODE_PRESETS
}

pub fn infer_from_flags(web: bool, deep_web: bool) -> WebMode {
    if deep_web {
        web_mode_from_key("linkup-deep")
    } else if web {
        web_mode_from_key("online")
    } else {
        WebMode::Off
    }
}

pub fn infer_from_model(model: &str) -> Option<WebMode> {
    let (_, suffix) = model.split_once(":online")?;
    if suffix.is_empty() {
        return Some(web_mode_from_key("online"));
    }
    Some(WebMode::Enabled {
        suffix: format!("online{suffix}"),
    })
}

pub fn parse_web_mode_arg(arg: &str) -> Result<WebMode, String> {
    let raw = arg.trim().to_lowercase();
    if raw.is_empty() {
        return Ok(web_mode_from_key("online"));
    }

    match raw.as_str() {
        "off" | "none" | "disable" | "disabled" => return Ok(WebMode::Off),
        "on" | "standard" | "online" | "auto" => return Ok(web_mode_from_key("online")),
        "deep" => return Ok(web_mode_from_key("linkup-deep")),
        _ => {}
    }

    let normalized = normalize_web_mode_key(&raw);
    if WEB_MODE_PRESETS
        .iter()
        .any(|preset| preset.key == normalized)
    {
        return Ok(web_mode_from_key(normalized.as_str()));
    }

    Err("Usage: /webmode <off|on|deep|provider-mode>".to_string())
}

pub fn web_mode_from_key(key: &str) -> WebMode {
    if key == "online" {
        WebMode::Enabled {
            suffix: "online".to_string(),
        }
    } else {
        WebMode::Enabled {
            suffix: format!("online/{key}"),
        }
    }
}

pub fn web_mode_key(mode: &WebMode) -> Option<&str> {
    match mode {
        WebMode::Off => None,
        WebMode::Enabled { suffix } if suffix == "online" => Some("online"),
        WebMode::Enabled { suffix } => suffix.strip_prefix("online/"),
    }
}

pub fn web_mode_display(mode: &WebMode) -> String {
    match mode {
        WebMode::Off => "off".to_string(),
        _ => web_mode_key(mode).unwrap_or("online").to_string(),
    }
}

pub fn apply_web_mode(model: &str, web_mode: &WebMode) -> String {
    let base = strip_online_suffix(model);

    match web_mode {
        WebMode::Off => base,
        WebMode::Enabled { suffix } => format!("{base}:{suffix}"),
    }
}

fn strip_online_suffix(model: &str) -> String {
    if let Some((left, _)) = model.split_once(":online") {
        left.to_string()
    } else {
        model.to_string()
    }
}

fn normalize_web_mode_key(raw: &str) -> String {
    raw.trim_start_matches(':')
        .trim_start_matches("online/")
        .replace('_', "-")
        .replace('/', "-")
        .replace(' ', "-")
}
