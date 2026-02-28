# nanogpt-cli

A Rust CLI + TUI client for [NanoGPT](https://nano-gpt.com/).

It is designed for day-to-day chat use in the terminal, with persistent conversations, fast model switching, provider controls, and direct access to NanoGPT endpoints.

## Highlights

- Full-screen TUI by default (`cargo run --`)
- Persistent conversations with quick switching
- Model browser with catalog scopes:
  - canonical (`/v1/models`)
  - subscription (`/subscription/v1/models`)
  - paid (`/paid/v1/models`)
- Provider routing per model (`/models/{canonicalId}/providers`)
- Web routing modes for chat (`/webmode` and picker)
- Built-in web search tool with all supported providers:
  - `linkup`, `tavily`, `exa`, `kagi`, `perplexity`, `valyu`, `brave`
- Markdown rendering in chat (headings, lists, code blocks, inline styles, links)
- API key onboarding in TUI if no key is configured
- Generic API passthrough (JSON + multipart) for full API coverage

## Quick Start

### 1. Build

```bash
cargo build
```

### 2. Run

```bash
# Default (opens TUI)
cargo run --

# Explicit TUI
cargo run -- tui

# REPL chat mode
cargo run -- chat --model openai/gpt-5.2 --stream
```

### 3. API key

You can authenticate with:

- environment variable: `NANOGPT_API_KEY`
- CLI flag: `--api-key`
- TUI prompt on first run (auto-saved to config)

## TUI Guide

### Keybindings

- `Tab`: switch focus (conversations/input)
- `Enter`: send message (input) or open selected conversation (list)
- `Ctrl+N`: create new conversation
- `Ctrl+R`: reload conversation list
- `Ctrl+S`: save current conversation
- `Ctrl+M`: open model gallery
- `Ctrl+P`: open provider routing for current model
- `Ctrl+G`: open web provider/mode picker for chat routing
- `Ctrl+W`: open web-search tool
- `Ctrl+H`: open help
- `Esc` / `Ctrl+C`: quit

### Slash commands

- `/help`
- `/model <id>`
- `/webmode` (open picker)
- `/webmode <mode>` (set directly, e.g. `off`, `deep`, `exa-neural`, `tavily-deep`)
- `/system <prompt>`
- `/system off`
- `/clear`
- `/save`
- `/history`
- `/models`
- `/providers`

### Slash autocomplete

- Type `/` to see command suggestions
- Use `Up/Down` to select
- Press `Tab` or `Enter` to autocomplete

## CLI Examples

### Show model catalogs

```bash
cargo run -- models --scope canonical --detailed
cargo run -- models --scope subscription --detailed
cargo run -- models --scope paid --detailed
```

### Show providers for one model

```bash
cargo run -- models --show-providers-for openai/gpt-oss-120b
```

### Web search

```bash
cargo run -- web-search "latest AI news" --provider linkup --depth standard
cargo run -- web-search "rust lang updates" --provider exa --depth fast
cargo run -- web-search "market headlines" --provider brave --depth deep
```

### Responses API

```bash
cargo run -- responses create --model openai/gpt-5.2 --input "Summarize this"
cargo run -- responses info
cargo run -- responses get <response_id>
cargo run -- responses delete <response_id>
```

### Balance

```bash
cargo run -- balance
```

### Generic API passthrough

```bash
# JSON request
cargo run -- api --method POST --path /v1/embeddings \
  --json '{"model":"text-embedding-3-small","input":"hello"}'

# Multipart request
cargo run -- api --method POST --path /transcribe \
  --form model=whisper-1 \
  --form-file file=/absolute/path/audio.mp3
```

## Storage

- Config: `~/.nanogpt-cli/config.json`
- Conversations: `~/.nanogpt-cli/conversations/*.json`

## Release Workflow

This repo includes a manual GitHub Actions release workflow that:

- builds Linux x64 and macOS arm64 binaries
- packages artifacts and checksums
- creates a GitHub Release from a manually provided tag

See: [`.github/workflows/release.yml`](.github/workflows/release.yml)

## Useful Links

- https://docs.nano-gpt.com/llms.txt
- https://docs.nano-gpt.com/api-reference/openapi.json
