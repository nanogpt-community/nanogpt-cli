use std::collections::BTreeMap;
use std::io;
use std::sync::mpsc::{self, Receiver};
use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::prelude::*;
use ratatui::widgets::{
    Block, BorderType, Borders, Clear, List, ListItem, ListState, Paragraph, Wrap,
};
use reqwest::Method;
use serde_json::{Map, Value, json};
use url::form_urlencoded::byte_serialize;

use crate::app_config::{AppConfig, load_config, save_config};
use crate::cli::{ModelScope, TuiArgs};
use crate::client::{ChatRequest, NanoGptClient};
use crate::conversation::{
    self as conv_store, Conversation, ConversationMessage, ConversationSummary,
};
use crate::model_mode::{
    WebMode, apply_web_mode, infer_from_flags, infer_from_model, parse_web_mode_arg,
    web_mode_display, web_mode_from_key, web_mode_key, web_mode_presets,
};

pub fn run_tui(client: NanoGptClient, args: TuiArgs) -> Result<()> {
    let initial_web_mode = infer_from_flags(args.web, args.deep_web);
    let initial_model = apply_web_mode(&args.model, &initial_web_mode);
    let mut conversation = if let Some(id) = args.conversation.as_deref() {
        Conversation::load_or_create(Some(id), initial_model.clone(), args.system.clone())?
    } else {
        conv_store::delete_empty_conversations()?;
        Conversation::load_or_create(None, initial_model.clone(), args.system.clone())?
    };

    if args.conversation.is_none() {
        if conversation.messages.is_empty() {
            conversation.model = initial_model.clone();
            if args.system.is_some() {
                conversation.system_prompt = args.system.clone();
            }
        }
        conversation.save()?;
    }

    let mut app = App::new(client, args, conversation)?;
    let mut terminal = init_terminal()?;

    let run_result = (|| -> Result<()> {
        loop {
            terminal.draw(|frame| app.render(frame))?;
            app.poll_response()?;

            if event::poll(Duration::from_millis(50)).context("failed to poll event")? {
                if let Event::Key(key) = event::read().context("failed to read event")? {
                    if app.handle_key(key)? {
                        break;
                    }
                }
            }
        }

        app.conversation.save()?;
        Ok(())
    })();

    let restore_result = restore_terminal(&mut terminal);
    match (run_result, restore_result) {
        (Err(run_err), Err(restore_err)) => Err(run_err).context(format!(
            "tui run failed and terminal restore also failed: {restore_err}"
        )),
        (Err(run_err), Ok(())) => Err(run_err),
        (Ok(()), Err(restore_err)) => Err(restore_err),
        (Ok(()), Ok(())) => Ok(()),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Focus {
    Input,
    Conversations,
}

#[derive(Debug, Clone)]
struct ModelEntry {
    id: String,
    name: String,
    owned_by: String,
    category: Option<String>,
    subscription_included: Option<bool>,
}

#[derive(Debug, Clone)]
struct ProviderEntry {
    provider: String,
    available: bool,
    input_per_1k: Option<f64>,
    output_per_1k: Option<f64>,
}

#[derive(Debug, Clone)]
struct ApiKeyModal {
    value: String,
    cursor: usize,
    reveal: bool,
}

#[derive(Debug, Clone)]
struct ModelPickerModal {
    scope: ModelScope,
    search: String,
    cursor: usize,
    selected: usize,
    models: Vec<ModelEntry>,
}

#[derive(Debug, Clone)]
struct ProviderPickerModal {
    model_id: String,
    selected: usize,
    supports_provider_selection: bool,
    providers: Vec<ProviderEntry>,
    message: Option<String>,
}

#[derive(Debug, Clone)]
struct WebModePickerModal {
    selected: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WebProvider {
    Linkup,
    Tavily,
    Exa,
    Kagi,
    Perplexity,
    Valyu,
    Brave,
}

impl WebProvider {
    fn all() -> &'static [WebProvider] {
        &[
            WebProvider::Linkup,
            WebProvider::Tavily,
            WebProvider::Exa,
            WebProvider::Kagi,
            WebProvider::Perplexity,
            WebProvider::Valyu,
            WebProvider::Brave,
        ]
    }

    fn as_str(self) -> &'static str {
        match self {
            WebProvider::Linkup => "linkup",
            WebProvider::Tavily => "tavily",
            WebProvider::Exa => "exa",
            WebProvider::Kagi => "kagi",
            WebProvider::Perplexity => "perplexity",
            WebProvider::Valyu => "valyu",
            WebProvider::Brave => "brave",
        }
    }

    fn depth_options(self) -> &'static [&'static str] {
        match self {
            WebProvider::Exa => &["fast", "auto", "neural", "deep"],
            WebProvider::Linkup
            | WebProvider::Tavily
            | WebProvider::Kagi
            | WebProvider::Perplexity
            | WebProvider::Valyu
            | WebProvider::Brave => &["standard", "deep"],
        }
    }

    fn output_options(self) -> &'static [&'static str] {
        match self {
            WebProvider::Linkup => &["searchResults", "sourcedAnswer", "structured"],
            WebProvider::Tavily
            | WebProvider::Exa
            | WebProvider::Kagi
            | WebProvider::Perplexity
            | WebProvider::Valyu
            | WebProvider::Brave => &["searchResults"],
        }
    }
}

#[derive(Debug, Clone)]
struct WebSearchModal {
    query: String,
    cursor: usize,
    provider_idx: usize,
    depth_idx: usize,
    output_idx: usize,
    include_images: bool,
    result: String,
    result_scroll: u16,
}

impl WebSearchModal {
    fn new() -> Self {
        Self {
            query: String::new(),
            cursor: 0,
            provider_idx: 0,
            depth_idx: 0,
            output_idx: 0,
            include_images: false,
            result: String::new(),
            result_scroll: 0,
        }
    }

    fn provider(&self) -> WebProvider {
        WebProvider::all()[self.provider_idx]
    }

    fn depth(&self) -> &'static str {
        let options = self.provider().depth_options();
        options[self.depth_idx.min(options.len().saturating_sub(1))]
    }

    fn output_type(&self) -> &'static str {
        let options = self.provider().output_options();
        options[self.output_idx.min(options.len().saturating_sub(1))]
    }

    fn cycle_provider(&mut self) {
        self.provider_idx = (self.provider_idx + 1) % WebProvider::all().len();
        let depth_len = self.provider().depth_options().len();
        let out_len = self.provider().output_options().len();
        if self.depth_idx >= depth_len {
            self.depth_idx = 0;
        }
        if self.output_idx >= out_len {
            self.output_idx = 0;
        }
    }

    fn cycle_depth(&mut self) {
        let depth_len = self.provider().depth_options().len();
        self.depth_idx = (self.depth_idx + 1) % depth_len;
    }

    fn cycle_output(&mut self) {
        let out_len = self.provider().output_options().len();
        self.output_idx = (self.output_idx + 1) % out_len;
    }
}

#[derive(Debug, Clone)]
enum Modal {
    ApiKey(ApiKeyModal),
    ModelPicker(ModelPickerModal),
    ProviderPicker(ProviderPickerModal),
    WebModePicker(WebModePickerModal),
    WebSearch(WebSearchModal),
    Help,
}

#[derive(Clone, Copy)]
struct SlashCommandSpec {
    command: &'static str,
    takes_argument: bool,
    description: &'static str,
}

const SLASH_COMMANDS: &[SlashCommandSpec] = &[
    SlashCommandSpec {
        command: "/help",
        takes_argument: false,
        description: "Open command guide",
    },
    SlashCommandSpec {
        command: "/model",
        takes_argument: true,
        description: "Set current model",
    },
    SlashCommandSpec {
        command: "/webmode",
        takes_argument: true,
        description: "Set web mode or open picker",
    },
    SlashCommandSpec {
        command: "/system",
        takes_argument: true,
        description: "Set/clear system prompt",
    },
    SlashCommandSpec {
        command: "/clear",
        takes_argument: false,
        description: "Clear conversation history",
    },
    SlashCommandSpec {
        command: "/save",
        takes_argument: false,
        description: "Save conversation",
    },
    SlashCommandSpec {
        command: "/history",
        takes_argument: false,
        description: "Show message count",
    },
    SlashCommandSpec {
        command: "/models",
        takes_argument: false,
        description: "Open model picker",
    },
    SlashCommandSpec {
        command: "/providers",
        takes_argument: false,
        description: "Open provider picker",
    },
];

#[derive(Clone, Copy)]
struct UiTheme {
    canvas: Color,
    panel: Color,
    panel_alt: Color,
    border: Color,
    border_focus: Color,
    text: Color,
    muted: Color,
    accent: Color,
    accent_2: Color,
    success: Color,
    warning: Color,
    danger: Color,
    user: Color,
    assistant: Color,
    system: Color,
    selected_bg: Color,
    selected_fg: Color,
}

fn ui_theme() -> UiTheme {
    UiTheme {
        canvas: Color::Rgb(12, 14, 20),
        panel: Color::Rgb(18, 21, 30),
        panel_alt: Color::Rgb(23, 27, 38),
        border: Color::Rgb(55, 66, 89),
        border_focus: Color::Rgb(94, 205, 255),
        text: Color::Rgb(230, 236, 244),
        muted: Color::Rgb(150, 162, 184),
        accent: Color::Rgb(94, 205, 255),
        accent_2: Color::Rgb(126, 255, 178),
        success: Color::Rgb(115, 223, 139),
        warning: Color::Rgb(255, 195, 83),
        danger: Color::Rgb(255, 117, 117),
        user: Color::Rgb(126, 255, 178),
        assistant: Color::Rgb(120, 201, 255),
        system: Color::Rgb(255, 206, 117),
        selected_bg: Color::Rgb(55, 84, 132),
        selected_fg: Color::Rgb(246, 250, 255),
    }
}

struct App {
    client: NanoGptClient,
    args: TuiArgs,
    conversation: Conversation,
    conversations: Vec<ConversationSummary>,
    selected_conversation_idx: usize,
    focus: Focus,
    modal: Option<Modal>,
    current_model: String,
    web_mode: WebMode,
    provider_overrides: BTreeMap<String, String>,
    input: String,
    cursor: usize,
    slash_suggestion_idx: usize,
    chat_scroll: u16,
    status: String,
    pending: bool,
    response_rx: Option<Receiver<anyhow::Result<String>>>,
    pending_user: Option<String>,
}

impl App {
    fn new(client: NanoGptClient, args: TuiArgs, conversation: Conversation) -> Result<Self> {
        let current_model = if conversation.model.is_empty() {
            apply_web_mode(&args.model, &infer_from_flags(args.web, args.deep_web))
        } else {
            conversation.model.clone()
        };
        let web_mode =
            infer_from_model(&current_model).unwrap_or(infer_from_flags(args.web, args.deep_web));
        let mut cfg = load_config().unwrap_or_else(|_| AppConfig::default());
        if let Some(cli_provider) = args.provider.clone() {
            let base = current_model_base_from_value(&current_model).to_string();
            cfg.provider_overrides.insert(base, cli_provider);
        }

        let mut app = Self {
            client,
            args: args.clone(),
            conversation,
            conversations: vec![],
            selected_conversation_idx: 0,
            focus: Focus::Input,
            modal: None,
            current_model,
            web_mode,
            provider_overrides: cfg.provider_overrides,
            input: String::new(),
            cursor: 0,
            slash_suggestion_idx: 0,
            chat_scroll: 0,
            status: "Ready".to_string(),
            pending: false,
            response_rx: None,
            pending_user: None,
        };

        app.reload_conversations()?;

        if !app.client.has_api_key() {
            app.modal = Some(Modal::ApiKey(ApiKeyModal {
                value: String::new(),
                cursor: 0,
                reveal: false,
            }));
            app.status = "NanoGPT API key required".to_string();
        }

        Ok(app)
    }

    fn render(&self, frame: &mut Frame) {
        let theme = ui_theme();
        frame.render_widget(
            Block::default().style(Style::default().bg(theme.canvas)),
            frame.area(),
        );

        let root = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(4),
                Constraint::Min(3),
                Constraint::Length(4),
                Constraint::Length(3),
            ])
            .split(frame.area());

        self.render_header(frame, root[0]);
        self.render_body(frame, root[1]);
        self.render_input(frame, root[2]);
        self.render_footer(frame, root[3]);
        self.render_slash_suggestions(frame, root[2]);

        if let Some(modal) = &self.modal {
            match modal {
                Modal::ApiKey(state) => self.render_api_key_modal(frame, state),
                Modal::ModelPicker(state) => self.render_model_picker_modal(frame, state),
                Modal::ProviderPicker(state) => self.render_provider_picker_modal(frame, state),
                Modal::WebModePicker(state) => self.render_web_mode_picker_modal(frame, state),
                Modal::WebSearch(state) => self.render_web_search_modal(frame, state),
                Modal::Help => self.render_help_modal(frame),
            }
        }
    }

    fn render_header(&self, frame: &mut Frame, area: Rect) {
        let theme = ui_theme();
        let provider = self.active_provider_for_current_model().unwrap_or("auto");
        let web_mode = web_mode_display(&self.web_mode);
        let active_focus = match self.focus {
            Focus::Input => "composer",
            Focus::Conversations => "conversations",
        };
        let state = if self.pending { "thinking" } else { "ready" };

        let title = Line::from(vec![
            Span::styled(
                " NANO GPT ",
                Style::default().fg(theme.canvas).bg(theme.accent),
            ),
            Span::styled(
                " TERMINAL STUDIO ",
                Style::default()
                    .fg(theme.canvas)
                    .bg(theme.accent_2)
                    .add_modifier(Modifier::BOLD),
            ),
        ]);
        let header = Paragraph::new(vec![
            Line::from(vec![
                Span::styled("Conversation ", Style::default().fg(theme.muted)),
                Span::styled(
                    truncate_middle(&self.conversation.id, 42),
                    Style::default().fg(theme.text).add_modifier(Modifier::BOLD),
                ),
                Span::styled("  |  Messages ", Style::default().fg(theme.muted)),
                Span::styled(
                    self.conversation.messages.len().to_string(),
                    Style::default()
                        .fg(theme.accent_2)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled("  |  Focus ", Style::default().fg(theme.muted)),
                Span::styled(active_focus, Style::default().fg(theme.accent)),
            ]),
            Line::from(vec![
                Span::styled("Model ", Style::default().fg(theme.muted)),
                Span::styled(
                    truncate_middle(&self.current_model, 40),
                    Style::default().fg(theme.text),
                ),
                Span::styled("  |  Provider ", Style::default().fg(theme.muted)),
                Span::styled(provider, Style::default().fg(theme.warning)),
                Span::styled("  |  Web ", Style::default().fg(theme.muted)),
                Span::styled(web_mode, Style::default().fg(theme.accent_2)),
                Span::styled("  |  Status ", Style::default().fg(theme.muted)),
                Span::styled(
                    state,
                    Style::default()
                        .fg(if self.pending {
                            theme.warning
                        } else {
                            theme.success
                        })
                        .add_modifier(Modifier::BOLD),
                ),
            ]),
        ])
        .block(
            Block::default()
                .title(title)
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .style(Style::default().bg(theme.panel_alt))
                .border_style(Style::default().fg(theme.border_focus)),
        );
        frame.render_widget(header, area);
    }

    fn render_body(&self, frame: &mut Frame, area: Rect) {
        let theme = ui_theme();
        let body = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Length(40), Constraint::Min(30)])
            .split(area);

        let mut items = Vec::new();
        for c in &self.conversations {
            let is_active = c.id == self.conversation.id;
            let marker = if is_active { "●" } else { "○" };
            let when = c.updated_at.format("%m-%d %H:%M");
            let line = vec![
                Line::from(vec![
                    Span::styled(
                        format!("{marker} "),
                        Style::default().fg(if is_active {
                            theme.accent_2
                        } else {
                            theme.muted
                        }),
                    ),
                    Span::styled(
                        truncate_middle(&c.id, 34),
                        Style::default().fg(theme.text).add_modifier(Modifier::BOLD),
                    ),
                ]),
                Line::from(vec![
                    Span::styled(
                        format!("{} msgs", c.message_count),
                        Style::default().fg(theme.accent),
                    ),
                    Span::styled("  •  ", Style::default().fg(theme.muted)),
                    Span::styled(
                        truncate_middle(&c.model, 18),
                        Style::default().fg(theme.muted),
                    ),
                    Span::styled("  •  ", Style::default().fg(theme.muted)),
                    Span::styled(when.to_string(), Style::default().fg(theme.muted)),
                ]),
            ];
            items.push(ListItem::new(line));
        }

        if items.is_empty() {
            items.push(ListItem::new(Line::styled(
                "No conversations yet",
                Style::default().fg(theme.muted),
            )));
        }

        let conv_block = Block::default()
            .title(Line::from(vec![
                Span::styled(" Conversations ", Style::default().fg(theme.accent)),
                Span::styled(
                    if self.focus == Focus::Conversations {
                        " ACTIVE "
                    } else {
                        " BROWSE "
                    },
                    Style::default()
                        .fg(theme.canvas)
                        .bg(if self.focus == Focus::Conversations {
                            theme.accent_2
                        } else {
                            theme.muted
                        }),
                ),
            ]))
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .style(Style::default().bg(theme.panel))
            .border_style(if self.focus == Focus::Conversations {
                Style::default().fg(theme.border_focus)
            } else {
                Style::default().fg(theme.border)
            });

        let conv_list = List::new(items)
            .block(conv_block)
            .highlight_style(
                Style::default()
                    .bg(theme.selected_bg)
                    .fg(theme.selected_fg)
                    .add_modifier(Modifier::BOLD),
            )
            .highlight_symbol("▌ ");

        let mut list_state = ListState::default();
        list_state.select(Some(self.selected_conversation_idx));
        frame.render_stateful_widget(conv_list, body[0], &mut list_state);

        let mut lines: Vec<Line> = Vec::new();
        if let Some(system) = &self.conversation.system_prompt {
            lines.push(Line::from(vec![
                Span::styled(
                    " SYSTEM ",
                    Style::default()
                        .fg(theme.canvas)
                        .bg(theme.system)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled("  policy", Style::default().fg(theme.muted)),
            ]));
            let rendered = render_markdown_block(system, theme, Style::default().fg(theme.text));
            for line in rendered {
                lines.push(indent_line(line, "  ", Style::default().fg(theme.muted)));
            }
            lines.push(Line::raw(""));
        }

        for (idx, msg) in self.conversation.messages.iter().enumerate() {
            let (role, tone) = match msg.role.as_str() {
                "assistant" => ("ASSISTANT", theme.assistant),
                "user" => ("YOU", theme.user),
                _ => ("SYSTEM", theme.system),
            };

            lines.push(Line::from(vec![
                Span::styled(
                    format!(" {role} "),
                    Style::default()
                        .fg(theme.canvas)
                        .bg(tone)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    format!("  message #{:02}", idx + 1),
                    Style::default().fg(theme.muted),
                ),
            ]));

            let rendered =
                render_markdown_block(&msg.content, theme, Style::default().fg(theme.text));
            for line in rendered {
                lines.push(indent_line(line, "  ", Style::default().fg(theme.muted)));
            }
            if msg.content.is_empty() {
                lines.push(Line::from(Span::styled(
                    "  (empty)",
                    Style::default().fg(theme.muted),
                )));
            }
            lines.push(Line::raw(""));
        }

        if self.pending {
            if let Some(user) = &self.pending_user {
                lines.push(Line::from(vec![
                    Span::styled(
                        " YOU ",
                        Style::default()
                            .fg(theme.canvas)
                            .bg(theme.user)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(format!("  {}", user), Style::default().fg(theme.text)),
                ]));
                lines.push(Line::raw(""));
            }
            lines.push(Line::from(vec![
                Span::styled(
                    " ASSISTANT ",
                    Style::default()
                        .fg(theme.canvas)
                        .bg(theme.assistant)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    "  composing response...",
                    Style::default().fg(theme.warning),
                ),
            ]));
        }

        let chat = Paragraph::new(lines)
            .wrap(Wrap { trim: false })
            .scroll((self.chat_scroll, 0))
            .block(
                Block::default()
                    .title(Line::from(vec![
                        Span::styled(" Chat Stream ", Style::default().fg(theme.accent_2)),
                        Span::styled(" LIVE ", Style::default().fg(theme.canvas).bg(theme.accent)),
                    ]))
                    .borders(Borders::ALL)
                    .border_type(BorderType::Rounded)
                    .style(Style::default().bg(theme.panel))
                    .border_style(Style::default().fg(theme.border)),
            );
        frame.render_widget(chat, body[1]);
    }

    fn render_input(&self, frame: &mut Frame, area: Rect) {
        let theme = ui_theme();
        let count = self.input.chars().count();
        let placeholder = "Type a message or /command. Enter to send.";
        let input_line = if self.input.is_empty() {
            Line::from(Span::styled(placeholder, Style::default().fg(theme.muted)))
        } else {
            Line::from(Span::styled(
                self.input.as_str(),
                Style::default().fg(theme.text),
            ))
        };
        let hint_line = Line::from(vec![
            Span::styled("Ctrl+M", Style::default().fg(theme.accent)),
            Span::styled(" models  ", Style::default().fg(theme.muted)),
            Span::styled("Ctrl+P", Style::default().fg(theme.accent)),
            Span::styled(" providers  ", Style::default().fg(theme.muted)),
            Span::styled("Ctrl+G", Style::default().fg(theme.accent)),
            Span::styled(" web mode  ", Style::default().fg(theme.muted)),
            Span::styled("Ctrl+W", Style::default().fg(theme.accent)),
            Span::styled(" web search  ", Style::default().fg(theme.muted)),
            Span::styled(format!("{count} chars"), Style::default().fg(theme.warning)),
        ]);

        let input = Paragraph::new(vec![input_line, Line::raw(""), hint_line])
            .block(
                Block::default()
                    .title(Line::from(vec![
                        Span::styled(" Composer ", Style::default().fg(theme.accent)),
                        Span::styled(
                            if self.focus == Focus::Input {
                                " ACTIVE "
                            } else {
                                " READY "
                            },
                            Style::default()
                                .fg(theme.canvas)
                                .bg(if self.focus == Focus::Input {
                                    theme.accent_2
                                } else {
                                    theme.muted
                                }),
                        ),
                    ]))
                    .borders(Borders::ALL)
                    .border_type(BorderType::Rounded)
                    .style(Style::default().bg(theme.panel_alt))
                    .border_style(if self.focus == Focus::Input {
                        Style::default().fg(theme.border_focus)
                    } else {
                        Style::default().fg(theme.border)
                    }),
            )
            .wrap(Wrap { trim: false });
        frame.render_widget(input, area);

        if self.focus == Focus::Input && self.modal.is_none() {
            let cursor_x = area
                .x
                .saturating_add(self.cursor as u16)
                .saturating_add(1)
                .min(area.x + area.width.saturating_sub(2));
            let cursor_y = area.y + 1;
            frame.set_cursor_position((cursor_x, cursor_y));
        }
    }

    fn render_slash_suggestions(&self, frame: &mut Frame, input_area: Rect) {
        let theme = ui_theme();
        let suggestions = self.filtered_slash_suggestions();
        if suggestions.is_empty() || self.modal.is_some() || self.focus != Focus::Input {
            return;
        }

        let visible = suggestions.len().min(5);
        let height = (visible as u16) + 2;
        let y = input_area.y.saturating_sub(height.saturating_sub(1));
        let width = input_area.width.min(72);
        let x = input_area.x;
        let area = Rect::new(x, y, width, height);

        let mut items = Vec::new();
        for cmd in suggestions.iter().take(visible) {
            let marker = if cmd.takes_argument { "…" } else { "" };
            items.push(ListItem::new(Line::from(vec![
                Span::styled(
                    format!("{}{}", cmd.command, marker),
                    Style::default()
                        .fg(theme.accent)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled("  ", Style::default().fg(theme.muted)),
                Span::styled(cmd.description, Style::default().fg(theme.text)),
            ])));
        }

        let list = List::new(items)
            .block(
                Block::default()
                    .title(Line::from(vec![
                        Span::styled(" / commands ", Style::default().fg(theme.accent)),
                        Span::styled(
                            " TAB/ENTER ",
                            Style::default().fg(theme.canvas).bg(theme.accent_2),
                        ),
                    ]))
                    .borders(Borders::ALL)
                    .border_type(BorderType::Rounded)
                    .style(Style::default().bg(theme.panel_alt))
                    .border_style(Style::default().fg(theme.border_focus)),
            )
            .highlight_style(
                Style::default()
                    .bg(theme.selected_bg)
                    .fg(theme.selected_fg)
                    .add_modifier(Modifier::BOLD),
            )
            .highlight_symbol("▸ ");

        let mut state = ListState::default();
        state.select(Some(
            self.slash_suggestion_idx.min(visible.saturating_sub(1)),
        ));

        frame.render_widget(Clear, area);
        frame.render_stateful_widget(list, area, &mut state);
    }

    fn render_footer(&self, frame: &mut Frame, area: Rect) {
        let theme = ui_theme();
        let status_lower = self.status.to_lowercase();
        let status_color = if status_lower.contains("fail")
            || status_lower.contains("error")
            || status_lower.contains("invalid")
        {
            theme.danger
        } else if status_lower.contains("saved")
            || status_lower.contains("completed")
            || status_lower.contains("received")
        {
            theme.success
        } else {
            theme.warning
        };

        let shortcuts = Line::from(vec![
            Span::styled("Tab", Style::default().fg(theme.accent)),
            Span::styled(" focus  ", Style::default().fg(theme.muted)),
            Span::styled("Enter", Style::default().fg(theme.accent)),
            Span::styled(" send/open  ", Style::default().fg(theme.muted)),
            Span::styled("Ctrl+N", Style::default().fg(theme.accent)),
            Span::styled(" new  ", Style::default().fg(theme.muted)),
            Span::styled("Ctrl+H", Style::default().fg(theme.accent)),
            Span::styled(" help  ", Style::default().fg(theme.muted)),
            Span::styled("Esc", Style::default().fg(theme.accent)),
            Span::styled(" quit", Style::default().fg(theme.muted)),
        ]);

        let status_line = Line::from(vec![
            Span::styled("STATUS  ", Style::default().fg(theme.muted)),
            Span::styled(
                self.status.as_str(),
                Style::default()
                    .fg(status_color)
                    .add_modifier(Modifier::BOLD),
            ),
        ]);

        let footer = Paragraph::new(vec![shortcuts, status_line]).block(
            Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .style(Style::default().bg(theme.panel_alt))
                .border_style(Style::default().fg(theme.border)),
        );
        frame.render_widget(footer, area);
    }

    fn render_api_key_modal(&self, frame: &mut Frame, state: &ApiKeyModal) {
        let theme = ui_theme();
        let area = centered_rect(frame.area(), 70, 30);
        frame.render_widget(Clear, area);

        let shown = if state.reveal {
            state.value.clone()
        } else {
            "*".repeat(state.value.len())
        };

        let text = vec![
            Line::from(Span::styled(
                "NanoGPT API key is required to continue.",
                Style::default().fg(theme.text).add_modifier(Modifier::BOLD),
            )),
            Line::raw(""),
            Line::from(vec![
                Span::styled("Press ", Style::default().fg(theme.muted)),
                Span::styled("Enter", Style::default().fg(theme.accent)),
                Span::styled(" to save. ", Style::default().fg(theme.muted)),
                Span::styled("Ctrl+R", Style::default().fg(theme.accent)),
                Span::styled(" toggles reveal.", Style::default().fg(theme.muted)),
            ]),
            Line::raw(""),
            Line::from(vec![
                Span::styled("API Key  ", Style::default().fg(theme.warning)),
                Span::styled(shown, Style::default().fg(theme.text)),
            ]),
        ];
        let widget = Paragraph::new(text).wrap(Wrap { trim: false }).block(
            Block::default()
                .title(Line::from(vec![
                    Span::styled(" Access Setup ", Style::default().fg(theme.accent)),
                    Span::styled(
                        " REQUIRED ",
                        Style::default()
                            .fg(theme.canvas)
                            .bg(theme.warning)
                            .add_modifier(Modifier::BOLD),
                    ),
                ]))
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .style(Style::default().bg(theme.panel_alt))
                .border_style(Style::default().fg(theme.border_focus)),
        );
        frame.render_widget(widget, area);

        let prefix = "API Key  ".len() as u16;
        let cursor_x = area
            .x
            .saturating_add(prefix)
            .saturating_add(state.cursor as u16)
            .saturating_add(1)
            .min(area.x + area.width.saturating_sub(2));
        let cursor_y = area.y + 5;
        frame.set_cursor_position((cursor_x, cursor_y));
    }

    fn render_model_picker_modal(&self, frame: &mut Frame, state: &ModelPickerModal) {
        let theme = ui_theme();
        let area = centered_rect(frame.area(), 85, 80);
        frame.render_widget(Clear, area);

        let inner = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(5), Constraint::Min(6)])
            .split(area);

        let scope = match state.scope {
            ModelScope::Canonical => "canonical",
            ModelScope::Subscription => "subscription",
            ModelScope::Paid => "paid",
        };
        let filtered = state.filtered_models();

        let header = Paragraph::new(vec![
            Line::from(vec![
                Span::styled("Scope  ", Style::default().fg(theme.muted)),
                Span::styled(
                    scope,
                    Style::default()
                        .fg(theme.accent_2)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled("  •  Models  ", Style::default().fg(theme.muted)),
                Span::styled(
                    filtered.len().to_string(),
                    Style::default().fg(theme.accent),
                ),
                Span::styled("  •  Tab cycles scope", Style::default().fg(theme.muted)),
            ]),
            Line::from(vec![
                Span::styled("Search  ", Style::default().fg(theme.warning)),
                Span::styled(state.search.as_str(), Style::default().fg(theme.text)),
            ]),
            Line::from(Span::styled(
                "Enter select | P provider options | Esc close",
                Style::default().fg(theme.muted),
            )),
        ])
        .block(
            Block::default()
                .title(Line::from(vec![
                    Span::styled(" Model Gallery ", Style::default().fg(theme.accent)),
                    Span::styled(
                        " BROWSE ",
                        Style::default().fg(theme.canvas).bg(theme.accent_2),
                    ),
                ]))
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .style(Style::default().bg(theme.panel_alt))
                .border_style(Style::default().fg(theme.border_focus)),
        );
        frame.render_widget(header, inner[0]);

        let mut items = Vec::new();
        for m in &filtered {
            let sub = m
                .subscription_included
                .map(|v| if v { "subscription" } else { "paygo" })
                .unwrap_or("sub=?");
            let category = m.category.clone().unwrap_or_default();
            items.push(ListItem::new(vec![
                Line::from(vec![
                    Span::styled(
                        truncate_middle(&m.id, 62),
                        Style::default().fg(theme.text).add_modifier(Modifier::BOLD),
                    ),
                    Span::styled("  ", Style::default().fg(theme.muted)),
                    Span::styled(format!("[{sub}]"), Style::default().fg(theme.warning)),
                ]),
                Line::from(vec![
                    Span::styled(m.owned_by.as_str(), Style::default().fg(theme.accent)),
                    Span::styled("  •  ", Style::default().fg(theme.muted)),
                    Span::styled(category, Style::default().fg(theme.muted)),
                ]),
            ]));
        }

        if items.is_empty() {
            items.push(ListItem::new(Line::styled(
                "No models matched your search",
                Style::default().fg(theme.muted),
            )));
        }

        let list = List::new(items)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_type(BorderType::Rounded)
                    .style(Style::default().bg(theme.panel))
                    .border_style(Style::default().fg(theme.border)),
            )
            .highlight_style(
                Style::default()
                    .bg(theme.selected_bg)
                    .fg(theme.selected_fg)
                    .add_modifier(Modifier::BOLD),
            )
            .highlight_symbol("▸ ");

        let mut list_state = ListState::default();
        list_state.select(Some(state.selected.min(filtered.len().saturating_sub(1))));
        frame.render_stateful_widget(list, inner[1], &mut list_state);

        let cursor_x = inner[0]
            .x
            .saturating_add("Search  ".len() as u16)
            .saturating_add(state.cursor as u16)
            .saturating_add(1)
            .min(inner[0].x + inner[0].width.saturating_sub(2));
        let cursor_y = inner[0].y + 2;
        frame.set_cursor_position((cursor_x, cursor_y));
    }

    fn render_provider_picker_modal(&self, frame: &mut Frame, state: &ProviderPickerModal) {
        let theme = ui_theme();
        let area = centered_rect(frame.area(), 80, 70);
        frame.render_widget(Clear, area);

        let lines = vec![
            Line::from(vec![
                Span::styled("Model  ", Style::default().fg(theme.muted)),
                Span::styled(
                    truncate_middle(&state.model_id, 72),
                    Style::default().fg(theme.text).add_modifier(Modifier::BOLD),
                ),
            ]),
            Line::from(Span::styled(
                "Enter selects provider | C clears override | Esc closes",
                Style::default().fg(theme.muted),
            )),
        ];

        if !state.supports_provider_selection {
            let mut disabled = lines;
            disabled.push(Line::raw(""));
            disabled.push(Line::styled(
                state
                    .message
                    .clone()
                    .unwrap_or_else(|| "Provider selection is not supported".to_string()),
                Style::default().fg(theme.warning),
            ));
            let widget = Paragraph::new(disabled).wrap(Wrap { trim: false }).block(
                Block::default()
                    .title(Line::from(vec![
                        Span::styled(" Provider Routing ", Style::default().fg(theme.accent)),
                        Span::styled(
                            " DISABLED ",
                            Style::default().fg(theme.canvas).bg(theme.warning),
                        ),
                    ]))
                    .borders(Borders::ALL)
                    .border_type(BorderType::Rounded)
                    .style(Style::default().bg(theme.panel_alt))
                    .border_style(Style::default().fg(theme.warning)),
            );
            frame.render_widget(widget, area);
            return;
        }

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(4), Constraint::Min(5)])
            .split(area);

        let header = Paragraph::new(lines).wrap(Wrap { trim: false }).block(
            Block::default()
                .title(Line::from(vec![
                    Span::styled(" Provider Routing ", Style::default().fg(theme.accent)),
                    Span::styled(
                        " ACTIVE ",
                        Style::default().fg(theme.canvas).bg(theme.accent_2),
                    ),
                ]))
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .style(Style::default().bg(theme.panel_alt))
                .border_style(Style::default().fg(theme.border_focus)),
        );
        frame.render_widget(header, chunks[0]);

        let mut items = Vec::new();
        for p in &state.providers {
            let avail = if p.available {
                "available"
            } else {
                "unavailable"
            };
            let i = p
                .input_per_1k
                .map(|v| format!("{v:.6}"))
                .unwrap_or_else(|| "-".to_string());
            let o = p
                .output_per_1k
                .map(|v| format!("{v:.6}"))
                .unwrap_or_else(|| "-".to_string());
            items.push(ListItem::new(vec![
                Line::from(vec![
                    Span::styled(
                        p.provider.as_str(),
                        Style::default().fg(theme.text).add_modifier(Modifier::BOLD),
                    ),
                    Span::styled("  ", Style::default().fg(theme.muted)),
                    Span::styled(
                        avail,
                        Style::default().fg(if p.available {
                            theme.success
                        } else {
                            theme.warning
                        }),
                    ),
                ]),
                Line::from(vec![
                    Span::styled("input/1k ", Style::default().fg(theme.muted)),
                    Span::styled(i, Style::default().fg(theme.accent)),
                    Span::styled("  output/1k ", Style::default().fg(theme.muted)),
                    Span::styled(o, Style::default().fg(theme.accent_2)),
                ]),
            ]));
        }

        if items.is_empty() {
            items.push(ListItem::new(Line::styled(
                "No providers returned",
                Style::default().fg(theme.muted),
            )));
        }

        let list = List::new(items)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_type(BorderType::Rounded)
                    .style(Style::default().bg(theme.panel))
                    .border_style(Style::default().fg(theme.border)),
            )
            .highlight_style(
                Style::default()
                    .bg(theme.selected_bg)
                    .fg(theme.selected_fg)
                    .add_modifier(Modifier::BOLD),
            )
            .highlight_symbol("▸ ");
        let mut list_state = ListState::default();
        list_state.select(Some(
            state.selected.min(state.providers.len().saturating_sub(1)),
        ));
        frame.render_stateful_widget(list, chunks[1], &mut list_state);
    }

    fn render_web_mode_picker_modal(&self, frame: &mut Frame, state: &WebModePickerModal) {
        let theme = ui_theme();
        let area = centered_rect(frame.area(), 86, 78);
        frame.render_widget(Clear, area);

        let presets = web_mode_presets();
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(4), Constraint::Min(8)])
            .split(area);

        let header = Paragraph::new(vec![
            Line::from(vec![
                Span::styled("Current  ", Style::default().fg(theme.muted)),
                Span::styled(
                    web_mode_display(&self.web_mode),
                    Style::default()
                        .fg(theme.accent_2)
                        .add_modifier(Modifier::BOLD),
                ),
            ]),
            Line::from(Span::styled(
                "Enter apply | C set off | Esc close",
                Style::default().fg(theme.muted),
            )),
        ])
        .block(
            Block::default()
                .title(Line::from(vec![
                    Span::styled(" Web Routing ", Style::default().fg(theme.accent)),
                    Span::styled(
                        " PROVIDER/MODE ",
                        Style::default().fg(theme.canvas).bg(theme.accent_2),
                    ),
                ]))
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .style(Style::default().bg(theme.panel_alt))
                .border_style(Style::default().fg(theme.border_focus)),
        );
        frame.render_widget(header, chunks[0]);

        let mut items = Vec::new();
        items.push(ListItem::new(vec![
            Line::from(vec![
                Span::styled(
                    "off",
                    Style::default()
                        .fg(theme.warning)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled("  ", Style::default().fg(theme.muted)),
                Span::styled("disable web search", Style::default().fg(theme.text)),
            ]),
            Line::from(Span::styled(
                "Model is sent without :online suffix",
                Style::default().fg(theme.muted),
            )),
        ]));

        for preset in presets {
            let suffix = if preset.key == "online" {
                ":online".to_string()
            } else {
                format!(":online/{}", preset.key)
            };
            items.push(ListItem::new(vec![
                Line::from(vec![
                    Span::styled(
                        format!("{:<12}", preset.provider),
                        Style::default()
                            .fg(theme.accent)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::styled("  ", Style::default().fg(theme.muted)),
                    Span::styled(preset.mode, Style::default().fg(theme.warning)),
                    Span::styled("  ", Style::default().fg(theme.muted)),
                    Span::styled(suffix, Style::default().fg(theme.text)),
                ]),
                Line::from(Span::styled(
                    preset.description,
                    Style::default().fg(theme.muted),
                )),
            ]));
        }

        let list = List::new(items)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_type(BorderType::Rounded)
                    .style(Style::default().bg(theme.panel))
                    .border_style(Style::default().fg(theme.border)),
            )
            .highlight_style(
                Style::default()
                    .bg(theme.selected_bg)
                    .fg(theme.selected_fg)
                    .add_modifier(Modifier::BOLD),
            )
            .highlight_symbol("▸ ");

        let max_idx = presets.len();
        let mut list_state = ListState::default();
        list_state.select(Some(state.selected.min(max_idx)));
        frame.render_stateful_widget(list, chunks[1], &mut list_state);
    }

    fn render_web_search_modal(&self, frame: &mut Frame, state: &WebSearchModal) {
        let theme = ui_theme();
        let area = centered_rect(frame.area(), 90, 85);
        frame.render_widget(Clear, area);

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(5),
                Constraint::Length(4),
                Constraint::Min(8),
            ])
            .split(area);

        let header = Paragraph::new(vec![
            Line::from(vec![
                Span::styled("Provider  ", Style::default().fg(theme.muted)),
                Span::styled(state.provider().as_str(), Style::default().fg(theme.accent)),
                Span::styled("  |  Depth  ", Style::default().fg(theme.muted)),
                Span::styled(state.depth(), Style::default().fg(theme.accent_2)),
            ]),
            Line::from(vec![
                Span::styled("Output  ", Style::default().fg(theme.muted)),
                Span::styled(state.output_type(), Style::default().fg(theme.warning)),
                Span::styled("  |  Include images  ", Style::default().fg(theme.muted)),
                Span::styled(
                    if state.include_images { "yes" } else { "no" },
                    Style::default().fg(theme.text),
                ),
            ]),
            Line::from(Span::styled(
                "Tab provider | D depth | O output | I images",
                Style::default().fg(theme.muted),
            )),
        ])
        .block(
            Block::default()
                .title(Line::from(vec![
                    Span::styled(" Web Search Lab ", Style::default().fg(theme.accent)),
                    Span::styled(
                        " LIVE ",
                        Style::default().fg(theme.canvas).bg(theme.accent_2),
                    ),
                ]))
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .style(Style::default().bg(theme.panel_alt))
                .border_style(Style::default().fg(theme.border_focus)),
        );
        frame.render_widget(header, chunks[0]);

        let query = Paragraph::new(vec![
            Line::from(Span::styled(
                if state.query.is_empty() {
                    "Type your search query..."
                } else {
                    state.query.as_str()
                },
                Style::default().fg(if state.query.is_empty() {
                    theme.muted
                } else {
                    theme.text
                }),
            )),
            Line::raw(""),
            Line::from(Span::styled(
                "Enter executes search | Esc closes",
                Style::default().fg(theme.muted),
            )),
        ])
        .block(
            Block::default()
                .title(Line::from(Span::styled(
                    " Query ",
                    Style::default().fg(theme.warning),
                )))
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .style(Style::default().bg(theme.panel))
                .border_style(Style::default().fg(theme.border)),
        );
        frame.render_widget(query, chunks[1]);

        let results = Paragraph::new(state.result.clone())
            .wrap(Wrap { trim: false })
            .scroll((state.result_scroll, 0))
            .block(
                Block::default()
                    .title(Line::from(Span::styled(
                        " Results ",
                        Style::default().fg(theme.accent_2),
                    )))
                    .borders(Borders::ALL)
                    .border_type(BorderType::Rounded)
                    .style(Style::default().bg(theme.panel))
                    .border_style(Style::default().fg(theme.border)),
            );
        frame.render_widget(results, chunks[2]);

        let cursor_x = chunks[1]
            .x
            .saturating_add(state.cursor as u16)
            .saturating_add(1)
            .min(chunks[1].x + chunks[1].width.saturating_sub(2));
        let cursor_y = chunks[1].y + 1;
        frame.set_cursor_position((cursor_x, cursor_y));
    }

    fn render_help_modal(&self, frame: &mut Frame) {
        let theme = ui_theme();
        let area = centered_rect(frame.area(), 75, 60);
        frame.render_widget(Clear, area);

        let text = "Global\n  Esc / Ctrl+C  quit\n  Tab           switch focus\n  Ctrl+N        new conversation\n  Ctrl+R        reload conversations\n  Ctrl+S        save conversation\n  Ctrl+M        open model gallery\n  Ctrl+P        provider routing for current model\n  Ctrl+G        web provider/mode selector\n  Ctrl+W        open web search lab\n\nConversations pane\n  Up/Down       select conversation\n  Enter         open selected\n  D / Delete    delete selected\n\nComposer pane\n  Enter         send message\n  /help /model /webmode /system /clear /save /history\n\nPress Esc to close.";

        let widget = Paragraph::new(text).wrap(Wrap { trim: false }).block(
            Block::default()
                .title(Line::from(vec![
                    Span::styled(" Command Guide ", Style::default().fg(theme.accent)),
                    Span::styled(
                        " HELP ",
                        Style::default().fg(theme.canvas).bg(theme.accent_2),
                    ),
                ]))
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .style(Style::default().bg(theme.panel_alt))
                .border_style(Style::default().fg(theme.border_focus)),
        );
        frame.render_widget(widget, area);
    }

    fn slash_token(&self) -> Option<String> {
        if !self.input.starts_with('/') {
            return None;
        }
        let token = self
            .input
            .split_whitespace()
            .next()
            .unwrap_or_default()
            .trim();
        if token.is_empty() {
            return None;
        }
        Some(token.to_lowercase())
    }

    fn filtered_slash_suggestions(&self) -> Vec<SlashCommandSpec> {
        if self.input.contains(' ') {
            return vec![];
        }

        let Some(token) = self.slash_token() else {
            return vec![];
        };

        if token == "/" {
            return SLASH_COMMANDS.to_vec();
        }

        SLASH_COMMANDS
            .iter()
            .copied()
            .filter(|spec| spec.command.starts_with(&token))
            .collect()
    }

    fn has_active_slash_suggestions(&self) -> bool {
        !self.filtered_slash_suggestions().is_empty()
    }

    fn slash_token_is_exact_command(&self) -> bool {
        let Some(token) = self.slash_token() else {
            return false;
        };
        SLASH_COMMANDS.iter().any(|spec| spec.command == token)
    }

    fn can_autocomplete_slash(&self) -> bool {
        self.input.starts_with('/') && !self.input.contains(' ')
    }

    fn sync_slash_suggestion_index(&mut self) {
        let len = self.filtered_slash_suggestions().len();
        if len == 0 {
            self.slash_suggestion_idx = 0;
            return;
        }
        if self.slash_suggestion_idx >= len {
            self.slash_suggestion_idx = len - 1;
        }
    }

    fn move_slash_suggestion(&mut self, delta: isize) {
        let len = self.filtered_slash_suggestions().len();
        if len == 0 {
            self.slash_suggestion_idx = 0;
            return;
        }
        let current = self.slash_suggestion_idx as isize;
        let next = (current + delta).clamp(0, (len.saturating_sub(1)) as isize);
        self.slash_suggestion_idx = next as usize;
    }

    fn apply_selected_slash_suggestion(&mut self) -> bool {
        let suggestions = self.filtered_slash_suggestions();
        if suggestions.is_empty() {
            return false;
        }

        let idx = self
            .slash_suggestion_idx
            .min(suggestions.len().saturating_sub(1));
        let selected = suggestions[idx];
        self.input = if selected.takes_argument {
            format!("{} ", selected.command)
        } else {
            selected.command.to_string()
        };
        self.cursor = self.input.len();
        self.slash_suggestion_idx = 0;
        true
    }

    fn handle_key(&mut self, key: KeyEvent) -> Result<bool> {
        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
            return Ok(true);
        }

        if self.modal.is_some() {
            return self.handle_modal_key(key);
        }

        if key.modifiers.contains(KeyModifiers::CONTROL) {
            match key.code {
                KeyCode::Char('n') => {
                    self.create_new_conversation()?;
                    return Ok(false);
                }
                KeyCode::Char('r') => {
                    self.reload_conversations()?;
                    self.status = "Conversations reloaded".to_string();
                    return Ok(false);
                }
                KeyCode::Char('s') => {
                    self.conversation.save()?;
                    self.reload_conversations()?;
                    self.status = format!("Saved {}", self.conversation.id);
                    return Ok(false);
                }
                KeyCode::Char('m') => {
                    self.open_model_picker()?;
                    return Ok(false);
                }
                KeyCode::Char('p') => {
                    let model = self.current_model_base().to_string();
                    self.open_provider_picker(&model)?;
                    return Ok(false);
                }
                KeyCode::Char('g') => {
                    self.open_web_mode_picker();
                    return Ok(false);
                }
                KeyCode::Char('w') => {
                    self.modal = Some(Modal::WebSearch(WebSearchModal::new()));
                    return Ok(false);
                }
                KeyCode::Char('h') => {
                    self.modal = Some(Modal::Help);
                    return Ok(false);
                }
                _ => {}
            }
        }

        if key.code == KeyCode::Tab
            && self.focus == Focus::Input
            && self.modal.is_none()
            && self.has_active_slash_suggestions()
            && self.can_autocomplete_slash()
        {
            if self.apply_selected_slash_suggestion() {
                self.status = "Slash command autocompleted".to_string();
                return Ok(false);
            }
        }

        match key.code {
            KeyCode::Esc => return Ok(true),
            KeyCode::Tab => {
                self.focus = match self.focus {
                    Focus::Input => Focus::Conversations,
                    Focus::Conversations => Focus::Input,
                };
            }
            _ => match self.focus {
                Focus::Conversations => self.handle_conversations_key(key)?,
                Focus::Input => self.handle_input_key(key)?,
            },
        }

        Ok(false)
    }

    fn handle_conversations_key(&mut self, key: KeyEvent) -> Result<()> {
        match key.code {
            KeyCode::Up => {
                self.selected_conversation_idx = self.selected_conversation_idx.saturating_sub(1)
            }
            KeyCode::Down => {
                if !self.conversations.is_empty() {
                    self.selected_conversation_idx =
                        (self.selected_conversation_idx + 1).min(self.conversations.len() - 1);
                }
            }
            KeyCode::Enter => self.switch_to_selected_conversation()?,
            KeyCode::Char('d') | KeyCode::Delete => self.delete_selected_conversation()?,
            _ => {}
        }
        Ok(())
    }

    fn handle_input_key(&mut self, key: KeyEvent) -> Result<()> {
        match key.code {
            KeyCode::Enter => {
                if self.pending {
                    self.status = "Request in progress".to_string();
                    return Ok(());
                }

                if self.has_active_slash_suggestions()
                    && self.can_autocomplete_slash()
                    && !self.slash_token_is_exact_command()
                {
                    if self.apply_selected_slash_suggestion() {
                        self.status = "Slash command autocompleted".to_string();
                        return Ok(());
                    }
                }

                let input = self.input.trim().to_string();
                self.input.clear();
                self.cursor = 0;
                self.slash_suggestion_idx = 0;

                if input.is_empty() {
                    return Ok(());
                }

                if input.starts_with('/') {
                    self.handle_command(&input)?;
                } else {
                    self.send_message(input)?;
                }
            }
            KeyCode::Char(c) => {
                self.input.insert(self.cursor, c);
                self.cursor += 1;
                self.sync_slash_suggestion_index();
            }
            KeyCode::Backspace => {
                if self.cursor > 0 {
                    self.cursor -= 1;
                    self.input.remove(self.cursor);
                    self.sync_slash_suggestion_index();
                }
            }
            KeyCode::Delete => {
                if self.cursor < self.input.len() {
                    self.input.remove(self.cursor);
                    self.sync_slash_suggestion_index();
                }
            }
            KeyCode::Left => {
                self.cursor = self.cursor.saturating_sub(1);
            }
            KeyCode::Right => {
                self.cursor = (self.cursor + 1).min(self.input.len());
            }
            KeyCode::Up => {
                if self.has_active_slash_suggestions() {
                    self.move_slash_suggestion(-1);
                } else {
                    self.chat_scroll = self.chat_scroll.saturating_sub(1);
                }
            }
            KeyCode::Down => {
                if self.has_active_slash_suggestions() {
                    self.move_slash_suggestion(1);
                } else {
                    self.chat_scroll = self.chat_scroll.saturating_add(1);
                }
            }
            _ => {}
        }

        Ok(())
    }

    fn handle_modal_key(&mut self, key: KeyEvent) -> Result<bool> {
        let Some(modal) = self.modal.clone() else {
            return Ok(false);
        };

        match modal {
            Modal::ApiKey(mut state) => {
                if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('r') {
                    state.reveal = !state.reveal;
                    self.modal = Some(Modal::ApiKey(state));
                    return Ok(false);
                }

                match key.code {
                    KeyCode::Enter => {
                        let key = state.value.trim().to_string();
                        if key.is_empty() {
                            self.status = "API key cannot be empty".to_string();
                        } else {
                            self.client.set_api_key(key.clone());
                            let mut cfg = load_config().unwrap_or_else(|_| AppConfig::default());
                            cfg.api_key = Some(key);
                            save_config(&cfg)?;
                            self.modal = None;
                            self.status = "API key saved".to_string();
                        }
                    }
                    KeyCode::Esc => {
                        self.status = "API key required to continue".to_string();
                        self.modal = Some(Modal::ApiKey(state));
                        return Ok(false);
                    }
                    KeyCode::Char(c) => {
                        state.value.insert(state.cursor, c);
                        state.cursor += 1;
                    }
                    KeyCode::Backspace => {
                        if state.cursor > 0 {
                            state.cursor -= 1;
                            state.value.remove(state.cursor);
                        }
                    }
                    KeyCode::Delete => {
                        if state.cursor < state.value.len() {
                            state.value.remove(state.cursor);
                        }
                    }
                    KeyCode::Left => {
                        state.cursor = state.cursor.saturating_sub(1);
                    }
                    KeyCode::Right => {
                        state.cursor = (state.cursor + 1).min(state.value.len());
                    }
                    _ => {}
                }
                if self.modal.is_some() {
                    self.modal = Some(Modal::ApiKey(state));
                }
            }
            Modal::ModelPicker(mut state) => {
                match key.code {
                    KeyCode::Esc => {
                        self.modal = None;
                        return Ok(false);
                    }
                    KeyCode::Tab => {
                        state.scope = match state.scope {
                            ModelScope::Canonical => ModelScope::Subscription,
                            ModelScope::Subscription => ModelScope::Paid,
                            ModelScope::Paid => ModelScope::Canonical,
                        };
                        state.models = self.fetch_models(state.scope)?;
                        state.selected = 0;
                    }
                    KeyCode::Up => state.selected = state.selected.saturating_sub(1),
                    KeyCode::Down => {
                        let len = state.filtered_models().len();
                        if len > 0 {
                            state.selected = (state.selected + 1).min(len - 1);
                        }
                    }
                    KeyCode::Enter => {
                        if let Some(model) = state.filtered_models().get(state.selected) {
                            self.current_model = apply_web_mode(&model.id, &self.web_mode);
                            self.conversation.model = self.current_model.clone();
                            let mut cfg = load_config().unwrap_or_else(|_| AppConfig::default());
                            cfg.default_model = Some(model.id.clone());
                            save_config(&cfg)?;
                            self.status = format!("Model set: {}", self.current_model);
                            self.modal = None;
                            return Ok(false);
                        }
                    }
                    KeyCode::Char('p') | KeyCode::Char('P') => {
                        if let Some(model) = state.filtered_models().get(state.selected) {
                            self.open_provider_picker(&model.id)?;
                            return Ok(false);
                        }
                    }
                    KeyCode::Char(c) => {
                        state.search.insert(state.cursor, c);
                        state.cursor += 1;
                        state.selected = 0;
                    }
                    KeyCode::Backspace => {
                        if state.cursor > 0 {
                            state.cursor -= 1;
                            state.search.remove(state.cursor);
                            state.selected = 0;
                        }
                    }
                    KeyCode::Delete => {
                        if state.cursor < state.search.len() {
                            state.search.remove(state.cursor);
                            state.selected = 0;
                        }
                    }
                    KeyCode::Left => {
                        state.cursor = state.cursor.saturating_sub(1);
                    }
                    KeyCode::Right => {
                        state.cursor = (state.cursor + 1).min(state.search.len());
                    }
                    _ => {}
                }
                self.modal = Some(Modal::ModelPicker(state));
            }
            Modal::ProviderPicker(mut state) => {
                match key.code {
                    KeyCode::Esc => {
                        self.modal = None;
                        return Ok(false);
                    }
                    KeyCode::Char('c') | KeyCode::Char('C') => {
                        self.provider_overrides.remove(&state.model_id);
                        self.persist_provider_overrides()?;
                        self.status = format!("Provider override cleared for {}", state.model_id);
                        self.modal = None;
                        return Ok(false);
                    }
                    KeyCode::Up => state.selected = state.selected.saturating_sub(1),
                    KeyCode::Down => {
                        if !state.providers.is_empty() {
                            state.selected = (state.selected + 1).min(state.providers.len() - 1);
                        }
                    }
                    KeyCode::Enter => {
                        if let Some(provider) = state.providers.get(state.selected) {
                            self.provider_overrides
                                .insert(state.model_id.clone(), provider.provider.clone());
                            self.persist_provider_overrides()?;
                            self.status = format!(
                                "Provider override for {}: {}",
                                state.model_id, provider.provider
                            );
                            self.modal = None;
                            return Ok(false);
                        }
                    }
                    _ => {}
                }
                self.modal = Some(Modal::ProviderPicker(state));
            }
            Modal::WebModePicker(mut state) => {
                let max_idx = web_mode_presets().len();
                match key.code {
                    KeyCode::Esc => {
                        self.modal = None;
                        return Ok(false);
                    }
                    KeyCode::Char('c') | KeyCode::Char('C') => {
                        self.web_mode = WebMode::Off;
                        let base = self.current_model_base().to_string();
                        self.current_model = apply_web_mode(&base, &self.web_mode);
                        self.conversation.model = self.current_model.clone();
                        self.status = "Web mode set: off".to_string();
                        self.modal = None;
                        return Ok(false);
                    }
                    KeyCode::Up => state.selected = state.selected.saturating_sub(1),
                    KeyCode::Down => {
                        state.selected = (state.selected + 1).min(max_idx);
                    }
                    KeyCode::Enter => {
                        self.web_mode = if state.selected == 0 {
                            WebMode::Off
                        } else {
                            let preset = web_mode_presets()
                                .get(state.selected.saturating_sub(1))
                                .copied()
                                .unwrap_or(web_mode_presets()[0]);
                            web_mode_from_key(preset.key)
                        };

                        let base = self.current_model_base().to_string();
                        self.current_model = apply_web_mode(&base, &self.web_mode);
                        self.conversation.model = self.current_model.clone();
                        self.status = format!("Web mode set: {}", web_mode_display(&self.web_mode));
                        self.modal = None;
                        return Ok(false);
                    }
                    _ => {}
                }
                self.modal = Some(Modal::WebModePicker(state));
            }
            Modal::WebSearch(mut state) => {
                match key.code {
                    KeyCode::Esc => {
                        self.modal = None;
                        return Ok(false);
                    }
                    KeyCode::Tab => state.cycle_provider(),
                    KeyCode::Char('d') | KeyCode::Char('D') => state.cycle_depth(),
                    KeyCode::Char('o') | KeyCode::Char('O') => state.cycle_output(),
                    KeyCode::Char('i') | KeyCode::Char('I') => {
                        state.include_images = !state.include_images
                    }
                    KeyCode::Enter => {
                        let query = state.query.trim().to_string();
                        if query.is_empty() {
                            self.status = "Web search query cannot be empty".to_string();
                        } else {
                            self.status = format!(
                                "Searching with provider={} depth={}...",
                                state.provider().as_str(),
                                state.depth()
                            );

                            let mut body = Map::new();
                            body.insert("query".to_string(), json!(query));
                            body.insert("provider".to_string(), json!(state.provider().as_str()));
                            body.insert("depth".to_string(), json!(state.depth()));
                            body.insert("outputType".to_string(), json!(state.output_type()));
                            body.insert("includeImages".to_string(), json!(state.include_images));

                            if state.output_type() == "structured" {
                                body.insert(
                                    "structuredOutputSchema".to_string(),
                                    json!("{\"type\":\"object\",\"properties\":{\"items\":{\"type\":\"array\"}}}"),
                                );
                            }

                            match self.client.request_json(
                                Method::POST,
                                "/web",
                                &[],
                                &[],
                                Some(Value::Object(body)),
                            ) {
                                Ok(value) => {
                                    state.result = serde_json::to_string_pretty(&value)
                                        .unwrap_or_else(|_| value.to_string());
                                    state.result_scroll = 0;
                                    self.status = "Web search completed".to_string();
                                }
                                Err(err) => {
                                    state.result = format!("Error: {err}");
                                    self.status = "Web search failed".to_string();
                                }
                            }
                        }
                    }
                    KeyCode::Char(c) => {
                        state.query.insert(state.cursor, c);
                        state.cursor += 1;
                    }
                    KeyCode::Backspace => {
                        if state.cursor > 0 {
                            state.cursor -= 1;
                            state.query.remove(state.cursor);
                        }
                    }
                    KeyCode::Delete => {
                        if state.cursor < state.query.len() {
                            state.query.remove(state.cursor);
                        }
                    }
                    KeyCode::Left => state.cursor = state.cursor.saturating_sub(1),
                    KeyCode::Right => state.cursor = (state.cursor + 1).min(state.query.len()),
                    KeyCode::Up => state.result_scroll = state.result_scroll.saturating_sub(1),
                    KeyCode::Down => state.result_scroll = state.result_scroll.saturating_add(1),
                    _ => {}
                }
                self.modal = Some(Modal::WebSearch(state));
            }
            Modal::Help => {
                if key.code == KeyCode::Esc {
                    self.modal = None;
                }
            }
        }

        Ok(false)
    }

    fn handle_command(&mut self, input: &str) -> Result<()> {
        let mut split = input.splitn(2, ' ');
        let cmd = split.next().unwrap_or_default();
        let arg = split.next().map(str::trim).unwrap_or("");

        match cmd {
            "/help" => self.modal = Some(Modal::Help),
            "/model" => {
                if arg.is_empty() {
                    self.status = format!("Current model: {}", self.current_model);
                } else {
                    self.current_model = apply_web_mode(arg, &self.web_mode);
                    self.conversation.model = self.current_model.clone();
                    self.status = format!("Model set: {}", self.current_model);
                }
            }
            "/webmode" => {
                if arg.is_empty() {
                    self.open_web_mode_picker();
                    return Ok(());
                }

                match parse_web_mode_arg(arg) {
                    Ok(next) => self.web_mode = next,
                    Err(usage) => {
                        self.status = usage;
                        return Ok(());
                    }
                }
                let base = self.current_model_base().to_string();
                self.current_model = apply_web_mode(&base, &self.web_mode);
                self.conversation.model = self.current_model.clone();
                self.status = format!("Web mode set: {}", web_mode_display(&self.web_mode));
            }
            "/system" => {
                if arg.eq_ignore_ascii_case("off") {
                    self.conversation.system_prompt = None;
                    self.status = "System prompt cleared".to_string();
                } else if arg.is_empty() {
                    self.status = match &self.conversation.system_prompt {
                        Some(v) => format!("System prompt: {v}"),
                        None => "System prompt is empty".to_string(),
                    };
                } else {
                    self.conversation.system_prompt = Some(arg.to_string());
                    self.status = "System prompt updated".to_string();
                }
            }
            "/clear" => {
                self.conversation.clear_history();
                self.status = "Conversation history cleared".to_string();
            }
            "/save" => {
                self.conversation.save()?;
                self.reload_conversations()?;
                self.status = format!("Saved {}", self.conversation.id);
            }
            "/history" => {
                self.status = format!("History messages: {}", self.conversation.messages.len());
            }
            "/models" => {
                self.open_model_picker()?;
            }
            "/providers" => {
                let model = self.current_model_base().to_string();
                self.open_provider_picker(&model)?;
            }
            _ => {
                self.status = format!("Unknown command: {cmd}");
            }
        }

        Ok(())
    }

    fn send_message(&mut self, input: String) -> Result<()> {
        if !self.client.has_api_key() {
            self.modal = Some(Modal::ApiKey(ApiKeyModal {
                value: String::new(),
                cursor: 0,
                reveal: false,
            }));
            self.status = "API key required".to_string();
            return Ok(());
        }

        self.pending = true;
        self.status = "Waiting for model response...".to_string();
        self.pending_user = Some(input.clone());

        let mut outgoing = self.conversation.messages.clone();
        outgoing.push(ConversationMessage {
            role: "user".to_string(),
            content: input,
        });

        let request = ChatRequest {
            model: self.current_model.clone(),
            system_prompt: self.conversation.system_prompt.clone(),
            messages: outgoing,
            temperature: self.args.temperature,
            max_tokens: self.args.max_tokens,
            top_p: self.args.top_p,
            service_tier: self.args.service_tier.clone(),
            reasoning_effort: self.args.reasoning_effort.clone(),
            billing_mode: self.args.billing_mode.clone(),
            provider: self
                .active_provider_for_current_model()
                .map(|v| v.to_string()),
        };

        let (tx, rx) = mpsc::channel();
        let client = self.client.clone();

        std::thread::spawn(move || {
            let result = client.chat_completion(&request).map(|r| r.content);
            let _ = tx.send(result);
        });

        self.response_rx = Some(rx);
        Ok(())
    }

    fn poll_response(&mut self) -> Result<()> {
        if !self.pending {
            return Ok(());
        }

        if let Some(rx) = &self.response_rx {
            match rx.try_recv() {
                Ok(result) => {
                    self.pending = false;
                    let user = self.pending_user.take().unwrap_or_default();
                    self.response_rx = None;

                    match result {
                        Ok(content) => {
                            self.conversation.push_user_message(user);
                            self.conversation.push_assistant_message(content);
                            self.conversation.model = self.current_model.clone();
                            self.conversation.save()?;
                            self.reload_conversations()?;
                            self.status = "Response received".to_string();
                        }
                        Err(err) => {
                            self.status = format!("Request failed: {err}");
                        }
                    }
                }
                Err(mpsc::TryRecvError::Empty) => {}
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.pending = false;
                    self.pending_user = None;
                    self.response_rx = None;
                    self.status = "Request channel disconnected".to_string();
                }
            }
        }

        Ok(())
    }

    fn reload_conversations(&mut self) -> Result<()> {
        self.conversations = conv_store::list()?;

        if self.conversations.is_empty() {
            self.selected_conversation_idx = 0;
            return Ok(());
        }

        self.selected_conversation_idx = self
            .conversations
            .iter()
            .position(|c| c.id == self.conversation.id)
            .unwrap_or(0);
        Ok(())
    }

    fn switch_to_selected_conversation(&mut self) -> Result<()> {
        if self.pending {
            self.status = "Wait for current request to finish".to_string();
            return Ok(());
        }

        let Some(summary) = self.conversations.get(self.selected_conversation_idx) else {
            return Ok(());
        };

        if summary.id == self.conversation.id {
            return Ok(());
        }

        let conv = conv_store::load(&summary.id)?;
        self.current_model = if conv.model.is_empty() {
            self.current_model.clone()
        } else {
            conv.model.clone()
        };
        self.web_mode = infer_from_model(&self.current_model).unwrap_or(self.web_mode.clone());
        self.conversation = conv;
        self.status = format!("Switched to {}", self.conversation.id);
        Ok(())
    }

    fn create_new_conversation(&mut self) -> Result<()> {
        if self.pending {
            self.status = "Wait for current request to finish".to_string();
            return Ok(());
        }

        self.conversation.save()?;
        let mut conv = Conversation::load_or_create(
            None,
            self.current_model.clone(),
            self.conversation.system_prompt.clone(),
        )?;
        conv.model = self.current_model.clone();
        conv.save()?;
        self.conversation = conv;
        self.reload_conversations()?;
        self.focus = Focus::Input;
        self.status = format!("Created {}", self.conversation.id);
        Ok(())
    }

    fn delete_selected_conversation(&mut self) -> Result<()> {
        let Some(summary) = self
            .conversations
            .get(self.selected_conversation_idx)
            .cloned()
        else {
            return Ok(());
        };

        conv_store::delete(&summary.id)?;

        if summary.id == self.conversation.id {
            let mut conv = Conversation::load_or_create(
                None,
                self.current_model.clone(),
                self.conversation.system_prompt.clone(),
            )?;
            conv.save()?;
            self.conversation = conv;
        }

        self.reload_conversations()?;
        self.status = format!("Deleted {}", summary.id);
        Ok(())
    }

    fn open_web_mode_picker(&mut self) {
        let selected = match web_mode_key(&self.web_mode) {
            None => 0,
            Some(key) => web_mode_presets()
                .iter()
                .position(|preset| preset.key == key)
                .map(|idx| idx + 1)
                .unwrap_or(1),
        };
        self.modal = Some(Modal::WebModePicker(WebModePickerModal { selected }));
    }

    fn open_model_picker(&mut self) -> Result<()> {
        let scope = ModelScope::Canonical;
        let models = self.fetch_models(scope)?;

        self.modal = Some(Modal::ModelPicker(ModelPickerModal {
            scope,
            search: String::new(),
            cursor: 0,
            selected: 0,
            models,
        }));
        Ok(())
    }

    fn open_provider_picker(&mut self, model_id: &str) -> Result<()> {
        let encoded = encode_path_segment(model_id);
        let path = format!("/models/{encoded}/providers");
        let value = self
            .client
            .request_json(Method::GET, &path, &[], &[], None)
            .with_context(|| format!("failed to load providers for {model_id}"))?;

        let supports = value
            .get("supportsProviderSelection")
            .and_then(Value::as_bool)
            .unwrap_or(false);

        let providers = value
            .get("providers")
            .and_then(Value::as_array)
            .map(|arr| {
                arr.iter()
                    .map(|item| ProviderEntry {
                        provider: item
                            .get("provider")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string(),
                        available: item
                            .get("available")
                            .and_then(Value::as_bool)
                            .unwrap_or(false),
                        input_per_1k: item
                            .pointer("/pricing/inputPer1kTokens")
                            .and_then(Value::as_f64),
                        output_per_1k: item
                            .pointer("/pricing/outputPer1kTokens")
                            .and_then(Value::as_f64),
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        let message = value
            .get("message")
            .and_then(Value::as_str)
            .map(|v| v.to_string());

        let selected = self
            .provider_overrides
            .get(model_id)
            .and_then(|selected_provider| {
                providers
                    .iter()
                    .position(|p| p.provider.as_str() == selected_provider.as_str())
            })
            .unwrap_or(0);

        self.modal = Some(Modal::ProviderPicker(ProviderPickerModal {
            model_id: model_id.to_string(),
            selected,
            supports_provider_selection: supports,
            providers,
            message,
        }));

        Ok(())
    }

    fn fetch_models(&self, scope: ModelScope) -> Result<Vec<ModelEntry>> {
        let path = match scope {
            ModelScope::Canonical => "/v1/models",
            ModelScope::Subscription => "/subscription/v1/models",
            ModelScope::Paid => "/paid/v1/models",
        };

        let query = vec![("detailed".to_string(), "true".to_string())];
        let value = self
            .client
            .request_json(Method::GET, path, &query, &[], None)?;
        let data = value
            .get("data")
            .and_then(Value::as_array)
            .ok_or_else(|| anyhow!("unexpected model response format"))?;

        let mut models = data
            .iter()
            .map(|item| ModelEntry {
                id: item
                    .get("id")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                name: item
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                owned_by: item
                    .get("owned_by")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                category: item
                    .get("category")
                    .and_then(Value::as_str)
                    .map(|v| v.to_string()),
                subscription_included: item
                    .pointer("/subscription/included")
                    .and_then(Value::as_bool),
            })
            .collect::<Vec<_>>();

        models.sort_by(|a, b| a.id.cmp(&b.id));
        Ok(models)
    }

    fn active_provider_for_current_model(&self) -> Option<&str> {
        let base = self.current_model_base();
        self.provider_overrides.get(base).map(|v| v.as_str())
    }

    fn persist_provider_overrides(&self) -> Result<()> {
        let mut cfg = load_config().unwrap_or_else(|_| AppConfig::default());
        cfg.provider_overrides = self.provider_overrides.clone();
        save_config(&cfg)
    }

    fn current_model_base(&self) -> &str {
        current_model_base_from_value(&self.current_model)
    }
}

impl ModelPickerModal {
    fn filtered_models(&self) -> Vec<ModelEntry> {
        let q = self.search.trim().to_lowercase();
        if q.is_empty() {
            return self.models.clone();
        }

        self.models
            .iter()
            .filter(|m| {
                m.id.to_lowercase().contains(&q)
                    || m.name.to_lowercase().contains(&q)
                    || m.owned_by.to_lowercase().contains(&q)
                    || m.category
                        .as_ref()
                        .map(|v| v.to_lowercase().contains(&q))
                        .unwrap_or(false)
            })
            .cloned()
            .collect()
    }
}

fn indent_line(line: Line<'static>, indent: &str, style: Style) -> Line<'static> {
    let mut spans = Vec::with_capacity(line.spans.len() + 1);
    spans.push(Span::styled(indent.to_string(), style));
    spans.extend(line.spans);
    Line::from(spans)
}

fn render_markdown_block(content: &str, theme: UiTheme, base: Style) -> Vec<Line<'static>> {
    if content.is_empty() {
        return vec![];
    }

    let mut lines: Vec<Line<'static>> = Vec::new();
    let mut in_code_block = false;

    for raw_line in content.split('\n') {
        let trimmed = raw_line.trim_start();

        if trimmed.starts_with("```") {
            if in_code_block {
                in_code_block = false;
            } else {
                in_code_block = true;
                let lang = trimmed.trim_start_matches("```").trim();
                if !lang.is_empty() {
                    lines.push(Line::from(vec![Span::styled(
                        format!("[code: {lang}]"),
                        Style::default()
                            .fg(theme.muted)
                            .add_modifier(Modifier::ITALIC),
                    )]));
                }
            }
            continue;
        }

        if in_code_block {
            lines.push(Line::from(vec![
                Span::styled("│ ", Style::default().fg(theme.muted)),
                Span::styled(
                    raw_line.to_string(),
                    Style::default().fg(theme.accent_2).bg(theme.panel_alt),
                ),
            ]));
            continue;
        }

        if trimmed.is_empty() {
            lines.push(Line::raw(""));
            continue;
        }

        let (quote_depth, body) = split_blockquote_prefix(trimmed);
        let mut spans: Vec<Span<'static>> = Vec::new();

        if quote_depth > 0 {
            spans.push(Span::styled(
                format!("{} ", ">".repeat(quote_depth)),
                Style::default().fg(theme.muted),
            ));
        }

        if is_markdown_rule(body) {
            spans.push(Span::styled(
                "--------------------",
                Style::default().fg(theme.muted),
            ));
            lines.push(Line::from(spans));
            continue;
        }

        if let Some((level, heading_text)) = split_heading_marker(body) {
            let heading_style = Style::default()
                .fg(theme.warning)
                .add_modifier(Modifier::BOLD);
            spans.push(Span::styled(
                format!("{} ", "#".repeat(level)),
                heading_style,
            ));
            spans.extend(parse_inline_markdown(heading_text, theme, heading_style));
            lines.push(Line::from(spans));
            continue;
        }

        if let Some((marker, item_text)) = split_ordered_list_marker(body) {
            spans.push(Span::styled(
                format!("{marker} "),
                Style::default()
                    .fg(theme.accent)
                    .add_modifier(Modifier::BOLD),
            ));
            spans.extend(parse_inline_markdown(item_text, theme, base));
            lines.push(Line::from(spans));
            continue;
        }

        if let Some(item_text) = split_unordered_list_marker(body) {
            spans.push(Span::styled(
                "- ",
                Style::default()
                    .fg(theme.accent)
                    .add_modifier(Modifier::BOLD),
            ));
            spans.extend(parse_inline_markdown(item_text, theme, base));
            lines.push(Line::from(spans));
            continue;
        }

        spans.extend(parse_inline_markdown(body, theme, base));
        lines.push(Line::from(spans));
    }

    lines
}

fn parse_inline_markdown(text: &str, theme: UiTheme, base: Style) -> Vec<Span<'static>> {
    if text.is_empty() {
        return vec![];
    }

    let mut spans: Vec<Span<'static>> = Vec::new();
    let mut buffer = String::new();
    let mut idx = 0usize;

    let mut strong = false;
    let mut emphasis = false;
    let mut strike = false;
    let mut code = false;

    let flush = |spans: &mut Vec<Span<'static>>,
                 buffer: &mut String,
                 strong: bool,
                 emphasis: bool,
                 strike: bool,
                 code: bool| {
        if buffer.is_empty() {
            return;
        }
        let style = markdown_inline_style(base, theme, strong, emphasis, strike, code, false);
        spans.push(Span::styled(std::mem::take(buffer), style));
    };

    while idx < text.len() {
        let tail = &text[idx..];

        if !code && tail.starts_with('[') {
            if let Some((label, url, consumed)) = parse_markdown_link(tail) {
                flush(&mut spans, &mut buffer, strong, emphasis, strike, code);
                let link_style =
                    markdown_inline_style(base, theme, strong, emphasis, strike, false, true);
                spans.push(Span::styled(label.to_string(), link_style));
                if !url.is_empty() {
                    spans.push(Span::styled(
                        format!(" ({url})"),
                        Style::default().fg(theme.muted),
                    ));
                }
                idx += consumed;
                continue;
            }
        }

        if !code && tail.starts_with("**") {
            flush(&mut spans, &mut buffer, strong, emphasis, strike, code);
            strong = !strong;
            idx += 2;
            continue;
        }

        if !code && tail.starts_with("~~") {
            flush(&mut spans, &mut buffer, strong, emphasis, strike, code);
            strike = !strike;
            idx += 2;
            continue;
        }

        if !code && tail.starts_with('*') {
            flush(&mut spans, &mut buffer, strong, emphasis, strike, code);
            emphasis = !emphasis;
            idx += 1;
            continue;
        }

        if tail.starts_with('`') {
            flush(&mut spans, &mut buffer, strong, emphasis, strike, code);
            code = !code;
            idx += 1;
            continue;
        }

        if let Some(ch) = tail.chars().next() {
            buffer.push(ch);
            idx += ch.len_utf8();
        } else {
            break;
        }
    }

    flush(&mut spans, &mut buffer, strong, emphasis, strike, code);
    spans
}

fn markdown_inline_style(
    base: Style,
    theme: UiTheme,
    strong: bool,
    emphasis: bool,
    strike: bool,
    code: bool,
    link: bool,
) -> Style {
    let mut style = base;

    if strong {
        style = style.add_modifier(Modifier::BOLD);
    }
    if emphasis {
        style = style.add_modifier(Modifier::ITALIC);
    }
    if strike {
        style = style.add_modifier(Modifier::CROSSED_OUT);
    }
    if link {
        style = style
            .fg(theme.accent)
            .add_modifier(Modifier::UNDERLINED | Modifier::BOLD);
    }
    if code {
        style = style.fg(theme.accent_2).bg(theme.panel_alt);
    }

    style
}

fn parse_markdown_link(tail: &str) -> Option<(&str, &str, usize)> {
    let remainder = tail.strip_prefix('[')?;
    let label_end_rel = remainder.find(']')?;
    let label_end = label_end_rel + 1;
    if !tail.get(label_end + 1..)?.starts_with('(') {
        return None;
    }

    let url_start = label_end + 2;
    let url_end_rel = tail.get(url_start..)?.find(')')?;
    let url_end = url_start + url_end_rel;

    let label = tail.get(1..label_end)?;
    let url = tail.get(url_start..url_end)?;
    let consumed = url_end + 1;
    Some((label, url, consumed))
}

fn split_blockquote_prefix(input: &str) -> (usize, &str) {
    let mut depth = 0usize;
    let mut rest = input.trim_start();

    loop {
        if let Some(next) = rest.strip_prefix('>') {
            depth += 1;
            rest = next.trim_start();
        } else {
            break;
        }
    }

    (depth, rest)
}

fn split_heading_marker(input: &str) -> Option<(usize, &str)> {
    let bytes = input.as_bytes();
    let mut level = 0usize;
    while level < bytes.len() && bytes[level] == b'#' && level < 6 {
        level += 1;
    }

    if level == 0 || bytes.get(level) != Some(&b' ') {
        return None;
    }

    Some((level, &input[level + 1..]))
}

fn split_unordered_list_marker(input: &str) -> Option<&str> {
    input
        .strip_prefix("- ")
        .or_else(|| input.strip_prefix("* "))
        .or_else(|| input.strip_prefix("+ "))
}

fn split_ordered_list_marker(input: &str) -> Option<(String, &str)> {
    let mut end = 0usize;
    for ch in input.chars() {
        if ch.is_ascii_digit() {
            end += ch.len_utf8();
        } else {
            break;
        }
    }

    if end == 0 || !input.get(end..)?.starts_with(". ") {
        return None;
    }

    let marker = input.get(..end)?.to_string() + ".";
    let remainder = input.get(end + 2..)?;
    Some((marker, remainder))
}

fn is_markdown_rule(input: &str) -> bool {
    let compact = input
        .chars()
        .filter(|c| !c.is_whitespace())
        .collect::<String>();
    if compact.len() < 3 {
        return false;
    }

    let mut chars = compact.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if first != '-' && first != '*' && first != '_' {
        return false;
    }

    chars.all(|c| c == first)
}

fn init_terminal() -> Result<Terminal<CrosstermBackend<io::Stdout>>> {
    enable_raw_mode().context("failed to enable raw mode")?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen).context("failed to enter alternate screen")?;
    let backend = CrosstermBackend::new(stdout);
    Terminal::new(backend).context("failed to initialize terminal")
}

fn restore_terminal(terminal: &mut Terminal<CrosstermBackend<io::Stdout>>) -> Result<()> {
    disable_raw_mode().ok();
    execute!(terminal.backend_mut(), LeaveAlternateScreen).ok();
    terminal.show_cursor().ok();
    Ok(())
}

fn centered_rect(area: Rect, width_percent: u16, height_percent: u16) -> Rect {
    let vertical = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - height_percent) / 2),
            Constraint::Percentage(height_percent),
            Constraint::Percentage((100 - height_percent) / 2),
        ])
        .split(area);

    let horizontal = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - width_percent) / 2),
            Constraint::Percentage(width_percent),
            Constraint::Percentage((100 - width_percent) / 2),
        ])
        .split(vertical[1]);

    horizontal[1]
}

fn encode_path_segment(value: &str) -> String {
    byte_serialize(value.as_bytes()).collect::<String>()
}

fn current_model_base_from_value(model: &str) -> &str {
    model
        .split_once(":online")
        .map(|(left, _)| left)
        .unwrap_or(model)
}

fn truncate_middle(value: &str, max_chars: usize) -> String {
    let chars = value.chars().collect::<Vec<_>>();
    if chars.len() <= max_chars {
        return value.to_string();
    }

    if max_chars <= 3 {
        return "...".to_string();
    }

    let left = (max_chars - 3) / 2;
    let right = max_chars - 3 - left;
    let start = chars[..left].iter().collect::<String>();
    let end = chars[chars.len().saturating_sub(right)..]
        .iter()
        .collect::<String>();
    format!("{start}...{end}")
}
