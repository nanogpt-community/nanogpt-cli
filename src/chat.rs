use std::io::{self, Write};

use anyhow::{Context, Result};

use crate::cli::ChatArgs;
use crate::client::{ChatRequest, NanoGptClient};
use crate::conversation::{Conversation, ConversationMessage};
use crate::model_mode::{
    WebMode, apply_web_mode, infer_from_flags, infer_from_model, parse_web_mode_arg,
    web_mode_display,
};

pub fn run_chat_repl(client: &NanoGptClient, args: ChatArgs) -> Result<()> {
    let initial_model = apply_web_mode(&args.model, &infer_from_flags(args.web, args.deep_web));
    let mut conversation = if let Some(id) = args.conversation.as_deref() {
        Conversation::load_or_create(Some(id), initial_model.clone(), args.system.clone())?
    } else {
        crate::conversation::delete_empty_conversations()?;
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

    let mut current_model = if conversation.model.is_empty() {
        apply_web_mode(&args.model, &infer_from_flags(args.web, args.deep_web))
    } else {
        conversation.model.clone()
    };
    let mut web_mode =
        infer_from_model(&current_model).unwrap_or(infer_from_flags(args.web, args.deep_web));

    println!("NanoGPT CLI chat");
    println!("Conversation: {}", conversation.id);
    println!("Model: {current_model}");
    if let Some(system) = &conversation.system_prompt {
        println!("System prompt: {system}");
    }
    println!("Type /help for commands.\n");

    let mut stdin = String::new();
    loop {
        print!("you> ");
        io::stdout().flush().ok();

        stdin.clear();
        io::stdin()
            .read_line(&mut stdin)
            .context("failed to read input")?;
        let input = stdin.trim();

        if input.is_empty() {
            continue;
        }

        if input.starts_with('/') {
            if handle_command(
                input,
                &mut conversation,
                &mut current_model,
                &mut web_mode,
                args.system.as_ref(),
            )? {
                break;
            }
            continue;
        }

        let user_message = ConversationMessage {
            role: "user".to_string(),
            content: input.to_string(),
        };
        let mut outgoing = conversation.messages.clone();
        outgoing.push(user_message.clone());

        let chat_req = ChatRequest {
            model: current_model.clone(),
            system_prompt: conversation.system_prompt.clone(),
            messages: outgoing,
            temperature: args.temperature,
            max_tokens: args.max_tokens,
            top_p: args.top_p,
            service_tier: args.service_tier.clone(),
            reasoning_effort: args.reasoning_effort.clone(),
            billing_mode: args.billing_mode.clone(),
            provider: args.provider.clone(),
        };

        print!("assistant> ");
        io::stdout().flush().ok();

        let response = if args.stream {
            let streamed = client.chat_completion_stream(&chat_req, |delta| {
                print!("{delta}");
                io::stdout().flush().ok();
            });
            println!();
            streamed
        } else {
            let r = client.chat_completion(&chat_req);
            if let Ok(resp) = &r {
                println!("{}", resp.content);
            }
            r
        };

        match response {
            Ok(resp) => {
                conversation.model = current_model.clone();
                conversation.messages.push(user_message);
                conversation.push_assistant_message(resp.content);
                conversation.save()?;
            }
            Err(err) => {
                println!("[error] {err}");
            }
        }
    }

    conversation.save()?;
    Ok(())
}

fn handle_command(
    input: &str,
    conversation: &mut Conversation,
    current_model: &mut String,
    web_mode: &mut WebMode,
    initial_system_prompt: Option<&String>,
) -> Result<bool> {
    let mut parts = input.splitn(2, ' ');
    let cmd = parts.next().unwrap_or_default();
    let arg = parts.next().map(str::trim).unwrap_or("");

    match cmd {
        "/exit" | "/quit" => Ok(true),
        "/help" => {
            print_help();
            Ok(false)
        }
        "/model" => {
            if arg.is_empty() {
                println!("Current model: {current_model}");
            } else {
                *current_model = apply_web_mode(arg, web_mode);
                conversation.model = current_model.clone();
                println!("Model set to {current_model}");
            }
            Ok(false)
        }
        "/system" => {
            if arg.eq_ignore_ascii_case("off") {
                conversation.system_prompt = None;
                println!("System prompt cleared");
            } else if arg.is_empty() {
                match &conversation.system_prompt {
                    Some(s) => println!("System prompt: {s}"),
                    None => println!("System prompt is empty"),
                }
            } else {
                conversation.system_prompt = Some(arg.to_string());
                println!("System prompt updated");
            }
            Ok(false)
        }
        "/system-reset" => {
            conversation.system_prompt = initial_system_prompt.cloned();
            println!("System prompt reset");
            Ok(false)
        }
        "/webmode" => {
            if arg.is_empty() {
                println!("Web mode: {}", web_mode_display(web_mode));
                return Ok(false);
            }

            match parse_web_mode_arg(arg) {
                Ok(next) => *web_mode = next,
                Err(usage) => {
                    println!("{usage}");
                    return Ok(false);
                }
            }

            *current_model = apply_web_mode(current_model, web_mode);
            conversation.model = current_model.clone();
            println!(
                "Web mode: {} (model: {current_model})",
                web_mode_display(web_mode)
            );
            Ok(false)
        }
        "/clear" => {
            conversation.clear_history();
            println!("Conversation history cleared");
            Ok(false)
        }
        "/history" => {
            for (idx, msg) in conversation.messages.iter().enumerate() {
                println!("{} [{}] {}", idx + 1, msg.role, msg.content);
            }
            Ok(false)
        }
        "/save" => {
            conversation.save()?;
            println!("Conversation saved: {}", conversation.id);
            Ok(false)
        }
        _ => {
            println!("Unknown command: {cmd}. Type /help");
            Ok(false)
        }
    }
}

fn print_help() {
    println!("Commands:");
    println!("  /help                Show this help");
    println!("  /exit                Exit chat");
    println!("  /model <id>          Switch model");
    println!("  /webmode <mode>      Set web mode (off/on/deep/exa-neural/...)");
    println!("  /system <prompt>     Set or show system prompt");
    println!("  /system off          Clear system prompt");
    println!("  /system-reset        Restore initial --system value");
    println!("  /clear               Clear conversation messages");
    println!("  /history             Print conversation messages");
    println!("  /save                Persist conversation now");
}
