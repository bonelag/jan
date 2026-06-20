//! jan — headless CLI for Jan.
//!
//! Shares all core logic with the Jan desktop app.
//! Build with: cargo build --features cli --bin jan

use std::collections::HashMap;
use std::sync::Arc;
use std::io::Write;

use clap::{Args, CommandFactory, FromArgMatches, Parser, Subcommand};
use console::Style;
use indicatif::{ProgressBar, ProgressStyle};

// Import the library crate so we can access core modules.
// The lib target is named "app_lib" (see [lib] section in Cargo.toml).
use app_lib::core::cli::{
    cli_delete_thread, cli_get_data_folder, cli_get_thread,
    cli_list_messages, cli_list_threads, discover_llamacpp_binary,
    download_hf_model, fetch_hf_gguf_files, init_llamacpp_state,
    list_models, looks_like_hf_repo, resolve_model_engine, HfFileInfo,
};
// MLX is macOS-only; these CLI symbols don't exist on other platforms.
#[cfg(target_os = "macos")]
use app_lib::core::cli::{
    discover_mlx_binary, init_mlx_state, load_mlx_model_impl, resolve_model_by_id, MlxConfig,
};
use tauri_plugin_llamacpp::router as llamacpp_router;
use tauri_plugin_llamacpp::state::LlamacppState;
use std::path::PathBuf;

// ── Top-level CLI ──────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(
    name = "jan",
    about = "Serve local AI models and wire them to agents — no cloud required",
    long_about = "Jan runs local AI models (LlamaCPP / MLX) and exposes them via an\n\
OpenAI-compatible API, then wires AI coding agent like Claude Code\n\
directly to your own hardware — no cloud account, no usage fees, full privacy.\n\n\
Models downloaded in the Jan desktop app are automatically available here.",
    after_help = "Examples:\n  \
  jan launch claude                                      # pick a model, then run Claude Code against it\n  \
  jan launch claude --model janhq/Jan-code-4b-gguf       # use a specific model\n  \
  jan launch openclaw --model janhq/Jan-code-4b-gguf     # wire openclaw to a local model\n  \
  jan serve janhq/Jan-code-4b-gguf                       # expose a model at localhost:6767/v1\n  \
  jan serve janhq/Jan-code-4b-gguf --fit                 # auto-fit context to available VRAM\n  \
  jan serve janhq/Jan-code-4b-gguf --detach              # run in the background\n  \
  jan models list                                        # show all installed models",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Load a local model and expose it at localhost:6767/v1 (auto-detects LlamaCPP or MLX)
    #[command(display_order = 1)]
    Serve {
        #[command(flatten)]
        args: ServeArgs,
    },
    /// Load a local model and chat with it in a beautiful Terminal UI
    #[command(display_order = 3)]
    Tui {
        #[command(flatten)]
        args: ServeArgs,
    },
    /// Start a local model, then launch an AI agent with it pre-wired (env vars set automatically)
    #[command(display_order = 2)]
    Launch {
        /// Agent or program to run after the model is ready (e.g. claude, openclaw)
        /// Omit to pick interactively from: claude, openclaw
        program: Option<String>,
        /// Arguments forwarded to the program
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        program_args: Vec<String>,
        /// Model ID to load (omit to pick interactively)
        #[arg(long)]
        model: Option<String>,
        /// Path to the inference binary (auto-discovered from Jan data folder when omitted)
        #[arg(long)]
        bin: Option<String>,
        /// Port the model server listens on
        #[arg(long, default_value_t = 6767)]
        port: u16,
        /// API key for the model server (exported as OPENAI_API_KEY and ANTHROPIC_AUTH_TOKEN)
        #[arg(long, default_value = "jan")]
        api_key: String,
        /// GPU layers to offload (-1 = all layers, 0 = CPU only)
        #[arg(long, default_value_t = -1)]
        n_gpu_layers: i32,
        /// Context window size in tokens (default: 4096; disables --fit when set explicitly)
        #[arg(long)]
        ctx_size: Option<i32>,
        /// Auto-fit context to available VRAM (default: on when launching claude, unless --ctx-size is set)
        #[arg(long)]
        fit: Option<bool>,
        /// Print full server logs (llama.cpp / mlx output) instead of the loading spinner
        #[arg(long, short = 'v', default_value_t = false)]
        verbose: bool,
        /// When downloading a model, show quantization selection list
        #[arg(long, default_value_t = false)]
        select: bool,
    },
    /// List and inspect conversation threads saved by the Jan app
    #[command(display_order = 10)]
    Threads {
        #[command(subcommand)]
        cmd: ThreadsCommands,
    },
    /// List and load models installed in the Jan data folder
    #[command(display_order = 11)]
    Models {
        #[command(subcommand)]
        cmd: ModelsCommands,
    },
}


// ── Threads subcommands ────────────────────────────────────────────────────

#[derive(Subcommand)]
enum ThreadsCommands {
    /// Print all threads as JSON
    List,
    /// Print a single thread's metadata as JSON
    Get {
        /// Thread ID
        id: String,
    },
    /// Permanently delete a thread and all its messages
    Delete {
        /// Thread ID
        id: String,
    },
    /// Print all messages in a thread as JSON
    Messages {
        /// Thread ID
        thread_id: String,
    },
}

// ── Serve args (shared by `models load` and top-level `serve`) ────────────

#[derive(Args)]
struct ServeArgs {
    /// Model ID to load (omit to pick interactively from installed models)
    model_id: Option<String>,
    /// Path to the GGUF file (auto-resolved from model.yml when omitted)
    #[arg(long)]
    model_path: Option<String>,
    /// Path to the inference binary (auto-discovered from Jan data folder when omitted)
    #[arg(long)]
    bin: Option<String>,
    /// Port the model server listens on (0 = pick a random free port)
    #[arg(long, default_value_t = 6767)]
    port: u16,
    /// mmproj path for vision-language models (auto-resolved from model.yml when omitted)
    #[arg(long)]
    mmproj: Option<String>,
    /// Treat the model as an embedding model
    #[arg(long, default_value_t = false)]
    embedding: bool,
    /// Seconds to wait for the model server to become ready
    #[arg(long, default_value_t = 120)]
    timeout: u64,
    /// GPU layers to offload (-1 = all layers, 0 = CPU only)
    #[arg(long, default_value_t = -1)]
    n_gpu_layers: i32,
    /// Context window size in tokens (0 = model default)
    #[arg(long, default_value_t = 32768)]
    ctx_size: i32,
    /// Auto-fit context to available VRAM, maximising the context window
    #[arg(long, default_value_t = false)]
    fit: bool,
    /// CPU threads for inference (0 = auto-detect)
    #[arg(long, default_value_t = 0)]
    threads: i32,
    /// API key required by clients (sets LLAMA_API_KEY / MLX_API_KEY on the server)
    #[arg(long, default_value = "")]
    api_key: String,
    /// Run in the background (detach from terminal) and print the PID
    #[arg(long, short = 'd', default_value_t = false)]
    detach: bool,
    /// Log file for background mode (default: <data-folder>/logs/serve.log)
    #[arg(long)]
    log: Option<String>,
    /// Print full server logs (llama.cpp / mlx output) instead of the loading spinner
    #[arg(long, short = 'v', default_value_t = false)]
    verbose: bool,
    /// When downloading a model, show quantization selection list
    #[arg(long, default_value_t = false)]
    select: bool,
}

// ── Models subcommands ─────────────────────────────────────────────────────

#[derive(Subcommand)]
enum ModelsCommands {
    /// Print all installed models as JSON (from the Jan data folder)
    List {
        /// Filter by engine: llamacpp, mlx, or all
        #[arg(long, default_value = "all")]
        engine: String,
    },
    /// Load a model and serve it — alias for the top-level `serve` command
    Load {
        #[command(flatten)]
        args: ServeArgs,
    },
    /// Load an MLX model directly (macOS / Apple Silicon only)
    #[cfg(target_os = "macos")]
    LoadMlx {
        /// Model ID as shown by `jan models list --engine mlx`
        #[arg(long)]
        model_id: String,
        /// Path to the MLX model directory (auto-resolved from model.yml when omitted)
        #[arg(long)]
        model_path: Option<String>,
        /// Path to the mlx-server binary (auto-discovered from Jan.app when omitted)
        #[arg(long)]
        bin: Option<String>,
        /// Port the model server listens on (0 = pick a random free port)
        #[arg(long, default_value_t = 6767)]
        port: u16,
        /// Context window size in tokens (0 = model default)
        #[arg(long, default_value_t = 0)]
        ctx_size: i32,
        /// Treat the model as an embedding model
        #[arg(long, default_value_t = false)]
        embedding: bool,
        /// Seconds to wait for the model server to become ready
        #[arg(long, default_value_t = 120)]
        timeout: u64,
        /// API key required by clients (sets MLX_API_KEY on the server)
        #[arg(long, default_value = "")]
        api_key: String,
    },
}

// ── ASCII logo ─────────────────────────────────────────────────────────────

/// Build a left-aligned, bright-yellow ASCII logo for the help header.
fn make_logo() -> String {
    // "JAN" in ANSI Shadow block letters
    let lines = [
        r"     ██╗ █████╗ ███╗  ██╗",
        r"     ██║██╔══██╗████╗ ██║",
        r"     ██║███████║██╔██╗██║",
        r"██   ██║██╔══██║██║╚████║",
        r"╚█████╔╝██║  ██║██║ ╚███║",
        r" ╚════╝ ╚═╝  ╚═╝╚═╝  ╚══╝",
    ];

    // Fixed left-aligned indent (2 spaces)
    let indent = "  ";

    let yellow = Style::new().yellow().bold();

    let mut out: Vec<String> = Vec::new();

    // Add padding at top
    out.push(String::new());
    out.push(String::new());

    // Logo lines
    for l in &lines {
        out.push(format!("{}{}", indent, yellow.apply_to(l)));
    }

    out.join("\n")
}

// ── Entry point ────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    // Pre-scan raw args for --verbose / -v before full parse so we can set
    // the log level before any logging happens.
    let verbose = std::env::args().any(|a| a == "--verbose" || a == "-v");
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or(if verbose { "info" } else { "warn" }),
    )
    .init();

    // Inject the logo at runtime so we can use ANSI styling.
    let logo = make_logo();
    let matches = Cli::command()
        .before_help(logo.clone())
        .before_long_help(logo)
        .get_matches();
    let cli = Cli::from_arg_matches(&matches).unwrap_or_else(|e| e.exit());

    match cli.command {
        Commands::Threads { cmd } => handle_threads(cmd).await,
        Commands::Models { cmd } => handle_models(cmd).await,
        Commands::Serve { args } => handle_serve(args).await,
        Commands::Tui { args } => handle_tui(args).await,
        Commands::Launch { program, program_args, model, bin, port, api_key, n_gpu_layers, ctx_size, fit, verbose, select } => {
            let program = program.unwrap_or_else(select_program_interactively);
            let ctx_size_val = ctx_size.unwrap_or(32768);
            handle_launch(program, program_args, model, bin, port, api_key, n_gpu_layers, ctx_size_val, fit, ctx_size.is_none(), verbose, select).await
        }
    }
}


// ── Threads handlers ───────────────────────────────────────────────────────

async fn handle_threads(cmd: ThreadsCommands) {
    match cmd {
        ThreadsCommands::List => match cli_list_threads().await {
            Ok(threads) => {
                println!("{}", serde_json::to_string_pretty(&threads).unwrap());
            }
            Err(e) => {
                eprintln!("Error: {e}");
                std::process::exit(1);
            }
        },

        ThreadsCommands::Get { id } => match cli_get_thread(&id) {
            Ok(thread) => println!("{}", serde_json::to_string_pretty(&thread).unwrap()),
            Err(e) => {
                eprintln!("Error: {e}");
                std::process::exit(1);
            }
        },

        ThreadsCommands::Delete { id } => match cli_delete_thread(&id) {
            Ok(()) => println!("{}", serde_json::json!({ "deleted": true, "id": id })),
            Err(e) => {
                eprintln!("Error: {e}");
                std::process::exit(1);
            }
        },

        ThreadsCommands::Messages { thread_id } => match cli_list_messages(&thread_id) {
            Ok(messages) => println!("{}", serde_json::to_string_pretty(&messages).unwrap()),
            Err(e) => {
                eprintln!("Error: {e}");
                std::process::exit(1);
            }
        },
    }
}

// ── Models handlers ────────────────────────────────────────────────────────

async fn handle_models(cmd: ModelsCommands) {
    match cmd {
        ModelsCommands::List { engine } => {
            let engines: &[&str] = match engine.as_str() {
                "all" => &["llamacpp", "mlx"],
                other => &[other],
            };
            let mut output: Vec<serde_json::Value> = Vec::new();
            for eng in engines {
                for (id, yml) in list_models(eng) {
                    output.push(serde_json::json!({
                        "id": id,
                        "engine": eng,
                        "name": yml.name,
                        "model_path": yml.model_path,
                        "size_bytes": yml.size_bytes,
                        "embedding": yml.embedding,
                        "capabilities": yml.capabilities,
                        "mmproj_path": yml.mmproj_path,
                    }));
                }
            }
            println!("{}", serde_json::to_string_pretty(&output).unwrap());
        }

        ModelsCommands::Load { args } => handle_serve(args).await,

        #[cfg(target_os = "macos")]
        ModelsCommands::LoadMlx {
            model_id,
            model_path,
            bin,
            port,
            ctx_size,
            embedding,
            timeout,
            api_key,
        } => {
            use std::path::Path;

            // Resolve model path from model.yml when not explicitly given
            let resolved_model_path = match model_path {
                Some(p) => p,
                None => match resolve_model_by_id(&model_id, "mlx") {
                    Ok((mp, _)) => mp.to_string_lossy().into_owned(),
                    Err(e) => {
                        eprintln!("Error: {e}");
                        std::process::exit(1);
                    }
                },
            };

            // Resolve binary path: use --bin if provided, otherwise auto-discover
            let bin_path = match bin {
                Some(b) => b,
                None => match discover_mlx_binary() {
                    Some(p) => p.to_string_lossy().into_owned(),
                    None => {
                        eprintln!(
                            "Error: mlx-server binary not found. \
                            Install Jan from https://jan.ai or pass --bin <path>."
                        );
                        std::process::exit(1);
                    }
                },
            };

            let mlx_state = Arc::new(init_mlx_state());
            let mut envs: HashMap<String, String> = HashMap::new();
            if !api_key.is_empty() {
                envs.insert("MLX_API_KEY".to_string(), api_key);
            }

            match load_mlx_model_impl(
                mlx_state.mlx_server_process.clone(),
                Path::new(&bin_path),
                model_id,
                resolved_model_path,
                port,
                MlxConfig { ctx_size },
                envs,
                embedding,
                timeout,
            )
            .await
            {
                Ok(info) => println!("{}", serde_json::to_string_pretty(&info).unwrap()),
                Err(e) => {
                    eprintln!(
                        "Error loading MLX model:\n{}",
                        serde_json::to_string_pretty(&e)
                            .unwrap_or_else(|_| format!("{e:?}"))
                    );
                    std::process::exit(1);
                }
            }
        }
    }
}

// ── Spinner / progress helpers ─────────────────────────────────────────────

fn make_spinner(msg: impl Into<std::borrow::Cow<'static, str>>) -> ProgressBar {
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"])
            .template("{spinner:.cyan} {msg}")
            .unwrap(),
    );
    pb.set_message(msg);
    pb.enable_steady_tick(std::time::Duration::from_millis(80));
    pb
}

/// Start a spinner, or print a plain status line if verbose mode is on.
/// Returns `None` in verbose mode so callers know to skip spinner updates.
fn start_progress(verbose: bool, msg: impl Into<String>) -> Option<ProgressBar> {
    if verbose {
        eprintln!("{}", msg.into());
        None
    } else {
        Some(make_spinner(msg.into()))
    }
}

/// Finish the spinner with a final message, or print the message plainly in verbose mode.
fn finish_progress(pb: Option<ProgressBar>, msg: impl AsRef<str>) {
    match pb {
        Some(pb) => {
            pb.finish_and_clear();
            eprintln!("{}", msg.as_ref());
        }
        None => eprintln!("{}", msg.as_ref()),
    }
}

// ── HuggingFace auto-download ──────────────────────────────────────────────

/// Read `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` from the environment.
fn hf_token() -> Option<String> {
    std::env::var("HF_TOKEN")
        .or_else(|_| std::env::var("HUGGING_FACE_HUB_TOKEN"))
        .ok()
        .filter(|s| !s.is_empty())
}

/// Format a byte count as a human-readable string (GB / MB / KB).
fn fmt_bytes(b: u64) -> String {
    if b >= 1_000_000_000 {
        format!("{:.1} GB", b as f64 / 1_000_000_000.0)
    } else if b >= 1_000_000 {
        format!("{:.0} MB", b as f64 / 1_000_000.0)
    } else {
        format!("{:.0} KB", b as f64 / 1_000.0)
    }
}

/// Show an interactive picker for a list of HF GGUF files and return the chosen one.
///
/// If there is only one file it is returned immediately without prompting.
fn pick_hf_file(files: &[HfFileInfo]) -> &HfFileInfo {
    if files.len() == 1 {
        return &files[0];
    }

    let labels: Vec<String> = files
        .iter()
        .map(|f| format!("{:<55} {}", f.filename, fmt_bytes(f.size)))
        .collect();

    let idx = dialoguer::Select::new()
        .with_prompt("Select a quantization to download")
        .items(&labels)
        .default(0)
        .interact()
        .unwrap_or_else(|_| std::process::exit(1));

    &files[idx]
}

/// Fetch GGUF files from HuggingFace, let the user pick one, download it,
/// and return the local model ID ready to load.
///
/// Exits the process on any unrecoverable error.
async fn auto_download_hf_model(repo_id: &str, select_quantization: bool) -> String {
    let token = hf_token();
    let tok_ref = token.as_deref();

    // Fetch available GGUF files from the HF API
    eprintln!();
    let fetch_pb = make_spinner(format!("Fetching file list for '{repo_id}' from HuggingFace…"));
    let files = fetch_hf_gguf_files(repo_id, tok_ref)
        .await
        .unwrap_or_else(|e| {
            fetch_pb.finish_with_message(format!("✗ {e}"));
            std::process::exit(1);
        });
    fetch_pb.finish_and_clear();

    // Select quantization: show picker if select_quantization is true, otherwise auto-pick Q4_K_XL
    let chosen = if select_quantization {
        pick_hf_file(&files)
    } else {
        files
            .iter()
            .find(|f| f.filename.contains("Q4_K_XL"))
            .unwrap_or_else(|| {
                files.iter().max_by_key(|f| f.size).unwrap()
            })
    };
    eprintln!("  Downloading  {}", chosen.filename);
    eprintln!("  Size         {}", fmt_bytes(chosen.size));
    eprintln!();

    // Progress bar — byte-count style
    let dl_pb = ProgressBar::new(chosen.size);
    dl_pb.set_style(
        ProgressStyle::default_bar()
            .template(
                "  {bar:45.yellow/dim}  {bytes:>9}/{total_bytes}  {bytes_per_sec}  eta {eta}",
            )
            .unwrap()
            .progress_chars("█▉▊▋▌▍▎▏  "),
    );

    let dl_pb_clone = dl_pb.clone();
    let model_id = download_hf_model(repo_id, chosen, tok_ref, move |done, _total| {
        dl_pb_clone.set_position(done);
    })
    .await
    .unwrap_or_else(|e| {
        dl_pb.finish_with_message(format!("✗ Download failed: {e}"));
        std::process::exit(1);
    });

    dl_pb.finish_and_clear();
    eprintln!("  ✓ Saved to Jan data folder\n");

    model_id
}

// ── Interactive pickers ────────────────────────────────────────────────────

/// Present an interactive menu for the supported AI agents.
fn select_program_interactively() -> String {
    const CHOICES: &[(&str, &str)] = &[
        ("claude",   "Claude Code  — Anthropic's AI coding agent"),
        ("openclaw", "OpenClaw     — Open-source autonomous AI agent"),
    ];

    println!();
    let header = Style::new().cyan().bold().apply_to("━━━ Select Agent ━━━");
    println!("{}", header);
    println!();

    let installed: Vec<bool> = CHOICES
        .iter()
        .map(|(key, _)| {
            is_command_installed(key) || (*key == "openclaw" && is_command_installed("opencode"))
        })
        .collect();

    if !installed.iter().any(|&i| i) {
        eprintln!("  No supported agents are installed.");
        eprintln!("  Install Claude Code: npm install -g @anthropic-ai/claude-code");
        eprintln!("  Install OpenClaw:    curl -fsSL https://openclaw.ai/install.sh | bash -s -- --no-onboard");
        std::process::exit(1);
    }

    let labels: Vec<String> = CHOICES
        .iter()
        .zip(installed.iter())
        .map(|((_, desc), &ok)| {
            if ok {
                desc.to_string()
            } else {
                format!("{} {}", Style::new().dim().apply_to(desc), Style::new().dim().apply_to("[not installed]"))
            }
        })
        .collect();

    // Keep re-prompting until the user picks an installed agent
    loop {
        let idx = dialoguer::Select::new()
            .with_prompt("Choose an agent to launch")
            .items(&labels)
            .default(0)
            .interact()
            .unwrap_or_else(|_| std::process::exit(1));

        if installed[idx] {
            return CHOICES[idx].0.to_string();
        }

        eprintln!("  {} is not installed. Please choose an installed agent.", CHOICES[idx].1);
    }
}

async fn select_model_interactively(select_quantization: bool) -> String {
    let mut all: Vec<(String, String)> = Vec::new(); // (id, engine)
    for engine in &["llamacpp", "mlx"] {
        for (id, _) in list_models(engine) {
            all.push((id, engine.to_string()));
        }
    }

    if all.is_empty() {
        let default_model = "janhq/Jan-v3-4B-base-instruct-gguf";
        println!();
        let msg = Style::new().yellow().apply_to(
            "No models found. Downloading default model..."
        );
        println!("{}", msg);
        println!();
        println!("  {} {}", Style::new().dim().apply_to("Model:"), default_model);
        println!();

        // Auto-download the default model
        let model_id = auto_download_hf_model(default_model, select_quantization).await;
        return model_id;
    }

    println!();
    let header = Style::new().cyan().bold().apply_to("━━━ Select Model ━━━");
    println!("{}", header);
    println!();

    // Group by engine for better display
    let mut llamacpp_models: Vec<&String> = Vec::new();
    let mut mlx_models: Vec<&String> = Vec::new();

    for (id, engine) in &all {
        match engine.as_str() {
            "llamacpp" => llamacpp_models.push(id),
            "mlx" => mlx_models.push(id),
            _ => {}
        }
    }

    // Build selection list with engine indicator next to model name
    let selection_items: Vec<(usize, String)> = all
        .iter()
        .enumerate()
        .map(|(i, (id, engine))| {
            let indicator = match engine.as_str() {
                "llamacpp" => Style::new().green().apply_to("[LlamaCPP]"),
                "mlx" => Style::new().magenta().apply_to("[MLX]"),
                _ => Style::new().dim().apply_to("[---]"),
            };
            (i, format!("{} {}", id, indicator))
        })
        .collect();

    // If only one model, skip interactive selection
    if selection_items.len() == 1 {
        println!("  Using model: {}", selection_items[0].1);
        println!();
        return all[selection_items[0].0].0.clone();
    }

    let labels: Vec<String> = selection_items.iter().map(|(_, label)| label.clone()).collect();

    let selection = dialoguer::Select::new()
        .with_prompt("Choose a model")
        .items(&labels)
        .default(0)
        .interact()
        .unwrap_or_else(|_| std::process::exit(1));

    all[selection_items[selection].0].0.clone()
}

// ── Detached spawn ─────────────────────────────────────────────────────────

fn spawn_detached(model_id: &str, args: &ServeArgs) {
    let exe = std::env::current_exe().expect("cannot resolve current exe");

    // Rebuild argv from ServeArgs fields so we have full control
    // (avoids needing to filter --detach/-d from the raw OS args).
    // Use --flag=value format throughout to avoid negative numbers being
    // misinterpreted as short flags (e.g. --n-gpu-layers -1 → -1 looks like a flag).
    let mut argv: Vec<String> = vec!["serve".into(), model_id.to_string()];
    if let Some(p) = &args.model_path { argv.push(format!("--model-path={p}")); }
    if let Some(b) = &args.bin        { argv.push(format!("--bin={b}")); }
    argv.push(format!("--port={}", args.port));
    if let Some(m) = &args.mmproj     { argv.push(format!("--mmproj={m}")); }
    if args.embedding                  { argv.push("--embedding".into()); }
    argv.push(format!("--timeout={}",      args.timeout));
    argv.push(format!("--n-gpu-layers={}", args.n_gpu_layers));
    argv.push(format!("--ctx-size={}",     args.ctx_size));
    argv.push(format!("--threads={}",      args.threads));
    if !args.api_key.is_empty()        { argv.push(format!("--api-key={}", args.api_key)); }
    if args.fit                        { argv.push("--fit".into()); }
    if args.verbose                    { argv.push("--verbose".into()); }

    // Resolve log file path
    let log_path: PathBuf = args.log.as_deref()
        .map(PathBuf::from)
        .unwrap_or_else(|| cli_get_data_folder().join("logs").join("serve.log"));

    if let Some(parent) = log_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }

    let log_file = std::fs::OpenOptions::new()
        .create(true).append(true).open(&log_path)
        .unwrap_or_else(|e| { eprintln!("Cannot open log file {}: {e}", log_path.display()); std::process::exit(1); });
    let log_out = log_file.try_clone().expect("clone log file");

    let mut cmd = std::process::Command::new(&exe);
    cmd.args(&argv)
        .stdin(std::process::Stdio::null())
        .stdout(log_out)
        .stderr(log_file);

    // Detach from the current terminal session on Unix
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        unsafe {
            cmd.pre_exec(|| {
                nix::unistd::setsid()
                    .map(|_| ())
                    .map_err(|e| std::io::Error::other(e.to_string()))
            });
        }
    }

    // child is intentionally detached via setsid; reaping is the OS's job
    #[allow(clippy::zombie_processes)]
    let child = cmd.spawn().unwrap_or_else(|e| {
        eprintln!("Failed to spawn detached process: {e}");
        std::process::exit(1);
    });

    println!("{}", serde_json::to_string_pretty(&serde_json::json!({
        "pid":      child.id(),
        "model_id": model_id,
        "log":      log_path.display().to_string(),
    })).unwrap());
}

// ── Serve handler (shared by `models load` and top-level `serve`) ──────────

async fn handle_serve(args: ServeArgs) {
    // Resolve model_id:
    // 1. Use the explicit model_id if it is non-empty.
    // 2. When --model-path is given, derive the id from the filename stem (e.g.
    //    "/path/to/my-model.gguf" → "my-model") so the user never has to pass
    //    a dummy empty-string id.
    // 3. Fall back to the interactive picker only when neither is available.
    let model_id = match args.model_id.as_deref() {
        Some(id) if !id.is_empty() => id.to_string(),
        _ => {
            if let Some(ref path) = args.model_path {
                PathBuf::from(path)
                    .file_stem()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_else(|| "model".to_string())
            } else {
                select_model_interactively(args.select).await
            }
        }
    };

    if args.detach {
        spawn_detached(&model_id, &args);
        return;
    }

    let ServeArgs {
        model_id: _,
        model_path,
        bin,
        port,
        mmproj,
        embedding,
        timeout,
        n_gpu_layers,
        ctx_size: ctx_size_arg,
        threads,
        api_key,
        fit,
        detach: _,
        log: _,
        verbose,
        select: _,
    } = args;

    // When --fit is on, let llama.cpp decide the context size automatically
    let ctx_size = if fit { 0 } else { ctx_size_arg };

    // Auto-detect engine from data folder.
    // If the model isn't found locally and the ID looks like a HuggingFace repo,
    // offer to download it automatically before proceeding.
    let (engine, resolved_model_path, resolved_mmproj) =
        match resolve_model_engine(&model_id) {
            Ok((eng, mp, mmp)) => (
                eng,
                model_path.unwrap_or_else(|| mp.to_string_lossy().into_owned()),
                mmproj.or_else(|| mmp.map(|p| p.to_string_lossy().into_owned())),
            ),
            Err(_) if model_path.is_some() => {
                // Explicit --model-path provided: skip engine detection entirely.
                ("llamacpp".to_string(), model_path.unwrap(), mmproj)
            }
            Err(_) if looks_like_hf_repo(&model_id) => {
                // Looks like a HuggingFace repo ID — download then resolve.
                auto_download_hf_model(&model_id, args.select).await;
                match resolve_model_engine(&model_id) {
                    Ok((eng, mp, mmp)) => (
                        eng,
                        mp.to_string_lossy().into_owned(),
                        mmp.map(|p| p.to_string_lossy().into_owned()),
                    ),
                    Err(e) => {
                        eprintln!("Error after download: {e}");
                        std::process::exit(1);
                    }
                }
            }
            Err(e) => {
                eprintln!("Error: {e}");
                std::process::exit(1);
            }
        };

    let pb = start_progress(verbose, format!("Loading {} ({engine})…", model_id));

    if engine == "mlx" {
        #[cfg(not(target_os = "macos"))]
        {
            finish_progress(pb, "✗ MLX is only supported on macOS");
            eprintln!("MLX models require macOS / Apple Silicon. Use a llama.cpp (GGUF) model instead.");
            std::process::exit(1);
        }
        #[cfg(target_os = "macos")]
        {
        use std::path::Path;

        let bin_path = match bin {
            Some(b) => b,
            None => match discover_mlx_binary() {
                Some(p) => p.to_string_lossy().into_owned(),
                None => {
                    finish_progress(pb, "✗ mlx-server binary not found");
                    eprintln!("Install Jan from https://jan.ai or pass --bin <path>.");
                    std::process::exit(1);
                }
            },
        };

        let mlx_state = Arc::new(init_mlx_state());
        let mut envs: HashMap<String, String> = HashMap::new();
        if !api_key.is_empty() {
            envs.insert("MLX_API_KEY".to_string(), api_key);
        }

        match load_mlx_model_impl(
            mlx_state.mlx_server_process.clone(),
            Path::new(&bin_path),
            model_id.clone(),
            resolved_model_path,
            port,
            MlxConfig { ctx_size },
            envs,
            embedding,
            timeout,
        )
        .await
        {
            Ok(info) => {
                let url = format!("http://127.0.0.1:{}", info.port);
                finish_progress(pb, format!("✓ {model_id} ready · {url}"));
                eprintln!();
                eprintln!("  Endpoint  {url}/v1");
                eprintln!();
                eprintln!("  Press Ctrl+C to stop.");
                wait_for_shutdown(info.pid).await;
            }
            Err(e) => {
                finish_progress(pb, format!("✗ Failed to load {model_id}"));
                eprintln!(
                    "\n{}",
                    serde_json::to_string_pretty(&e).unwrap_or_else(|_| format!("{e:?}"))
                );
                std::process::exit(1);
            }
        }
        }
    } else {
        // LlamaCPP path
        let bin_path = match bin {
            Some(b) => b,
            None => match discover_llamacpp_binary() {
                Some(p) => p.to_string_lossy().into_owned(),
                None => {
                    finish_progress(pb, "✗ llama-server binary not found");
                    eprintln!("Install a backend from Jan's settings or pass --bin <path>.");
                    std::process::exit(1);
                }
            },
        };

        let _ = (n_gpu_layers, ctx_size, fit, threads, resolved_model_path, resolved_mmproj);
        let llama_state = Arc::new(init_llamacpp_state());
        let mut envs: HashMap<String, String> = HashMap::new();
        if !api_key.is_empty() {
            envs.insert("LLAMA_API_KEY".to_string(), api_key.clone());
        }

        match ensure_router_and_load(
            &llama_state,
            &bin_path,
            &model_id,
            port,
            api_key,
            embedding,
            envs,
            timeout,
        )
        .await
        {
            Ok(info) => {
                let url = format!("http://127.0.0.1:{}", info.port);
                finish_progress(pb, format!("✓ {model_id} ready · {url}"));
                eprintln!();
                eprintln!("  Endpoint  {url}/v1");
                eprintln!();
                eprintln!("  Press Ctrl+C to stop.");
                wait_for_shutdown(info.pid).await;
            }
            Err(e) => {
                finish_progress(pb, format!("✗ Failed to load {model_id}: {e}"));
                std::process::exit(1);
            }
        }
    }
}

struct RouterServeInfo {
    pid: i32,
    port: u16,
    #[allow(dead_code)]
    api_key: String,
}

#[allow(clippy::too_many_arguments)]
async fn ensure_router_and_load(
    llama_state: &std::sync::Arc<LlamacppState>,
    bin_path: &str,
    model_id: &str,
    port: u16,
    api_key: String,
    is_embedding: bool,
    envs: HashMap<String, String>,
    timeout: u64,
) -> Result<RouterServeInfo, String> {
    if is_embedding {
        return Err(
            "--embedding on the llamacpp engine requires router preset support; \
             use the desktop UI to load embedding models for now."
                .to_string(),
        );
    }

    let preset_path = cli_get_data_folder()
        .join("llamacpp")
        .join("router.preset.ini");
    if !preset_path.exists() {
        return Err(format!(
            "Router preset not found at {}; run the desktop app once to generate it, \
             or implement a Rust preset generator.",
            preset_path.display()
        ));
    }

    let already_running = { llama_state.router.lock().await.is_some() };
    if !already_running {
        let router_api_key = api_key.clone();
        let mut router_envs = envs.clone();
        router_envs
            .entry("LLAMA_ARG_TIMEOUT".to_string())
            .or_insert_with(|| timeout.to_string());

        let handle = llamacpp_router::start_router(
            std::path::PathBuf::from(bin_path),
            preset_path,
            port,
            router_api_key,
            0,
            Vec::new(),
            router_envs,
            None,
        )
        .await
        .map_err(|e| format!("{e:?}"))?;
        let mut guard = llama_state.router.lock().await;
        *guard = Some(handle);
    }

    let (router_port, router_key, router_pid) = {
        let guard = llama_state.router.lock().await;
        let h = guard
            .as_ref()
            .ok_or_else(|| "Router unexpectedly missing after start".to_string())?;
        (h.port, h.api_key.clone(), h.pid)
    };

    #[cfg(windows)]
    win_job::assign_process_to_kill_on_close_job(router_pid);

    let url = format!("http://127.0.0.1:{router_port}/models/load");
    let mut req = reqwest::Client::new().post(&url);
    if !router_key.is_empty() {
        req = req.header("Authorization", format!("Bearer {router_key}"));
    }
    let resp = req
        .json(&serde_json::json!({ "model": model_id }))
        .send()
        .await
        .map_err(|e| format!("Failed to POST {url}: {e}"))?;
    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        return Err(format!("Router /models/load returned {status}: {body}"));
    }

    Ok(RouterServeInfo {
        pid: router_pid as i32,
        port: router_port,
        api_key: router_key,
    })
}

/// Block until Ctrl+C, then terminate the child process.
async fn wait_for_shutdown(pid: i32) {
    tokio::signal::ctrl_c().await.ok();
    eprintln!("\nShutting down (pid {pid})...");
    kill_process(pid);
}

/// Send a termination signal to a child process by PID.
fn kill_process(pid: i32) {
    #[cfg(unix)]
    {
        use nix::sys::signal::{kill, Signal};
        use nix::unistd::Pid;
        let _ = kill(Pid::from_raw(pid), Signal::SIGTERM);
    }
    #[cfg(windows)]
    {
        let _ = std::process::Command::new("taskkill")
            .args(["/PID", &pid.to_string(), "/F"])
            .status();
    }
}

// ── Agent installer ─────────────────────────────────────────────────────────

/// Check if a command is available in PATH
fn is_command_installed(cmd: &str) -> bool {
    let which = if cfg!(windows) { "where" } else { "which" };
    std::process::Command::new(which)
        .arg(cmd)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

// ── Launch handler ─────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
async fn handle_launch(
    program: String,
    program_args: Vec<String>,
    model: Option<String>,
    bin: Option<String>,
    port: u16,
    api_key: String,
    n_gpu_layers: i32,
    ctx_size: i32,
    fit: Option<bool>,
    ctx_size_is_default: bool,
    verbose: bool,
    select: bool,
) {
    let model_id = model.unwrap_or_else(|| -> String {
        futures::executor::block_on(select_model_interactively(select))
    });

    // Detect known agents early so we can set fit default before starting the server.
    let prog_name = std::path::Path::new(&program)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(&program);
    let is_claude   = prog_name.contains("claude");
    let is_openclaw = prog_name.contains("openclaw");

    // --fit defaults to true when launching claude, but only if --ctx-size was
    // not explicitly provided (an explicit ctx-size means the user wants that
    // exact context, so fit would override it — don't do that).
    let effective_fit = fit.unwrap_or(is_claude && ctx_size_is_default);

    // Start the model server inline (same process, no detach).
    let (pid, actual_port, actual_api_key, _server_state) = start_model_server(&model_id, bin, port, api_key.clone(), n_gpu_layers, ctx_size, effective_fit, verbose).await;

    // Model is ready — silence server request/response logs so they don't
    // flood the launched program's terminal (e.g. Claude Code's shell).
    if verbose {
        log::set_max_level(log::LevelFilter::Warn);
    }

    let base_url = format!("http://127.0.0.1:{actual_port}");
    let v1_url   = format!("{base_url}/v1");

    // openclaw is configured via ~/.openclaw/openclaw.json, not env vars.
    // Write the jan provider entry and set the default model, then launch `openclaw tui`.
    let mut program_args = program_args;
    if is_openclaw {
        configure_openclaw(&v1_url, &actual_api_key, &model_id);
        // openclaw's TUI is a sub-command; prepend it unless the caller already did
        if program_args.first().map(|s| s.as_str()) != Some("tui") {
            program_args.insert(0, "tui".to_string());
        }
        eprintln!();
        eprintln!("  ~/.openclaw/openclaw.json → jan provider configured");
        eprintln!("  agents.defaults.model.primary = jan/{model_id}");
    } else {
        let anthropic_key_var = if is_claude { "ANTHROPIC_AUTH_TOKEN" } else { "ANTHROPIC_API_KEY" };
        eprintln!();
        eprintln!("  OPENAI_BASE_URL={v1_url}");
        eprintln!("  OPENAI_API_KEY={actual_api_key}");
        eprintln!("  OPENAI_MODEL={model_id}");
        eprintln!("  ANTHROPIC_BASE_URL={base_url}");
        eprintln!("  {anthropic_key_var}={actual_api_key}");
        eprintln!("  ANTHROPIC_DEFAULT_OPUS_MODEL={model_id}");
        eprintln!("  ANTHROPIC_DEFAULT_SONNET_MODEL={model_id}");
        eprintln!("  ANTHROPIC_DEFAULT_HAIKU_MODEL={model_id}");
    }
    eprintln!();
    let launch_cmd = if is_openclaw {
        format!("openclaw {}", program_args.join(" "))
    } else {
        format!("{} {}", program, program_args.join(" "))
    };
    eprintln!("  → Launching: {}", launch_cmd);
    eprintln!();

    // For openclaw, use npx if not installed locally
    let (cmd_program, cmd_args) = if is_openclaw {
        if is_command_installed("openclaw") {
            (program.clone(), program_args.clone())
        } else {
            let mut args = vec!["openclaw".to_string()];
            args.extend(program_args.clone());
            ("npx".to_string(), args)
        }
    } else {
        (program.clone(), program_args.clone())
    };

    let mut cmd = std::process::Command::new(&cmd_program);
    cmd.args(&cmd_args);
    if is_openclaw {
        // Clear any provider API keys that could override openclaw's config
        for var in &[
            "OPENAI_API_KEY", "OPENAI_BASE_URL",
            "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_OAUTH_TOKEN",
            "GEMINI_API_KEY", "MISTRAL_API_KEY", "GROQ_API_KEY",
            "XAI_API_KEY", "OPENROUTER_API_KEY",
        ] {
            cmd.env_remove(var);
        }
    } else {
        let anthropic_key_var = if is_claude { "ANTHROPIC_AUTH_TOKEN" } else { "ANTHROPIC_API_KEY" };
        cmd.env("OPENAI_BASE_URL", &v1_url)
            .env("OPENAI_API_KEY",  &actual_api_key)
            .env("OPENAI_MODEL",    &model_id)
            .env("ANTHROPIC_BASE_URL", &base_url)
            .env(anthropic_key_var,    &actual_api_key)
            .env("ANTHROPIC_DEFAULT_OPUS_MODEL",   &model_id)
            .env("ANTHROPIC_DEFAULT_SONNET_MODEL", &model_id)
            .env("ANTHROPIC_DEFAULT_HAIKU_MODEL",  &model_id);
    }
    let status = cmd.status();

    // Kill the model server when the program exits.
    kill_process(pid);

    match status {
        Ok(s) => std::process::exit(s.code().unwrap_or(0)),
        Err(e) => {
            eprintln!("Error launching '{program}': {e}");
            std::process::exit(1);
        }
    }
}

// ── openclaw config writer ─────────────────────────────────────────────────

/// Write (or merge into) `~/.openclaw/openclaw.json` so that openclaw uses
/// the local Jan server as its provider and selects `model_id` by default.
///
/// The "jan" provider entry is always overwritten with the current server
/// address and key. All other config values are preserved.
/// Also clears the session model override so the new default takes effect.
fn configure_openclaw(v1_url: &str, api_key: &str, model_id: &str) {
    let home = dirs::home_dir().unwrap_or_default();
    let config_path = home.join(".openclaw").join("openclaw.json");

    // Read existing config so we don't clobber other settings.
    let mut config: serde_json::Value = config_path
        .exists()
        .then(|| std::fs::read_to_string(&config_path).ok())
        .flatten()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or(serde_json::json!({}));

    // Inject (or overwrite) the jan provider.
    config["models"]["providers"]["jan"] = serde_json::json!({
        "baseUrl": v1_url,
        "apiKey":  api_key,
        "api":     "openai-completions",
        "models": [{
            "id":            model_id,
            "name":          model_id,
            "input":         ["text"],
            "reasoning":     false,
            "contextWindow": 131072,
            "maxTokens":     16384,
            "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 }
        }]
    });

    // Set jan/<model_id> as the primary default model.
    config["agents"]["defaults"]["model"]["primary"] =
        serde_json::json!(format!("jan/{model_id}"));

    if let Some(parent) = config_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Ok(json) = serde_json::to_string_pretty(&config) {
        let _ = std::fs::write(&config_path, json);
    }

    // Clear any per-session model override so the new default is picked up.
    let sessions_path = home
        .join(".openclaw")
        .join("agents")
        .join("main")
        .join("sessions")
        .join("sessions.json");
    if sessions_path.exists() {
        let _ = std::fs::write(&sessions_path, "{}");
    }
}

#[allow(dead_code)]
enum ActiveServerState {
    Llama(Arc<LlamacppState>),
    #[cfg(target_os = "macos")]
    Mlx(Arc<tauri_plugin_mlx::state::MlxState>),
}

/// Start the model server and return `(pid, actual_port, api_key, state)`.
/// Resolves the engine automatically (LlamaCPP or MLX).
#[allow(clippy::too_many_arguments)]
async fn start_model_server(
    model_id: &str,
    bin: Option<String>,
    port: u16,
    api_key: String,
    n_gpu_layers: i32,
    ctx_size: i32,
    fit: bool,
    verbose: bool,
) -> (i32, u16, String, ActiveServerState) {
    let (engine, model_path, mmproj) = match resolve_model_engine(model_id) {
        Ok(r) => r,
        Err(_) if looks_like_hf_repo(model_id) => {
            auto_download_hf_model(model_id, false).await;
            match resolve_model_engine(model_id) {
                Ok(r) => r,
                Err(e) => { eprintln!("Error after download: {e}"); std::process::exit(1); }
            }
        }
        Err(e) => { eprintln!("Error: {e}"); std::process::exit(1); }
    };
    let model_path = model_path.to_string_lossy().into_owned();

    let pb = start_progress(verbose, format!("Loading {} ({engine})…", model_id));

    if engine == "mlx" {
        #[cfg(not(target_os = "macos"))]
        {
            finish_progress(pb, "✗ MLX is only supported on macOS");
            eprintln!("MLX models require macOS / Apple Silicon. Use a llama.cpp (GGUF) model instead.");
            std::process::exit(1)
        }
        #[cfg(target_os = "macos")]
        {
        use std::path::Path;
        let bin_path = match bin.or_else(|| discover_mlx_binary().map(|p| p.to_string_lossy().into_owned())) {
            Some(p) => p,
            None => {
                finish_progress(pb, "✗ mlx-server binary not found");
                eprintln!("Install Jan from https://jan.ai or pass --bin <path>.");
                std::process::exit(1);
            }
        };
        let mlx_state = Arc::new(init_mlx_state());
        let mut envs: HashMap<String, String> = HashMap::new();
        if !api_key.is_empty() { envs.insert("MLX_API_KEY".to_string(), api_key.clone()); }
        let effective_ctx_size = if fit { 0 } else { ctx_size };
        let info = match load_mlx_model_impl(
            mlx_state.mlx_server_process.clone(),
            Path::new(&bin_path),
            model_id.to_string(),
            model_path,
            port,
            MlxConfig { ctx_size: effective_ctx_size },
            envs,
            false,
            120,
        ).await {
            Ok(info) => info,
            Err(e) => {
                finish_progress(pb, format!("✗ Failed to load {model_id}"));
                eprintln!("{}", serde_json::to_string_pretty(&e).unwrap_or_else(|_| format!("{e:?}")));
                std::process::exit(1);
            }
        };
        let url = format!("http://127.0.0.1:{}", info.port);
        finish_progress(pb, format!("✓ {model_id} ready · {url}"));
        (info.pid, info.port as u16, api_key.clone(), ActiveServerState::Mlx(mlx_state))
        }
    } else {
        let bin_path = match bin.or_else(|| discover_llamacpp_binary().map(|p| p.to_string_lossy().into_owned())) {
            Some(p) => p,
            None => {
                finish_progress(pb, "✗ llama-server binary not found");
                eprintln!("Install a backend from Jan's settings or pass --bin <path>.");
                std::process::exit(1);
            }
        };
        let _ = (n_gpu_layers, ctx_size, fit, model_path, mmproj);
        let llama_state = Arc::new(init_llamacpp_state());
        let mut envs: HashMap<String, String> = HashMap::new();
        if !api_key.is_empty() { envs.insert("LLAMA_API_KEY".to_string(), api_key.clone()); }
        let info = match ensure_router_and_load(
            &llama_state,
            &bin_path,
            model_id,
            port,
            api_key,
            false,
            envs,
            120,
        ).await {
            Ok(info) => info,
            Err(e) => {
                finish_progress(pb, format!("✗ Failed to load {model_id}: {e}"));
                std::process::exit(1);
            }
        };
        let url = format!("http://127.0.0.1:{}", info.port);
        finish_progress(pb, format!("✓ {model_id} ready · {url}"));
        (info.pid, info.port, info.api_key, ActiveServerState::Llama(llama_state))
    }
}

// ── TUI Chat Implementation ──────────────────────────────────────────────────

struct TuiRenderer {
    in_code_block: bool,
    in_inline_code: bool,
    in_bold: bool,
    is_line_start: bool,
    header_level: usize,
    buffering_lang: bool,
    lang_buffer: String,
    buffer: Vec<char>,
    current_code_block: String,
    in_thinking: bool,
}

impl TuiRenderer {
    fn new() -> Self {
        Self {
            in_code_block: false,
            in_inline_code: false,
            in_bold: false,
            is_line_start: true,
            header_level: 0,
            buffering_lang: false,
            lang_buffer: String::new(),
            buffer: Vec::new(),
            current_code_block: String::new(),
            in_thinking: false,
        }
    }

    fn render_chunk(&mut self, chunk: &str) {
        self.buffer.extend(chunk.chars());
        self.process_buffer(false);
    }

    fn process_buffer(&mut self, is_final: bool) {
        use console::Style;
        let mut i = 0;
        while i < self.buffer.len() {
            // Check for opening <think> tag when not thinking
            if !self.in_thinking && self.buffer[i] == '<' {
                let think_tag = vec!['<', 't', 'h', 'i', 'n', 'k', '>'];
                if i + think_tag.len() <= self.buffer.len() {
                    if self.buffer[i..i + think_tag.len()] == think_tag {
                        self.in_thinking = true;
                        i += think_tag.len();
                        continue;
                    }
                } else {
                    let len = self.buffer.len() - i;
                    if self.buffer[i..] == think_tag[..len] && !is_final {
                        break;
                    }
                }
            }

            if self.in_thinking {
                // Check for closing </think> tag
                if self.buffer[i] == '<' {
                    let close_think_tag = vec!['<', '/', 't', 'h', 'i', 'n', 'k', '>'];
                    if i + close_think_tag.len() <= self.buffer.len() {
                        if self.buffer[i..i + close_think_tag.len()] == close_think_tag {
                            self.in_thinking = false;
                            i += close_think_tag.len();
                            continue;
                        }
                    } else {
                        let len = self.buffer.len() - i;
                        if self.buffer[i..] == close_think_tag[..len] && !is_final {
                            break;
                        }
                    }
                }

                let c = self.buffer[i];
                i += 1;
                self.is_line_start = c == '\n';
                print!("{}", Style::new().dim().italic().apply_to(c));
                std::io::stdout().flush().ok();
                continue;
            }

            // Check if we have a backtick
            if self.buffer[i] == '`' {
                // Lookahead check for code block ```
                if i + 2 < self.buffer.len() {
                    if self.buffer[i+1] == '`' && self.buffer[i+2] == '`' {
                        self.in_code_block = !self.in_code_block;
                        i += 3;
                        
                        // Clear inline styles
                        self.in_inline_code = false;
                        self.in_bold = false;
                        
                        if self.in_code_block {
                            self.buffering_lang = true;
                            self.lang_buffer.clear();
                        } else {
                            let index = {
                                let mut idx = 1;
                                if let Ok(guard) = get_tui_code_blocks().lock() {
                                    idx = guard.len() + 1;
                                }
                                idx
                            };
                            
                            let pad_len = 50;
                            let border = format!("{}[c {}]", "-".repeat(pad_len), index);
                            print!("\n{}\n", Style::new().cyan().bold().apply_to(border));
                            
                            if let Ok(mut guard) = get_tui_code_blocks().lock() {
                                guard.push(self.current_code_block.clone());
                            }
                            
                            self.is_line_start = true;
                            std::io::stdout().flush().ok();
                        }
                        continue;
                    }
                } else if !is_final {
                    // Not enough characters to determine if it is ``` or `
                    // Wait for next chunk
                    break;
                }
                
                // Lookahead check for single backtick (inline code)
                if i + 1 < self.buffer.len() {
                    if self.buffer[i+1] != '`' {
                        self.in_inline_code = !self.in_inline_code;
                        i += 1;
                        continue;
                    }
                } else if is_final {
                    // Only 1 backtick left at the end of stream, must be inline code toggle
                    self.in_inline_code = !self.in_inline_code;
                    i += 1;
                    continue;
                } else {
                    break;
                }
            }
            
            // Check if we have an asterisk
            if self.buffer[i] == '*' {
                // Lookahead check for bold **
                if i + 1 < self.buffer.len() {
                    if self.buffer[i+1] == '*' {
                        self.in_bold = !self.in_bold;
                        i += 2;
                        continue;
                    }
                } else if !is_final {
                    // Not enough characters to determine if it is * or **
                    break;
                }
            }

            if self.in_code_block {
                let c = self.buffer[i];
                i += 1;
                
                if self.buffering_lang {
                    if c == '\n' {
                        self.buffering_lang = false;
                        let lang = self.lang_buffer.trim();
                        let index = {
                            let mut idx = 1;
                            if let Ok(guard) = get_tui_code_blocks().lock() {
                                idx = guard.len() + 1;
                            }
                            idx
                        };
                        
                        let lang_part = if lang.is_empty() {
                            format!("------[code ({index})]")
                        } else {
                            format!("------[code - {} ({index})]", lang)
                        };
                        
                        let pad_len = 50usize.saturating_sub(lang_part.len());
                        let border = format!("{}{}[c {}]", lang_part, "-".repeat(pad_len), index);
                        
                        print!("\n{}\n", Style::new().cyan().bold().apply_to(border));
                        self.current_code_block.clear();
                    } else {
                        self.lang_buffer.push(c);
                    }
                    std::io::stdout().flush().ok();
                    continue;
                }

                // Render character inside code block directly in cyan
                print!("{}", Style::new().cyan().apply_to(c));
                self.current_code_block.push(c);
                std::io::stdout().flush().ok();
                continue;
            }

            let c = self.buffer[i];
            
            if c == '\n' {
                i += 1;
                self.is_line_start = true;
                self.header_level = 0;
                print!("\n");
                std::io::stdout().flush().ok();
                continue;
            }

            if self.is_line_start {
                if c == ' ' || c == '\t' {
                    i += 1;
                    print!("{}", c);
                    std::io::stdout().flush().ok();
                    continue;
                }

                // Detect lists: - or * followed by a space
                if c == '-' || c == '*' {
                    if i + 1 < self.buffer.len() {
                        if self.buffer[i+1] == ' ' {
                            i += 2; // consume delimiter and space
                            print!(" {} ", Style::new().magenta().apply_to("•"));
                            self.is_line_start = false;
                            std::io::stdout().flush().ok();
                            continue;
                        }
                    } else if !is_final {
                        // End of buffer, wait
                        break;
                    }
                }

                // Detect thematic break ---
                if c == '-' {
                    let mut dash_count = 0;
                    while i + dash_count < self.buffer.len() && self.buffer[i + dash_count] == '-' {
                        dash_count += 1;
                    }
                    if dash_count >= 3 {
                        if i + dash_count < self.buffer.len() {
                            let next_char = self.buffer[i + dash_count];
                            if next_char == '\n' || next_char == ' ' {
                                i += dash_count;
                                if next_char == ' ' {
                                    i += 1; // consume space
                                }
                                print!("{}", Style::new().cyan().bold().apply_to("━".repeat(60)));
                                self.is_line_start = false;
                                std::io::stdout().flush().ok();
                                continue;
                            }
                        } else if is_final {
                            i += dash_count;
                            print!("{}", Style::new().cyan().bold().apply_to("━".repeat(60)));
                            self.is_line_start = false;
                            std::io::stdout().flush().ok();
                            continue;
                        } else {
                            // End of buffer, wait
                            break;
                        }
                    } else {
                        if i + dash_count >= self.buffer.len() {
                            if !is_final {
                                // Could become 3 dashes in next chunk, wait
                                break;
                            }
                        }
                    }
                }

                // Detect headers: #, ##, etc.
                if c == '#' {
                    let mut hash_count = 0;
                    while i + hash_count < self.buffer.len() && self.buffer[i + hash_count] == '#' {
                        hash_count += 1;
                    }
                    if i + hash_count < self.buffer.len() {
                        if self.buffer[i + hash_count] == ' ' {
                            self.header_level = hash_count;
                            i += hash_count + 1; // consume hashes and space
                            self.is_line_start = false;
                            continue;
                        }
                    } else if !is_final {
                        // End of buffer, wait
                        break;
                    }
                }
            }

            // If we got here, we consume c as a normal styled character
            i += 1;
            self.is_line_start = false;

            // Apply styles to the character
            let mut style = Style::new();
            if self.header_level > 0 {
                style = match self.header_level {
                    1 => Style::new().magenta().bold().underlined(),
                    2 => Style::new().blue().bold(),
                    _ => Style::new().yellow().bold(),
                };
            } else {
                if self.in_bold {
                    style = style.bold().yellow();
                }
                if self.in_inline_code {
                    style = style.green();
                }
            }
            print!("{}", style.apply_to(c));
            std::io::stdout().flush().ok();
        }

        // Drain processed characters
        self.buffer.drain(..i);
    }

    fn finish(&mut self) {
        use console::Style;
        self.process_buffer(true);
        if self.in_code_block {
            let index = {
                let mut idx = 1;
                if let Ok(guard) = get_tui_code_blocks().lock() {
                    idx = guard.len() + 1;
                }
                idx
            };
            
            let pad_len = 50;
            let border = format!("{}[c {}]", "-".repeat(pad_len), index);
            print!("\n{}\n", Style::new().cyan().bold().apply_to(border));
            
            if let Ok(mut guard) = get_tui_code_blocks().lock() {
                guard.push(self.current_code_block.clone());
            }
            
            self.in_code_block = false;
        }
        // Flush remaining buffer character by character (should be empty but just in case)
        while !self.buffer.is_empty() {
            let c = self.buffer.remove(0);
            if c == '\n' {
                print!("\n");
            } else {
                let mut style = Style::new();
                if self.in_thinking {
                    style = style.dim().italic();
                } else {
                    if self.in_bold {
                        style = style.bold().yellow();
                    }
                    if self.in_inline_code {
                        style = style.green();
                    }
                }
                print!("{}", style.apply_to(c));
            }
        }
        print!("\n");
        std::io::stdout().flush().ok();
    }
}

async fn handle_tui(args: ServeArgs) {
    // 1. Resolve model ID just like serve does
    let model_id = match args.model_id.as_deref() {
        Some(id) if !id.is_empty() => id.to_string(),
        _ => {
            if let Some(ref path) = args.model_path {
                PathBuf::from(path)
                    .file_stem()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_else(|| "model".to_string())
            } else {
                select_model_interactively(args.select).await
            }
        }
    };

    let ServeArgs {
        model_id: _,
        model_path: _,
        bin,
        port,
        mmproj: _,
        embedding: _,
        timeout: _,
        n_gpu_layers,
        ctx_size: ctx_size_arg,
        threads: _,
        api_key,
        fit,
        detach: _,
        log: _,
        verbose,
        select: _,
    } = args;

    // Use fit to auto-adjust context size if desired, otherwise default
    let effective_fit = fit;
    let ctx_size = if fit { 0 } else { ctx_size_arg };

    // Start model server
    let (pid, actual_port, actual_api_key, server_state) = start_model_server(
        &model_id,
        bin,
        port,
        api_key.clone(),
        n_gpu_layers,
        ctx_size,
        effective_fit,
        verbose,
    )
    .await;

    #[allow(unreachable_patterns)]
    let is_llamacpp = match &server_state {
        ActiveServerState::Llama(_) => true,
        _ => false,
    };
    let mut thinking_mode = "auto";

    // Set console level to warn if verbose is enabled, to keep terminal clean
    if verbose {
        log::set_max_level(log::LevelFilter::Warn);
    }

    // Clear terminal screen and show a beautiful greeting
    print!("{}[2J{}[1;1H", 27 as char, 27 as char);
    std::io::stdout().flush().ok();

    let logo = make_logo();
    println!("{}", logo);
    
    let border_style = Style::new().cyan().bold();
    let text_style = Style::new().white().bold();
    let italic_style = Style::new().italic().dim();
    
    println!("  {}", border_style.apply_to("╭──────────────────────────────────────────────────────────╮"));
    println!("  {} {} {}", border_style.apply_to("│"), text_style.apply_to("                  JAN TERMINAL CHAT                     "), border_style.apply_to("│"));
    println!("  {} {} {}", border_style.apply_to("│"), italic_style.apply_to("  Local AI Chat · Offline & Private · OpenAI compatible "), border_style.apply_to("│"));
    println!("  {}", border_style.apply_to("╰──────────────────────────────────────────────────────────╯"));
    println!();
    println!("  Model loaded: {}", Style::new().green().bold().apply_to(&model_id));
    println!("  API Port:     {}", Style::new().magenta().bold().apply_to(actual_port));
    println!();
    println!("  Type your messages below. Press Enter to send.");
    println!("  Special commands:");
    println!("    {} or {}  - Exit chat and shut down server", Style::new().yellow().apply_to("/exit"), Style::new().yellow().apply_to("/quit"));
    println!("    {}            - Clear chat history", Style::new().yellow().apply_to("/clear"));
    println!("    {}  - Enable/disable/auto model thinking mode (default: auto)", Style::new().yellow().apply_to("/think [on/off/auto]"));
    println!("    {}             - Show this help message", Style::new().yellow().apply_to("/help"));
    println!("  {}", border_style.apply_to("━".repeat(60)));
    println!();

    let mut history: Vec<serde_json::Value> = Vec::new();
    let client = reqwest::Client::new();

    loop {
        // Read user input. On Windows, support Shift+Enter for newlines and Enter to submit.
        let input;
        #[cfg(windows)]
        let got_input = if console::user_attended() {
            print!("{}", Style::new().green().bold().apply_to("You ❯ "));
            std::io::stdout().flush().ok();
            read_multiline_input()
        } else {
            None
        };
        
        #[cfg(windows)]
        if let Some(inp) = got_input {
            input = inp;
        } else {
            // Fallback for non-TTY / piped or other platforms
            let mut line = String::new();
            if std::io::stdin().read_line(&mut line).is_ok() {
                let trimmed = line.trim().to_string();
                if trimmed.is_empty() {
                    break;
                }
                input = trimmed;
                if !console::user_attended() {
                    println!("You ❯ {}", input);
                }
            } else {
                break;
            }
        }

        #[cfg(not(windows))]
        {
            let input_result = dialoguer::Input::<String>::new()
                .with_prompt(format!("{}", Style::new().green().bold().apply_to("You ❯")))
                .interact_text();

            match input_result {
                Ok(inp) => input = inp.trim().to_string(),
                Err(_) => {
                    // Fallback for non-TTY / piped environments
                    let mut line = String::new();
                    if std::io::stdin().read_line(&mut line).is_ok() {
                        let trimmed = line.trim().to_string();
                        if trimmed.is_empty() {
                            break;
                        }
                        input = trimmed;
                        // Print input for visibility in piped tests
                        println!("You ❯ {}", input);
                    } else {
                        break;
                    }
                }
            }
        }

        if input.is_empty() {
            continue;
        }

        if input == "/exit" || input == "/quit" {
            break;
        }

        if input == "/clear" {
            history.clear();
            if let Ok(mut guard) = get_tui_code_blocks().lock() {
                guard.clear();
            }
            println!("\n  {} Conversation history cleared.\n", Style::new().yellow().apply_to("✓"));
            continue;
        }

        if input.starts_with("/think") {
            let parts: Vec<&str> = input.split_whitespace().collect();
            if parts.len() > 1 {
                let mode = parts[1].to_lowercase();
                if mode == "on" || mode == "off" || mode == "auto" {
                    thinking_mode = match mode.as_str() {
                        "on" => "on",
                        "off" => "off",
                        _ => "auto",
                    };
                    println!("\n  {} Thinking mode set to '{}'.\n", Style::new().green().bold().apply_to("✓"), thinking_mode);
                } else {
                    println!("\n  {} Invalid mode '{}'. Usage: /think [on/off/auto]\n", Style::new().red().bold().apply_to("✗"), parts[1]);
                }
            } else {
                println!("\n  Thinking mode is currently: {}\n  Usage: /think [on/off/auto]\n", Style::new().yellow().apply_to(thinking_mode));
            }
            continue;
        }

        if input == "/help" {
            println!();
            println!("  Commands:");
            println!("    /exit, /quit - Exit the chat");
            println!("    /clear       - Clear conversation history");
            println!("    /think [on/off/auto] - Enable/disable/auto model thinking mode");
            println!("    /c           - Copy the last code block to clipboard");
            println!("    /c <number>  - Copy the code block with the given index");
            println!("    /help        - Show this help");
            println!();
            continue;
        }

        if input.starts_with("/c") || input.starts_with("/copy") {
            let parts: Vec<&str> = input.split_whitespace().collect();
            let mut index_to_copy = None;
            
            if parts.len() > 1 {
                if let Ok(idx) = parts[1].parse::<usize>() {
                    if idx > 0 {
                        index_to_copy = Some(idx - 1);
                    }
                }
            } else {
                // Copy the last code block if no index is specified
                if let Ok(guard) = get_tui_code_blocks().lock() {
                    if !guard.is_empty() {
                        index_to_copy = Some(guard.len() - 1);
                    }
                }
            }
            
            if let Some(idx) = index_to_copy {
                let mut text_to_copy = None;
                if let Ok(guard) = get_tui_code_blocks().lock() {
                    if idx < guard.len() {
                        text_to_copy = Some(guard[idx].clone());
                    }
                }
                
                if let Some(text) = text_to_copy {
                    copy_to_clipboard(&text);
                    println!("\n  {} Copied code block {} to clipboard!\n", Style::new().green().bold().apply_to("✓"), idx + 1);
                } else {
                    println!("\n  {} Invalid code block index {}.\n", Style::new().red().bold().apply_to("✗"), idx + 1);
                }
            } else {
                println!("\n  {} No code blocks available to copy.\n", Style::new().red().bold().apply_to("✗"));
            }
            continue;
        }

        // Add user message to history
        history.push(serde_json::json!({
            "role": "user",
            "content": input,
        }));

        // Send request to completions endpoint
        let endpoint = format!("http://127.0.0.1:{}/v1/chat/completions", actual_port);
        let mut request_payload = serde_json::json!({
            "model": model_id,
            "messages": history,
            "stream": true,
        });

        if is_llamacpp && (thinking_mode == "on" || thinking_mode == "off") {
            let enable_thinking = thinking_mode == "on";
            request_payload["chat_template_kwargs"] = serde_json::json!({
                "enable_thinking": enable_thinking
            });
        }

        // Show a temporary loading indicator
        print!("\n");
        let assistant_label = Style::new().magenta().bold().apply_to("Jan ❯ ");
        print!("{}", assistant_label);
        std::io::stdout().flush().ok();

        let mut req = client.post(&endpoint);
        if !actual_api_key.is_empty() {
            req = req.header("Authorization", format!("Bearer {}", actual_api_key));
        }
        let response_result = req
            .json(&request_payload)
            .send()
            .await;

        let response = match response_result {
            Ok(resp) => resp,
            Err(e) => {
                println!("{}", Style::new().red().apply_to(format!("[Request failed: {}]", e)));
                println!();
                continue;
            }
        };

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            println!("{}", Style::new().red().apply_to(format!("[API Error ({}): {}]", status, text)));
            println!();
            continue;
        }

        let mut stream = response.bytes_stream();
        let mut buffer = Vec::new();
        let mut assistant_response_text = String::new();
        let mut renderer = TuiRenderer::new();
        let mut in_thinking = false;

        use futures::StreamExt;
        
        while let Some(chunk_res) = stream.next().await {
            let chunk = match chunk_res {
                Ok(c) => c,
                Err(e) => {
                    println!("\n{}", Style::new().red().apply_to(format!("[Stream error: {}]", e)));
                    break;
                }
            };
            buffer.extend_from_slice(&chunk);

            while let Some(pos) = buffer.iter().position(|&b| b == b'\n') {
                let line_bytes: Vec<u8> = buffer.drain(..pos + 1).collect();
                let line = String::from_utf8_lossy(&line_bytes);
                let trimmed = line.trim();

                if trimmed.is_empty() {
                    continue;
                }

                if trimmed.starts_with("data: ") {
                    let data_str = &trimmed[6..];
                    if data_str == "[DONE]" {
                        break;
                    }

                    if let Ok(json_val) = serde_json::from_str::<serde_json::Value>(data_str) {
                        if let Some(choices) = json_val["choices"].as_array() {
                            if let Some(first_choice) = choices.first() {
                                if let Some(reasoning) = first_choice["delta"]["reasoning_content"]
                                    .as_str()
                                    .or_else(|| first_choice["delta"]["reasoning"].as_str())
                                {
                                    if !reasoning.is_empty() {
                                        if !in_thinking {
                                            in_thinking = true;
                                            renderer.render_chunk("<think>\n");
                                            assistant_response_text.push_str("<think>\n");
                                        }
                                        renderer.render_chunk(reasoning);
                                        assistant_response_text.push_str(reasoning);
                                    }
                                }

                                if let Some(delta_content) = first_choice["delta"]["content"].as_str() {
                                    if !delta_content.is_empty() {
                                        if in_thinking {
                                            in_thinking = false;
                                            renderer.render_chunk("\n</think>\n\n");
                                            assistant_response_text.push_str("\n</think>\n\n");
                                        }
                                        assistant_response_text.push_str(delta_content);
                                        renderer.render_chunk(delta_content);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Process any leftover bytes in the stream buffer that didn't end with a newline
        if !buffer.is_empty() {
            let line = String::from_utf8_lossy(&buffer);
            let trimmed = line.trim();
            if !trimmed.is_empty() && trimmed.starts_with("data: ") {
                let data_str = &trimmed[6..];
                if data_str != "[DONE]" {
                    if let Ok(json_val) = serde_json::from_str::<serde_json::Value>(data_str) {
                        if let Some(choices) = json_val["choices"].as_array() {
                            if let Some(first_choice) = choices.first() {
                                if let Some(reasoning) = first_choice["delta"]["reasoning_content"]
                                    .as_str()
                                    .or_else(|| first_choice["delta"]["reasoning"].as_str())
                                {
                                    if !reasoning.is_empty() {
                                        if !in_thinking {
                                            in_thinking = true;
                                            renderer.render_chunk("<think>\n");
                                            assistant_response_text.push_str("<think>\n");
                                        }
                                        renderer.render_chunk(reasoning);
                                        assistant_response_text.push_str(reasoning);
                                    }
                                }

                                if let Some(delta_content) = first_choice["delta"]["content"].as_str() {
                                    if !delta_content.is_empty() {
                                        if in_thinking {
                                            in_thinking = false;
                                            renderer.render_chunk("\n</think>\n\n");
                                            assistant_response_text.push_str("\n</think>\n\n");
                                        }
                                        assistant_response_text.push_str(delta_content);
                                        renderer.render_chunk(delta_content);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            buffer.clear();
        }

        if in_thinking {
            renderer.render_chunk("\n</think>\n\n");
            assistant_response_text.push_str("\n</think>\n\n");
        }

        renderer.finish();
        println!();

        // Save assistant response to history
        if !assistant_response_text.is_empty() {
            history.push(serde_json::json!({
                "role": "assistant",
                "content": assistant_response_text,
            }));
        }
    }

    println!("\nShutting down model server...");
    kill_process(pid);
}

use std::sync::{Mutex, OnceLock};

static TUI_CODE_BLOCKS: OnceLock<Mutex<Vec<String>>> = OnceLock::new();

fn get_tui_code_blocks() -> &'static Mutex<Vec<String>> {
    TUI_CODE_BLOCKS.get_or_init(|| Mutex::new(Vec::new()))
}

fn copy_to_clipboard(text: &str) {
    #[cfg(windows)]
    {
        use std::os::windows::ffi::OsStrExt;
        let utf16: Vec<u16> = std::ffi::OsStr::new(text)
            .encode_wide()
            .chain(std::iter::once(0))
            .collect();
            
        unsafe {
            if win_job::OpenClipboard(std::ptr::null_mut()) != 0 {
                win_job::EmptyClipboard();
                let size = utf16.len() * 2;
                let h_mem = win_job::GlobalAlloc(0x0002, size); // GMEM_MOVEABLE = 0x0002
                if !h_mem.is_null() {
                    let ptr = win_job::GlobalLock(h_mem);
                    if !ptr.is_null() {
                        std::ptr::copy_nonoverlapping(utf16.as_ptr(), ptr as *mut u16, utf16.len());
                        win_job::GlobalUnlock(h_mem);
                        win_job::SetClipboardData(13, h_mem); // CF_UNICODETEXT = 13
                    } else {
                        win_job::GlobalFree(h_mem);
                    }
                }
                win_job::CloseClipboard();
            }
        }
    }
    
    #[cfg(target_os = "macos")]
    {
        if let Ok(mut child) = std::process::Command::new("pbcopy")
            .stdin(std::process::Stdio::piped())
            .spawn()
        {
            if let Some(mut stdin) = child.stdin.take() {
                let _ = stdin.write_all(text.as_bytes());
            }
            let _ = child.wait();
        }
    }

    #[cfg(target_os = "linux")]
    {
        if let Ok(mut child) = std::process::Command::new("xclip")
            .args(["-selection", "clipboard"])
            .stdin(std::process::Stdio::piped())
            .spawn()
        {
            if let Some(mut stdin) = child.stdin.take() {
                let _ = stdin.write_all(text.as_bytes());
            }
            let _ = child.wait();
        } else if let Ok(mut child) = std::process::Command::new("xsel")
            .args(["-b", "-i"])
            .stdin(std::process::Stdio::piped())
            .spawn()
        {
            if let Some(mut stdin) = child.stdin.take() {
                let _ = stdin.write_all(text.as_bytes());
            }
            let _ = child.wait();
        }
    }
}




#[cfg(windows)]
fn is_paste_waiting(h_in: win_job::HANDLE) -> bool {
    unsafe {
        let mut num_events = 0;
        if win_job::GetNumberOfConsoleInputEvents(h_in, &mut num_events) == 0 || num_events == 0 {
            return false;
        }

        let peek_len = std::cmp::min(num_events, 20);
        let mut buffer = vec![std::mem::zeroed::<win_job::INPUT_RECORD>(); peek_len as usize];
        let mut read = 0;
        if win_job::PeekConsoleInputW(h_in, buffer.as_mut_ptr(), peek_len, &mut read) == 0 || read == 0 {
            return false;
        }

        for record in buffer.iter().take(read as usize) {
            if record.event_type == 1 && record.key_event.b_key_down != 0 {
                let code = record.key_event.w_virtual_key_code;
                if code != 0x0D && code != 0x10 {
                    return true;
                }
            }
        }
        false
    }
}

#[cfg(windows)]
fn read_multiline_input() -> Option<String> {
    use std::io::Write;
    unsafe {
        let h_in = win_job::GetStdHandle(win_job::STD_INPUT_HANDLE);
        if h_in == std::ptr::null_mut() {
            return None;
        }

        let mut original_mode = 0;
        if win_job::GetConsoleMode(h_in, &mut original_mode) == 0 {
            return None;
        }

        // Enable raw input mode (disable line input, echo input, and mouse input; keep quick edit enabled)
        // ENABLE_LINE_INPUT: 0x0002
        // ENABLE_ECHO_INPUT: 0x0004
        // ENABLE_MOUSE_INPUT: 0x0010
        // ENABLE_QUICK_EDIT_MODE: 0x0040
        // ENABLE_EXTENDED_FLAGS: 0x0080
        // ENABLE_PROCESSED_INPUT: 0x0001
        let raw_mode = (original_mode & !0x0002 & !0x0004 & !0x0010) | 0x0001 | 0x0080;
        if win_job::SetConsoleMode(h_in, raw_mode) == 0 {
            return None;
        }

        let mut input = String::new();
        let mut stdout = std::io::stdout();

        loop {
            let mut record = std::mem::zeroed::<win_job::INPUT_RECORD>();
            let mut read = 0;
            if win_job::ReadConsoleInputW(h_in, &mut record, 1, &mut read) == 0 || read == 0 {
                break;
            }

            if record.event_type == 1 && record.key_event.b_key_down != 0 {
                let key_code = record.key_event.w_virtual_key_code;
                let unicode_char = record.key_event.u_char;

                // Detect shift state directly using user32 GetKeyState
                let is_shift = win_job::GetKeyState(0x10) < 0;

                if key_code == 0x0D { // Enter
                    let is_paste = is_paste_waiting(h_in);
                    if is_shift || is_paste {
                        input.push('\n');
                        print!("\n");
                        stdout.flush().ok();
                    } else {
                        print!("\n");
                        stdout.flush().ok();
                        break;
                    }
                } else if key_code == 0x08 { // Backspace
                    if !input.is_empty() {
                        let popped = input.pop();
                        if popped == Some('\n') {
                            print!("\x1b[A\x1b[999C");
                            stdout.flush().ok();
                        } else {
                            print!("\u{0008} \u{0008}");
                            stdout.flush().ok();
                        }
                    }
                } else if unicode_char != 0 {
                    if let Some(c) = char::from_u32(unicode_char as u32) {
                        if c >= ' ' || c == '\t' {
                            input.push(c);
                            print!("{}", c);
                            stdout.flush().ok();
                        } else if c == '\u{3}' { // Ctrl+C
                            win_job::SetConsoleMode(h_in, original_mode);
                            std::process::exit(0);
                        }
                    }
                }
            }
        }

        win_job::SetConsoleMode(h_in, original_mode);
        Some(input.trim().to_string())
    }
}

#[cfg(windows)]
mod win_job {
    pub type HANDLE = *mut std::ffi::c_void;
    
    #[repr(C)]
    struct IO_COUNTERS {
        read_operation_count: u64,
        write_operation_count: u64,
        other_operation_count: u64,
        read_transfer_count: u64,
        write_transfer_count: u64,
        other_transfer_count: u64,
    }

    #[repr(C)]
    struct JOBOBJECT_BASIC_LIMIT_INFORMATION {
        per_process_user_time_limit: i64,
        per_job_user_time_limit: i64,
        limit_flags: u32,
        minimum_working_set_size: usize,
        maximum_working_set_size: usize,
        active_process_limit: u32,
        affinity: usize,
        priority_class: u32,
        scheduling_class: u32,
    }

    #[repr(C)]
    struct JOBOBJECT_EXTENDED_LIMIT_INFORMATION {
        basic_limit_information: JOBOBJECT_BASIC_LIMIT_INFORMATION,
        io_info: IO_COUNTERS,
        process_memory_limit: usize,
        job_memory_limit: usize,
        peak_process_memory_used: usize,
        peak_job_memory_used: usize,
    }

    const JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE: u32 = 0x00002000;
    const JobObjectExtendedLimitInformation: i32 = 9;
    const PROCESS_SET_QUOTA: u32 = 0x0100;
    const PROCESS_TERMINATE: u32 = 0x0001;

    pub const STD_INPUT_HANDLE: u32 = 0xFFFFFFF6;

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct KEY_EVENT_RECORD {
        pub b_key_down: i32,
        pub w_repeat_count: u16,
        pub w_virtual_key_code: u16,
        pub w_virtual_scan_code: u16,
        pub u_char: u16,
        pub dw_control_key_state: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct INPUT_RECORD {
        pub event_type: u16,
        pub _pad: u16,
        pub key_event: KEY_EVENT_RECORD,
        pub _more_padding: [u32; 2],
    }

    extern "system" {
        fn CreateJobObjectW(lpJobAttributes: *mut std::ffi::c_void, lpName: *const u16) -> HANDLE;
        fn SetInformationJobObject(
            hJob: HANDLE,
            JobObjectInformationClass: i32,
            lpJobObjectInformation: *mut std::ffi::c_void,
            cbJobObjectInformation: u32,
        ) -> i32;
        fn OpenProcess(dwDesiredAccess: u32, bInheritHandle: i32, dwProcessId: u32) -> HANDLE;
        fn AssignProcessToJobObject(hJob: HANDLE, hProcess: HANDLE) -> i32;
        fn CloseHandle(hObject: HANDLE) -> i32;

        pub fn GetStdHandle(nStdHandle: u32) -> HANDLE;
        pub fn ReadConsoleInputW(
            hConsoleInput: HANDLE,
            lpBuffer: *mut INPUT_RECORD,
            nLength: u32,
            lpNumberOfEventsRead: *mut u32,
        ) -> i32;
        pub fn GetConsoleMode(hConsoleHandle: HANDLE, lpMode: *mut u32) -> i32;
        pub fn SetConsoleMode(hConsoleHandle: HANDLE, dwMode: u32) -> i32;
        
        pub fn GetNumberOfConsoleInputEvents(hConsoleInput: HANDLE, lpcNumberOfEvents: *mut u32) -> i32;
        pub fn PeekConsoleInputW(
            hConsoleInput: HANDLE,
            lpBuffer: *mut INPUT_RECORD,
            nLength: u32,
            lpNumberOfEventsRead: *mut u32,
        ) -> i32;
    }

    #[link(name = "user32")]
    extern "system" {
        pub fn GetKeyState(nVirtKey: i32) -> i16;
        
        pub fn OpenClipboard(hWndNewOwner: HANDLE) -> i32;
        pub fn CloseClipboard() -> i32;
        pub fn EmptyClipboard() -> i32;
        pub fn SetClipboardData(uFormat: u32, hMem: HANDLE) -> HANDLE;
    }

    #[link(name = "kernel32")]
    extern "system" {
        pub fn GlobalAlloc(uFlags: u32, dwBytes: usize) -> HANDLE;
        pub fn GlobalLock(hMem: HANDLE) -> HANDLE;
        pub fn GlobalUnlock(hMem: HANDLE) -> i32;
        pub fn GlobalFree(hMem: HANDLE) -> HANDLE;
    }



    static mut JOB_HANDLE: Option<HANDLE> = None;

    pub fn assign_process_to_kill_on_close_job(pid: u32) {
        unsafe {
            if JOB_HANDLE.is_none() {
                let job = CreateJobObjectW(std::ptr::null_mut(), std::ptr::null());
                if job == std::ptr::null_mut() {
                    return;
                }
                
                let mut info = std::mem::zeroed::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>();
                info.basic_limit_information.limit_flags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
                
                let res = SetInformationJobObject(
                    job,
                    JobObjectExtendedLimitInformation,
                    &mut info as *mut _ as *mut std::ffi::c_void,
                    std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
                );
                
                if res == 0 {
                    CloseHandle(job);
                    return;
                }
                JOB_HANDLE = Some(job);
            }

            if let Some(job) = JOB_HANDLE {
                let process = OpenProcess(PROCESS_SET_QUOTA | PROCESS_TERMINATE, 0, pid);
                if process != std::ptr::null_mut() {
                    AssignProcessToJobObject(job, process);
                    CloseHandle(process);
                }
            }
        }
    }
}

