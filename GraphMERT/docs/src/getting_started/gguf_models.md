# GGUF Models for llama-cpp (Helper LLM only)

GraphMERT uses **two different kinds of models**. Do not confuse them.

| Role | What it is | Where it lives | Matches the paper? |
|------|------------|----------------|---------------------|
| **GraphMERT encoder** | RoBERTa + H-GAT (leafy chain, MLM/MNM) | Default: `~/.cache/llama-cpp/models/encoders/roberta-base`; load via `load_model()` or `load_model(path)` | **Yes** — this is the methodology in the paper |
| **Helper LLM** | Optional local LLM for entity/relation extraction (e.g. TinyLlama GGUF) | e.g. `~/.cache/llama-cpp/models/*.gguf`; set via `LocalLLMConfig(model_path = "...")` | **No** — auxiliary only; not RoBERTa |

**The 2 models in `~/.cache/llama-cpp/models` are helper LLMs** (GGUF, for entity/relation discovery). They are **not** RoBERTa and **do not** implement the paper’s encoder. The **paper-matching encoder** is RoBERTa + H-GAT: use **`load_model()`** (default path `~/.cache/llama-cpp/models/encoders/roberta-base`) or `load_model("path/to/checkpoint")`. Override with `GRAPHMERT_ENCODER_ROOT`. See `reports/REFERENCE_SOURCES_AND_ENCODER.md` and `reports/PROJECT_STATUS.md` in the repo root.

This page only covers **downloading GGUF helper LLMs** for the local-LLM extraction path (replacing ~/.ollama/models).

---

## Why not use ~/.ollama/models?

- Ollama keeps model blobs and manifests in its own layout under `~/.ollama/models`.
- Llama-cpp (and LlamaCpp.jl) expect a **single `.gguf` file** per model.
- You need to download GGUF versions from Hugging Face (or similar) into a directory that GraphMERT will use via `LocalLLMConfig(model_path = "...")`.

## Where to put GGUF models

Recommended directory (optional; you can use any path):

```text
~/.cache/llama-cpp/models/
```

Or set a custom directory and pass the full path to the `.gguf` file in `LocalLLMConfig(model_path = "...")`.

## Downloading GGUF models

### Option 1: Script (recommended)

From the repository root:

```bash
./GraphMERT/scripts/download_gguf_models.sh [TARGET_DIR]
```

If `TARGET_DIR` is omitted, the script uses `~/.cache/llama-cpp/models`. See the script help for presets (e.g. TinyLlama, SmolLM).

### Option 2: Hugging Face CLI

Install [huggingface-hub](https://huggingface.co/docs/huggingface_hub/installation) and run:

```bash
export HF_HOME=~/.cache/huggingface  # optional
mkdir -p ~/.cache/llama-cpp/models
huggingface-cli download TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf --local-dir ~/.cache/llama-cpp/models
```

Then in Julia:

```julia
LocalLLMConfig(model_path = expanduser("~/.cache/llama-cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"))
```

### Option 3: Manual download

1. Open a GGUF model page on Hugging Face (e.g. [TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF)).
2. Download the `.gguf` file you want (e.g. Q4_K_M for a balance of size and quality).
3. Save it to e.g. `~/.cache/llama-cpp/models/` and use that path in `LocalLLMConfig`.

## Model presets (Ollama → GGUF equivalents)

If you previously used these with Ollama, use the following GGUF sources for llama-cpp:

| Former Ollama model | GGUF alternative (Hugging Face) | Typical file | Notes |
|--------------------|----------------------------------|--------------|--------|
| tinyllama          | TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF | tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf | ~700MB, CPU-friendly |
| qwen2.5:0.5b / qwen3:0.6b | Qwen/Qwen2.5-0.5B-Instruct-GGUF (or community GGUF) | qwen2.5-0.5b-instruct-q4_k_m.gguf | Small, fast |
| smollm2 / ministral | SmolLM or Mistral GGUF repos | Check repo for Q4_K_M or similar | Better quality, more RAM |
| lfm2.5-thinking    | Search HF for “lfm 2.5” or use a similar small instruct model | — | No direct 1:1; pick a similar-sized instruct GGUF |

Quantization tips:

- **Q4_K_M**: Good balance of size and quality for CPU.
- **Q5_K_M**: Higher quality, larger.
- **Q8_0**: Closer to full precision, largest.

## Using the downloaded model in GraphMERT

The GGUF file is used only as the **helper LLM** (entity/relation extraction). You can still pass a **GraphMERT model** (RoBERTa checkpoint) to `extract_knowledge_graph` for the full paper pipeline; the helper LLM and the encoder are independent.

```julia
using GraphMERT
using GraphMERT.LocalLLM

# Helper LLM (GGUF) — for head/relation discovery
helper_path = expanduser("~/.cache/llama-cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
options = ProcessingOptions(
    domain = "wikipedia",
    use_local = true,
    local_config = LocalLLMConfig(model_path = helper_path),
)

# Paper encoder (RoBERTa + H-GAT) — load from your checkpoint path if you have one
model = load_model("path/to/graphmert_checkpoint")  # or nothing for extraction without tail prediction
kg = extract_knowledge_graph("Your text here.", model; options = options)
```

## Next steps

- [Quickstart](quickstart.md) and [API](../api/index.md) for extraction and `LocalLLMConfig`
- [Encoder alternatives](encoder_alternatives.md) — if not using RoBERTa, which encoder to try and how to download several (DistilBERT, PubMedBERT, MiniLM, etc.)
- Repo root `reports/REFERENCE_SOURCES_AND_ENCODER.md` and `reports/PROJECT_STATUS.md` for the RoBERTa encoder and checkpoint location
