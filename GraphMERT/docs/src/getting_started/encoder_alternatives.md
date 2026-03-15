# Encoder alternatives (if not using RoBERTa)

The paper and spec use a **RoBERTa** encoder (768-d, 12 layers). If you want to try other pretrained encoders — for speed, size, or domain — these are good options. The codebase expects **768-d** encoder output by default (H-GAT `input_dim = 768`); encoders with 384-d need a config change.

## Recommended encoders to try (download several)

| Encoder | Hugging Face ID | Hidden size | Params | Use case |
|--------|------------------|-------------|--------|----------|
| **RoBERTa-base** | `FacebookAI/roberta-base` | 768 | ~125M | Best paper alignment; drop-in. |
| **DistilBERT** | `distilbert-base-uncased` | 768 | ~66M | Lighter and faster; drop-in. |
| **PubMedBERT** | `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract` | 768 | ~110M | Biomedical; drop-in. Already default in `import_model_weights.jl`. |
| **BERT-base** | `bert-base-uncased` | 768 | ~110M | Classic BERT; drop-in. |
| **MiniLM-L6** | `sentence-transformers/all-MiniLM-L6-v2` | **384** | ~22M | Smallest/fastest; **requires** setting encoder and H-GAT to 384-d. |

**Recommendation:** Download **RoBERTa-base** and **DistilBERT** first (both 768-d, no code change). Add **PubMedBERT** if you care about biomedical; add **MiniLM-L6** if you want the smallest option and are okay changing `hidden_size` / `input_dim` to 384.

## How to download

### Option 1: Project script (Flux.jl conversion)

The package can pull from Hugging Face and convert to Flux. From the repo root:

```bash
cd GraphMERT
julia --project=. scripts/import_model_weights.jl --model-name FacebookAI/roberta-base --output-dir ./models/roberta-base
julia --project=. scripts/import_model_weights.jl --model-name distilbert-base-uncased --output-dir ./models/distilbert
julia --project=. scripts/import_model_weights.jl --model-name microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract --output-dir ./models/pubmedbert
```

See `GraphMERT/scripts/README.md` and `scripts/import_model_weights.jl` for more options (e.g. `--source local`, `--format onnx`).

### Option 2: Hugging Face CLI (raw PyTorch/ONNX)

To only download the weights (no conversion):

```bash
pip install huggingface_hub
huggingface-cli download FacebookAI/roberta-base --local-dir ~/.cache/graphmert/encoders/roberta-base
huggingface-cli download distilbert-base-uncased --local-dir ~/.cache/graphmert/encoders/distilbert
huggingface-cli download microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract --local-dir ~/.cache/graphmert/encoders/pubmedbert
```

Then point the persistence layer or your loader at the chosen directory. Weight loading into the Julia RoBERTa (or an adapter) is project-specific; see `reports/PROJECT_STATUS.md` and `reports/REFERENCE_SOURCES_AND_ENCODER.md`.

## Dimension compatibility

- **768-d encoders** (RoBERTa-base, DistilBERT, BERT-base, PubMedBERT): Match current `RoBERTaConfig` and H-GAT `input_dim = 768`. Use as-is once weights are wired.
- **384-d** (MiniLM-L6): Use only if you set encoder and H-GAT to 384 (e.g. `hidden_size = 384`, `input_dim = 384` in config). The notebook used MiniLM + PCA; here you’d use the 384-d output directly.

## Where this fits

These encoders replace or supplement the **GraphMERT encoder** (RoBERTa + H-GAT), not the **helper LLM**. Helper LLMs (GGUF in `~/.cache/llama-cpp/models`) stay as they are for entity/relation extraction. See [GGUF models (helper LLM only)](gguf_models.md) for the two model roles.
