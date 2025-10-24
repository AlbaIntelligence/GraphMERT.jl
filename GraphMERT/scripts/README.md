# GraphMERT Scripts

This directory contains utility scripts for GraphMERT.jl.

## Available Scripts

### import_model_weights.jl

Downloads and imports pre-trained model weights from various sources and converts them to Flux.jl format.

**Usage:**
```bash
julia scripts/import_model_weights.jl [options]
```

**Options:**
- `--model-name`: Name of the model to import (default: "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
- `--source`: Source of the model weights (default: "huggingface")
- `--output-dir`: Output directory for converted weights (default: "./models")
- `--format`: Input format (default: "pytorch")
- `--validate`: Validate imported weights (default: true)

**Examples:**
```bash
# Import from Hugging Face Hub
julia scripts/import_model_weights.jl --model-name microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract --source huggingface

# Import from local files
julia scripts/import_model_weights.jl --model-name my-model --source local --format pytorch

# Import with custom output directory
julia scripts/import_model_weights.jl --output-dir /path/to/models
```

## Supported Sources

- **Hugging Face Hub**: Download models from Hugging Face Model Hub
- **Local**: Load models from local files
- **URL**: Download from custom URLs

## Supported Formats

- **PyTorch**: Convert from PyTorch `.bin` files
- **TensorFlow**: Convert from TensorFlow SavedModel format
- **ONNX**: Convert from ONNX format

## Output

The script creates a directory structure:
```
output_dir/
├── config.json          # Model configuration
├── flux_weights.bson    # Converted Flux.jl weights
└── tokenizer.json       # Tokenizer configuration (if available)
```

## Requirements

- Julia 1.8+
- Flux.jl
- BSON.jl
- HTTP.jl
- FileIO.jl
