using Documenter
using GraphMERT

# Set up the documentation
makedocs(
    # Project information
    sitename = "GraphMERT.jl",
    authors = "Alba Intelligence",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://alba-intelligence.github.io/GraphMERT.jl",
        assets = ["assets/favicon.ico"],
        mathengine = Documenter.KaTeX()
    ),
    
    # Source and build directories
    source = ".",
    build = "build",
    
    # Documentation structure
    pages = [
        "Home" => "index.md",
        "Getting Started" => [
            "Installation" => "getting_started/installation.md",
            "Quick Start" => "getting_started/quickstart.md",
            "Basic Usage" => "getting_started/basic_usage.md"
        ],
        "User Guide" => [
            "Core Concepts" => "user_guide/core_concepts.md",
            "Knowledge Graph Extraction" => "user_guide/extraction.md",
            "Model Training" => "user_guide/training.md",
            "Evaluation Metrics" => "user_guide/evaluation.md",
            "Batch Processing" => "user_guide/batch_processing.md",
            "Performance Optimization" => "user_guide/performance.md"
        ],
        "Scientific Background" => [
            "Algorithm Overview" => "scientific/algorithm.md",
            "Research Paper" => "scientific/paper.md",
            "Performance Benchmarks" => "scientific/benchmarks.md",
            "Reproducibility" => "scientific/reproducibility.md"
        ],
        "Examples" => [
            "Basic Extraction" => "examples/basic_extraction.md",
            "Biomedical Processing" => "examples/biomedical_processing.md",
            "Training Pipeline" => "examples/training_pipeline.md",
            "Evaluation Workflow" => "examples/evaluation_workflow.md",
            "Batch Processing" => "examples/batch_processing.md",
            "Performance Benchmarking" => "examples/benchmarking.md"
        ],
        "API Reference" => [
            "Core API" => "api/core.md",
            "Data Structures" => "api/data_structures.md",
            "Configuration" => "api/configuration.md",
            "Evaluation" => "api/evaluation.md",
            "Utilities" => "api/utilities.md",
            "Serialization" => "api/serialization.md"
        ],
        "Developer Guide" => [
            "Architecture" => "developer/architecture.md",
            "Contributing" => "developer/contributing.md",
            "Testing" => "developer/testing.md",
            "Performance Guidelines" => "developer/performance.md"
        ],
        "Troubleshooting" => "troubleshooting.md",
        "Changelog" => "changelog.md"
    ],
    
    # Module documentation
    modules = [GraphMERT],
    
    # Documentation options
    doctest = true,
    linkcheck = true,
    checkdocs = :exports,
    strict = true,
    
    # Repository information
    repo = "https://github.com/alba-intelligence/GraphMERT.jl.git",
    devurl = "main"
)

# Deploy documentation (for CI/CD)
deploydocs(
    repo = "github.com/alba-intelligence/GraphMERT.jl.git",
    devbranch = "main",
    push_preview = true
)
