using Documenter

# Add the parent directory to the path to find GraphMERT
push!(LOAD_PATH, "..")
using GraphMERT

# Set up the documentation with minimal configuration
makedocs(
    sitename = "GraphMERT.jl",
    authors = "Alba Intelligence",
    format = Documenter.HTML(),
    source = ".",
    build = "docs_build",
    pages = [
        "Home" => "index.md",
        "Installation" => "getting_started/installation.md",
        "Quick Start" => "getting_started/quickstart.md",
        "Core Concepts" => "user_guide/core_concepts.md",
        "API Reference" => "api/core.md",
        "Troubleshooting" => "troubleshooting.md",
        "Changelog" => "changelog.md"
    ],
    modules = [GraphMERT],
    doctest = false,
    linkcheck = false,
    checkdocs = :none
)