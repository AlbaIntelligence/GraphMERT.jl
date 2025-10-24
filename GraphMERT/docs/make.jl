push!(LOAD_PATH, "..")

using Documenter
using DocStringExtensions
using GraphMERT

format = Documenter.HTML()

makedocs(
  sitename="GraphMERT",
  format=format,
  modules=[GraphMERT],
  pages=[
    "Home" => "index.md",
    "Installation" => "getting_started/installation.md",
    "Quick Start" => "getting_started/quickstart.md",
    "Core Concepts" => "user_guide/core_concepts.md",
    "API Reference" => "api/core.md",
    "Troubleshooting" => "troubleshooting.md",
    "Changelog" => "changelog.md"
  ],
  doctest=false,
  linkcheck=false,
  checkdocs=:none,
  clean=true,
  warnonly=[:missing_docs, :cross_references]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
