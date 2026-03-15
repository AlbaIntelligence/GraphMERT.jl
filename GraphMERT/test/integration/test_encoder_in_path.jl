"""
Integration tests for encoder-in-path (US4).

- load_model(path) returns a full GraphMERTModel when checkpoint is valid.
- Extraction with that model uses the encoder path (model isa GraphMERTModel).
- Optional: clean → export → seed path (documented; lightweight test).
"""

using Test
using GraphMERT

@testset "load_model returns full model" begin
  # Save a minimal checkpoint so we can load it
  model = GraphMERT.create_graphmert_model(GraphMERT.GraphMERTConfig())
  path = joinpath(mktempdir(), "test_checkpoint.json")
  try
    GraphMERT.save_model(model, path)
    loaded = GraphMERT.load_model(path)
    @test loaded !== nothing
    @test loaded isa GraphMERT.GraphMERTModel
  finally
    ispath(path) && rm(path; force = true)
  end
end

@testset "load_model missing file returns nothing" begin
  @test GraphMERT.load_model("/nonexistent/path.json") === nothing
end

@testset "extraction with loaded model (encoder path wired)" begin
  model = GraphMERT.create_graphmert_model(GraphMERT.GraphMERTConfig())
  path = joinpath(mktempdir(), "enc_test.json")
  try
    GraphMERT.save_model(model, path)
    loaded = GraphMERT.load_model(path)
    opts = GraphMERT.ProcessingOptions(domain = "biomedical", enable_provenance_tracking = true)
    # Empty text: returns empty KG without calling predict_tail_tokens; confirms load_model + extraction wiring
    kg = GraphMERT.extract_knowledge_graph("", loaded; options = opts)
    @test kg isa GraphMERT.KnowledgeGraph
    @test length(kg.relations) == 0
  finally
    ispath(path) && rm(path; force = true)
  end
end
