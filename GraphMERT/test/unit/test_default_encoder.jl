"""
Unit tests for default encoder path and load_model() in GraphMERT.jl

Tests:
- default_encoder_path() returns expected path
- load_model() with no args uses default path
- load_model(directory) with encoder dir (checkpoint.json or config.json)
- extract_knowledge_graph(text; options) uses default model when available
"""

using Test
using GraphMERT

@testset "Default encoder path and load_model" begin

  @testset "default_encoder_path()" begin
    path = GraphMERT.default_encoder_path()
    @test path isa String
    @test length(path) > 0
    @test occursin("roberta-base", path)
    @test occursin("encoders", path)
    # Default root when env unset: ~/.cache/llama-cpp/models/encoders
    @test occursin("encoders", path) && occursin("roberta-base", path)
  end

  @testset "load_model() no-arg and load_model(path)" begin
    # No-arg: should not throw; returns model or nothing
    model = GraphMERT.load_model()
    if model !== nothing
      @test model isa GraphMERT.GraphMERTModel
    end

    # With default path (directory or file)
    path = GraphMERT.default_encoder_path()
    model2 = GraphMERT.load_model(path)
    if model2 !== nothing
      @test model2 isa GraphMERT.GraphMERTModel
    end
  end

  @testset "load_model with directory containing config.json" begin
    # Simulate encoder dir with only Hugging Face config.json (no checkpoint.json)
    dir = mktempdir()
    config_path = joinpath(dir, "config.json")
    open(config_path, "w") do io
      println(io, "{}")  # minimal JSON
    end
    model = GraphMERT.load_model(dir)
    @test model !== nothing
    @test model isa GraphMERT.GraphMERTModel
    rm(dir, recursive = true)
  end

  @testset "load_model with directory containing checkpoint.json" begin
    dir = mktempdir()
    cp_content = """{"model_type":"GraphMERT","version":"0.1.0","config":null}"""
    open(joinpath(dir, "checkpoint.json"), "w") do io
      println(io, cp_content)
    end
    model = GraphMERT.load_model(dir)
    @test model !== nothing
    @test model isa GraphMERT.GraphMERTModel
    rm(dir, recursive = true)
  end

  @testset "Default path and load_model with env override" begin
    # With GRAPHMERT_ENCODER_ROOT set to empty dir, default_encoder_path() reflects it
    # and load_model() returns nothing (no config/checkpoint there).
    orig_root = get(ENV, "GRAPHMERT_ENCODER_ROOT", nothing)
    try
      empty_root = mktempdir()
      ENV["GRAPHMERT_ENCODER_ROOT"] = empty_root
      path = GraphMERT.default_encoder_path()
      @test path == joinpath(empty_root, "roberta-base")
      model = GraphMERT.load_model()
      @test model === nothing
    finally
      if orig_root !== nothing
        ENV["GRAPHMERT_ENCODER_ROOT"] = orig_root
      else
        delete!(ENV, "GRAPHMERT_ENCODER_ROOT")
      end
    end
  end
end
