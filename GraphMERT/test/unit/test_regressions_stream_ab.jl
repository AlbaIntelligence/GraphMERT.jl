using Test
using GraphMERT

@testset "Stream A/B Regression Tests" begin
  @testset "Confidence filtering uses real relations" begin
    entities = [
      GraphMERT.Entity("e1", "diabetes", "diabetes", "DISEASE", "biomedical"),
      GraphMERT.Entity("e2", "metformin", "metformin", "DRUG", "biomedical"),
      GraphMERT.Entity("e3", "glucose", "glucose", "BIOMARKER", "biomedical"),
    ]

    relations = [
      GraphMERT.Relation("e1", "e2", "TREATS", 0.9, "biomedical"),
      GraphMERT.Relation("e2", "e3", "AFFECTS", 0.4, "biomedical"),
      GraphMERT.Relation("missing", "e3", "BROKEN", 0.95, "biomedical"),
    ]

    kg = GraphMERT.KnowledgeGraph(
      entities,
      relations,
      Dict{String,Any}("source_text" => "diabetes is treated with metformin"),
    )
    triples = GraphMERT.filter_triples_by_confidence(kg, 0.5)

    @test triples == [(1, 1, 2)]
  end

  @testset "Position embeddings align with 1024 sequence length" begin
    roberta = GraphMERT.RoBERTaConfig()
    @test roberta.max_position_embeddings == 1024

    default_cfg = GraphMERT.get_default_config()
    @test default_cfg["model"]["max_position_embeddings"] == 1024
  end

  @testset "Tail formation returns text-derived candidates" begin
    tails = GraphMERT.form_tail_from_tokens([(5, 0.9), (8, 0.8), (11, 0.7)], "Diabetes is treated with metformin therapy")
    @test length(tails) == 3
    @test all(!isempty, tails)
    @test all(!startswith(t, "entity_") for t in tails)
  end
end
