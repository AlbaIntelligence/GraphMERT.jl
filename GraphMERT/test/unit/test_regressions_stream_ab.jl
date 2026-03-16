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
    tails = GraphMERT.form_tail_from_tokens(
      [(5, 0.9), (8, 0.8), (11, 0.7)],
      "Diabetes is treated with metformin therapy",
    )
    @test length(tails) == 3
    @test all(!isempty, tails)
    @test all(!startswith(t, "entity_") for t in tails)
  end

  @testset "train_graphmert performs real parameter updates" begin
    roberta = GraphMERT.RoBERTaConfig(
      vocab_size = 1000,
      hidden_size = 32,
      num_hidden_layers = 1,
      num_attention_heads = 4,
      intermediate_size = 64,
      max_position_embeddings = 64,
      type_vocab_size = 2,
      layer_norm_eps = 1e-12,
      hidden_dropout_prob = 0.0,
      attention_probs_dropout_prob = 0.0,
    )

    hgat = GraphMERT.HGATConfig(
      input_dim = 32,
      hidden_dim = 16,
      num_heads = 4,
      num_layers = 1,
      dropout_rate = 0.0,
      attention_dropout_rate = 0.0,
      layer_norm_eps = 1e-12,
      use_residual = true,
      use_layer_norm = true,
    )

    attn = GraphMERT.SpatialAttentionConfig(
      max_distance = 64,
      decay_lambda = 0.6,
      decay_p_init = 1.0,
      use_distance_bias = true,
      distance_bias_weight = 0.1,
    )

    cfg = GraphMERT.GraphMERTConfig(
      roberta_config = roberta,
      hgat_config = hgat,
      attention_config = attn,
      max_sequence_length = 32,
      hidden_dim = 32,
    )

    chain = GraphMERT.ChainGraphConfig(
      num_roots = 8,
      num_leaves_per_root = 3,
      max_sequence_length = 32,
      pad_token_id = 0,
      vocab_size = 1000,
    )

    mlm = GraphMERT.MLMConfig(
      vocab_size = 1000,
      hidden_size = 32,
      max_length = 32,
      mask_probability = 0.5,
      span_length = 3,
      mask_token_id = 103,
    )

    mnm = GraphMERT.MNMConfig(1000, 32, 3, 0.15, 0.3, 1.0, true, 103)

    model = GraphMERT.create_graphmert_model(cfg)
    ps0 = collect(GraphMERT.Flux.params(model))
    before = map(copy, ps0)

    GraphMERT.train_graphmert(
      ["GraphMERT tiny sanity run."],
      cfg;
      model = model,
      chain_config = chain,
      mlm_config = mlm,
      mnm_config = mnm,
      num_epochs = 1,
      max_steps_per_epoch = 1,
      learning_rate = 1e-3,
      checkpoint_dir = "",
      save_checkpoints = false,
      random_seed = 42,
    )

    ps1 = collect(GraphMERT.Flux.params(model))
    @test length(ps1) == length(before)

    deltas = [sum(abs.(ps1[i] .- before[i])) for i in eachindex(ps1)]
    @test any(>(0.0f0), deltas)
  end
end
