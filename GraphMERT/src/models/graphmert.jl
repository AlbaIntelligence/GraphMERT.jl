"""
GraphMERT model wrapper for GraphMERT.jl

This module implements the complete GraphMERT model that combines RoBERTa encoder
with H-GAT (Hierarchical Graph Attention) for biomedical knowledge graph construction.
"""

using Flux
using LinearAlgebra
using SparseArrays

# ============================================================================
# GraphMERT Model Types
# ============================================================================

"""
    GraphMERTConfig

Configuration for the complete GraphMERT model.

"""
struct GraphMERTConfig
    roberta_config::RoBERTaConfig
    hgat_config::HGATConfig
    attention_config::SpatialAttentionConfig
    entity_types::Vector{String}
    relation_types::Vector{String}
    max_sequence_length::Int
    hidden_dim::Int

    function GraphMERTConfig(;
        roberta_config::RoBERTaConfig = RoBERTaConfig(),
        hgat_config::HGATConfig = HGATConfig(),
        attention_config::SpatialAttentionConfig = SpatialAttentionConfig(),
        entity_types::Vector{String} = [
            "DISEASE",
            "DRUG",
            "PROTEIN",
            "SYMPTOM",
            "BIOMARKER",
        ],
        relation_types::Vector{String} = [
            "TREATS",
            "CAUSES",
            "ASSOCIATED_WITH",
            "INDICATES",
            "PREVENTS",
        ],
        max_sequence_length::Int = 1024,  # align with ChainGraphConfig (128 roots + 896 leaves)
        hidden_dim::Int = 768,
    )
        @assert roberta_config.max_position_embeddings >= max_sequence_length "RoBERTa position embeddings must cover max_sequence_length"
        new(
            roberta_config,
            hgat_config,
            attention_config,
            entity_types,
            relation_types,
            max_sequence_length,
            hidden_dim,
        )
    end
end

# ============================================================================
# Model Construction
# ============================================================================

"""
    GraphMERTModel

Complete GraphMERT model combining RoBERTa and H-GAT.

"""
struct GraphMERTModel
    roberta::RoBERTaModel
    hgat::HGATModel
    entity_classifier::Dense
    relation_classifier::Dense
    lm_head::Dense
    config::GraphMERTConfig

    function GraphMERTModel(
        roberta::RoBERTaModel,
        hgat::HGATModel,
        entity_classifier::Dense,
        relation_classifier::Dense,
        lm_head::Dense,
        config::GraphMERTConfig,
    )
        new(roberta, hgat, entity_classifier, relation_classifier, lm_head, config)
    end

    function GraphMERTModel(config::GraphMERTConfig)
        # Create RoBERTa model
        roberta = RoBERTaModel(config.roberta_config)

        # Create H-GAT model
        hgat = HGATModel(config.hgat_config)

        # Create entity classifier
        entity_classifier = Dense(config.hidden_dim, length(config.entity_types))

        # Create relation classifier
        relation_classifier = Dense(config.hidden_dim * 2, length(config.relation_types))

        # Create language modeling head for MLM/MNM (hidden_dim → vocab_size)
        lm_head = Dense(config.hidden_dim, config.roberta_config.vocab_size)

        return new(roberta, hgat, entity_classifier, relation_classifier, lm_head, config)
    end
end

"""
    create_graphmert_model(config::GraphMERTConfig)

Create a new GraphMERT model with the specified configuration.
"""
create_graphmert_model(config::GraphMERTConfig) = GraphMERTModel(config)

"""
    load_graphmert_model(model_path::String, config_path::String)

Load a pre-trained GraphMERT model from files.
"""
function load_graphmert_model(model_path::String, config_path::String)
    # Load configuration
    config = load_config(config_path)

    # Create model
    model = create_graphmert_model(config)

    # Load weights (this would be implemented with actual model loading)
    # For now, return the model with default weights
    return model
end

"""
    save_graphmert_model(model::GraphMERTModel, model_path::String, config_path::String)

Save a GraphMERT model to files.
"""
function save_graphmert_model(
    model::GraphMERTModel,
    model_path::String,
    config_path::String,
)
    # Save configuration
    save_config(model.config, config_path)

    # Save model weights (this would be implemented with actual model saving)
    # For now, just return success
    return true
end

# ============================================================================
# Forward Pass
# ============================================================================

"""
    (model::GraphMERTModel)(input_ids::Matrix{Int}, attention_mask::Matrix{Float32},
                           position_ids::Matrix{Int}, token_type_ids::Matrix{Int},
                           leafy_chain_graph::LeafyChainGraph)

Forward pass for the complete GraphMERT model.
"""
function (model::GraphMERTModel)(
    input_ids::Matrix{Int},
    attention_mask::Matrix{Float32},
    position_ids::Matrix{Int},
    token_type_ids::Matrix{Int},
    leafy_chain_graph::LeafyChainGraph,
)

    # RoBERTa encoding
    roberta_output, pooled_output =
        model.roberta(input_ids, attention_mask, position_ids, token_type_ids)

    # Create attention decay mask for the leafy chain graph
    attention_decay_mask = create_graph_attention_mask(
        leafy_chain_graph.adjacency_matrix,
        model.config.attention_config,
    )

    # H-GAT processing
    hgat_output = model.hgat(roberta_output, leafy_chain_graph.adjacency_matrix)

    # Entity classification
    entity_logits = model.entity_classifier(hgat_output)

    # Relation classification
    relation_logits =
        compute_relation_logits(hgat_output, leafy_chain_graph, model.relation_classifier)

    return entity_logits, relation_logits, hgat_output
end

"""
    compute_relation_logits(hgat_output::Matrix{Float32}, leafy_chain_graph::LeafyChainGraph,
                           relation_classifier::Dense)

Compute relation logits for all possible entity pairs.
"""
function compute_relation_logits(
    hgat_output::Matrix{Float32},
    leafy_chain_graph::LeafyChainGraph,
    relation_classifier::Dense,
)
    batch_size, seq_length, hidden_dim = size(hgat_output)
    num_relations = length(leafy_chain_graph.leaf_nodes)

    relation_logits =
        zeros(Float32, batch_size, num_relations, size(relation_classifier.weight, 1))

    for (rel_idx, (source, target)) in enumerate(leafy_chain_graph.edges)
        if source <= seq_length && target <= seq_length
            # Concatenate source and target embeddings
            source_embedding = hgat_output[:, source, :]
            target_embedding = hgat_output[:, target, :]
            pair_embedding = cat(source_embedding, target_embedding, dims = 2)

            # Classify relation
            relation_logits[:, rel_idx, :] = relation_classifier(pair_embedding)
        end
    end

    return relation_logits
end

# ============================================================================
# Entity and Relation Extraction
# ============================================================================

"""
    extract_entities(model::GraphMERTModel, input_ids::Matrix{Int}, attention_mask::Matrix{Float32},
                    position_ids::Matrix{Int}, token_type_ids::Matrix{Int},
                    leafy_chain_graph::LeafyChainGraph, confidence_threshold::Float32)

Extract entities from the input using the GraphMERT model.
"""
function extract_entities(
    model::GraphMERTModel,
    input_ids::Matrix{Int},
    attention_mask::Matrix{Float32},
    position_ids::Matrix{Int},
    token_type_ids::Matrix{Int},
    leafy_chain_graph::LeafyChainGraph,
    confidence_threshold::Float32,
)

    # Forward pass
    entity_logits, _, hgat_output =
        model(input_ids, attention_mask, position_ids, token_type_ids, leafy_chain_graph)

    # Apply softmax to get probabilities
    entity_probs = softmax(entity_logits, dims = 3)

    # Extract entities above confidence threshold
    entities = Vector{BiomedicalEntity}()

    for batch_idx = 1:size(entity_probs, 1)
        for token_idx = 1:size(entity_probs, 2)
            max_prob, max_class = findmax(entity_probs[batch_idx, token_idx, :])

            if max_prob >= confidence_threshold
                entity_type = model.config.entity_types[max_class]
                entity_text = get_token_text(input_ids, token_idx, batch_idx)
                entity_id = generate_entity_id(batch_idx, token_idx)

                # Create entity position
                position = TextPosition(token_idx, token_idx, 1, token_idx)

                # Create entity
                entity = BiomedicalEntity(
                    entity_id,
                    entity_text,
                    entity_type,
                    nothing,  # cui
                    String[],  # semantic_types
                    position,
                    max_prob,
                    "",  # provenance
                )
                push!(entities, entity)
            end
        end
    end

    return entities
end

"""
    extract_relations(model::GraphMERTModel, input_ids::Matrix{Int}, attention_mask::Matrix{Float32},
                    position_ids::Matrix{Int}, token_type_ids::Matrix{Int},
                    leafy_chain_graph::LeafyChainGraph, entities::Vector{BiomedicalEntity},
                    confidence_threshold::Float32)

Extract relations between entities using the GraphMERT model.
"""
function extract_relations(
    model::GraphMERTModel,
    input_ids::Matrix{Int},
    attention_mask::Matrix{Float32},
    position_ids::Matrix{Int},
    token_type_ids::Matrix{Int},
    leafy_chain_graph::LeafyChainGraph,
    entities::Vector{BiomedicalEntity},
    confidence_threshold::Float32,
)

    # Forward pass
    _, relation_logits, _ =
        model(input_ids, attention_mask, position_ids, token_type_ids, leafy_chain_graph)

    # Apply softmax to get probabilities
    relation_probs = softmax(relation_logits, dims = 3)

    # Extract relations above confidence threshold
    relations = Vector{BiomedicalRelation}()

    for batch_idx = 1:size(relation_probs, 1)
        for rel_idx = 1:size(relation_probs, 2)
            max_prob, max_class = findmax(relation_probs[batch_idx, rel_idx, :])

            if max_prob >= confidence_threshold
                relation_type = model.config.relation_types[max_class]

                # Get source and target entities
                if rel_idx <= length(leafy_chain_graph.edges)
                    source_idx, target_idx = leafy_chain_graph.edges[rel_idx]
                    source_entity = find_entity_by_position(entities, source_idx)
                    target_entity = find_entity_by_position(entities, target_idx)

                    if source_entity !== nothing && target_entity !== nothing
                        relation = BiomedicalRelation(
                            source_entity.id,
                            target_entity.id,
                            relation_type,
                            max_prob,
                            "",  # provenance
                            "",  # evidence
                        )
                        push!(relations, relation)
                    end
                end
            end
        end
    end

    return relations
end

# ============================================================================
# Knowledge Graph Construction
# ============================================================================

"""
    construct_knowledge_graph(model::GraphMERTModel, text::String, tokens::Vector{String},
                            semantic_nodes::Vector{String}, options::ProcessingOptions)

Construct a complete knowledge graph from text using the GraphMERT model.
"""
function construct_knowledge_graph(
    model::GraphMERTModel,
    text::String,
    tokens::Vector{String},
    semantic_nodes::Vector{String},
    options::ProcessingOptions,
)

    # Create leafy chain graph
    leafy_chain_graph = create_leafy_chain_graph(text, tokens, semantic_nodes)

    # Prepare input
    input_ids = tokenize_text(tokens)
    attention_mask = create_attention_mask(input_ids)
    position_ids = create_position_ids(size(input_ids, 1), size(input_ids, 2))
    token_type_ids = create_token_type_ids(size(input_ids, 1), size(input_ids, 2))

    # Extract entities
    entities = extract_entities(
        model,
        input_ids,
        attention_mask,
        position_ids,
        token_type_ids,
        leafy_chain_graph,
        options.confidence_threshold,
    )

    # Extract relations
    relations = extract_relations(
        model,
        input_ids,
        attention_mask,
        position_ids,
        token_type_ids,
        leafy_chain_graph,
        entities,
        options.confidence_threshold,
    )

    # Create model info
    model_info = GraphMERTModelInfo(
        "GraphMERT",
        "1.0",
        "RoBERTa+H-GAT",
        80_000_000,
        "Diabetes Dataset",
    )

    # Create knowledge graph
    knowledge_graph = KnowledgeGraph(
        entities,
        relations,
        options.confidence_threshold,
        model_info,
        Dict{String,String}(),
        0.0,
        0.0,
    )

    return knowledge_graph
end

# ============================================================================
# Utility Functions
# ============================================================================

"""
    get_token_text(input_ids::Matrix{Int}, token_idx::Int, batch_idx::Int)

Get the text representation of a token.
"""
function get_token_text(input_ids::Matrix{Int}, token_idx::Int, batch_idx::Int)
    # This would convert token ID back to text
    # For now, return a placeholder
    return "token_$(input_ids[token_idx, batch_idx])"
end

"""
    generate_entity_id(batch_idx::Int, token_idx::Int)

Generate a unique entity ID.
"""
function generate_entity_id(batch_idx::Int, token_idx::Int)
    return "entity_$(batch_idx)_$(token_idx)"
end

"""
    find_entity_by_position(entities::Vector{BiomedicalEntity}, position::Int)

Find an entity by its position in the text.
"""
function find_entity_by_position(entities::Vector{BiomedicalEntity}, position::Int)
    for entity in entities
        if entity.position.start <= position <= entity.position.stop
            return entity
        end
    end
    return nothing
end

"""
    get_model_parameters(model::GraphMERTModel)

Get the total number of parameters in the GraphMERT model.
"""
function get_model_parameters(model::GraphMERTModel)
    roberta_params = get_model_parameters(model.roberta)
    hgat_params = get_model_parameters(model.hgat)
    entity_params =
        length(model.entity_classifier.weight) + length(model.entity_classifier.bias)
    relation_params =
        length(model.relation_classifier.weight) + length(model.relation_classifier.bias)

    return roberta_params + hgat_params + entity_params + relation_params
end

"""
    get_model_size_mb(model::GraphMERTModel)

Get the approximate size of the model in megabytes.
"""
function get_model_size_mb(model::GraphMERTModel)
    # Estimate based on parameter count (assuming Float32)
    total_params = get_model_parameters(model)
    bytes_per_param = 4  # Float32
    total_bytes = total_params * bytes_per_param
    return total_bytes / (1024 * 1024)  # Convert to MB
end
