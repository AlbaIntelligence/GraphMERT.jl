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
    relation_embeddings::Embedding
    entity_classifier::Dense
    relation_classifier::Dense
    lm_head::Dense
    config::GraphMERTConfig

    function GraphMERTModel(
        roberta::RoBERTaModel,
        hgat::HGATModel,
        relation_embeddings::Embedding,
        entity_classifier::Dense,
        relation_classifier::Dense,
        lm_head::Dense,
        config::GraphMERTConfig,
    )
        new(roberta, hgat, relation_embeddings, entity_classifier, relation_classifier, lm_head, config)
    end

    function GraphMERTModel(config::GraphMERTConfig)
        # Create RoBERTa model
        roberta = RoBERTaModel(config.roberta_config)

        # Create H-GAT model
        hgat = HGATModel(config.hgat_config)

        # Create relation embeddings
        # +1 for padding/unknown relation
        relation_embeddings = Embedding(length(config.relation_types) + 1, config.hidden_dim)

        # Create entity classifier
        entity_classifier = Dense(config.hidden_dim, length(config.entity_types))

        # Create relation classifier
        relation_classifier = Dense(config.hidden_dim * 2, length(config.relation_types))

        # Create language modeling head for MLM/MNM (hidden_dim → vocab_size)
        lm_head = Dense(config.hidden_dim, config.roberta_config.vocab_size)

        return new(roberta, hgat, relation_embeddings, entity_classifier, relation_classifier, lm_head, config)
    end
end

Flux.@functor GraphMERTModel (roberta, hgat, relation_embeddings, entity_classifier, relation_classifier, lm_head)

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
    attention_mask::AbstractArray{Float32},
    position_ids::Matrix{Int},
    token_type_ids::Matrix{Int},
    leafy_chain_graph::LeafyChainGraph,
    ;
    attention_decay_mask::Union{GraphMERT.AttentionDecayMask, Nothing} = nothing,
    relation_ids::Union{Vector{Int}, Nothing} = nothing
)

    # 1. Get initial embeddings from RoBERTa
    embedding_output = model.roberta.embeddings(input_ids, position_ids, token_type_ids)

    # 2. Inject relation type embeddings into the sequence
    if relation_ids !== nothing
        # Optimized path: use pre-computed relation IDs
        # relation_ids is (seq_len,)
        
        rel_embeds = model.relation_embeddings(relation_ids) # (hidden_dim, seq_len)
        
        # Permute to (1, seq, hidden) to broadcast over batch
        rel_embeds_perm = permutedims(rel_embeds, (2, 1)) # (seq, hidden)
        rel_embeds_3d = reshape(rel_embeds_perm, 1, size(rel_embeds_perm, 1), size(rel_embeds_perm, 2)) # (1, seq, hidden)
        
        embedding_output = embedding_output .+ rel_embeds_3d
    else
        # Legacy/slow path: use get_relation_ids_for_sequence inside model
        # Note: this might trigger Zygote string issues if called inside gradient
        relation_ids_internal = get_relation_ids_for_sequence(leafy_chain_graph, model.config.relation_types)
        
        rel_embeds = model.relation_embeddings(relation_ids_internal)
        rel_embeds_perm = permutedims(rel_embeds, (2, 1))
        rel_embeds_3d = reshape(rel_embeds_perm, 1, size(rel_embeds_perm, 1), size(rel_embeds_perm, 2))
        embedding_output = embedding_output .+ rel_embeds_3d
    end

    # 3. Run H-GAT to refine embeddings based on graph structure
    # H-GAT takes (batch, nodes, hidden) which matches embedding_output
    hgat_output = model.hgat(embedding_output, leafy_chain_graph.adjacency_matrix)

    # 4. Prepare attention masks
    
    # Handle attention mask expansion if 2D (batch, seq) -> (batch, seq, seq)
    if ndims(attention_mask) == 2
        batch_size, seq_len = size(attention_mask) # Assume (batch, seq) or (seq, batch)?
        # input_ids is (seq, batch). attention_mask usually matches input_ids logic.
        # But RoBERTa uses (batch, seq, seq).
        # Let's assume attention_mask is (seq, batch) like input_ids.
        # We need to mask keys (columns) where padding is present.
        
        # Transpose to (batch, seq)
        mask_bs = permutedims(attention_mask, (2, 1))
        
        # Convert 1/0 to 0/-10000
        mask_vals = (1.0f0 .- mask_bs) .* -10000.0f0
        
        # Reshape to (batch, 1, seq)
        mask_3d = reshape(mask_vals, batch_size, 1, seq_len)
        
        # Broadcast/Repeat to (batch, seq, seq)
        attention_mask_3d = repeat(mask_3d, 1, seq_len, 1)
    else
        attention_mask_3d = attention_mask
    end

    # Create attention decay mask for spatial bias (use passed one if available)
    if attention_decay_mask === nothing
        attention_decay_mask = create_graph_attention_mask(
            leafy_chain_graph.adjacency_matrix,
            model.config.attention_config,
        )
    end

    # 5. Run RoBERTa encoder
    # We use hgat_output as the input to the encoder
    encoder_output = model.roberta.encoder(hgat_output, attention_mask_3d, attention_decay_mask)

    # 6. Entity classification
    # encoder_output is (batch, seq, hidden)
    # Dense layer expects (hidden, ...)
    
    # Permute to (hidden, seq, batch)
    encoder_permuted = permutedims(encoder_output, (3, 2, 1))
    
    # Apply classifier (Dense broadcasts over other dimensions)
    # Result: (num_entities, seq, batch)
    entity_logits_permuted = model.entity_classifier(encoder_permuted)
    
    # Permute back to (batch, seq, num_entities)
    entity_logits = permutedims(entity_logits_permuted, (3, 2, 1))

    # 7. Relation classification
    # Use encoder output for final classification
    relation_logits = compute_relation_logits(encoder_output, leafy_chain_graph, model.relation_classifier)

    # 8. LM Head (Masked Language Modeling / Masked Neighbor Modeling)
    # encoder_output is (batch, seq, hidden)
    # lm_head is Dense(hidden, vocab) -> expects (hidden, N)
    
    batch_size, seq_len, hidden_dim = size(encoder_output)
    
    # Permute to (hidden, seq, batch)
    encoder_permuted = permutedims(encoder_output, (3, 2, 1))
    
    # Reshape to (hidden, seq * batch)
    encoder_reshaped = reshape(encoder_permuted, hidden_dim, seq_len * batch_size)
    
    # Apply LM head
    lm_output_flat = model.lm_head(encoder_reshaped) # (vocab, seq * batch)
    
    # Reshape back to (vocab, seq, batch)
    lm_logits_permuted = reshape(lm_output_flat, size(lm_output_flat, 1), seq_len, batch_size)
    
    # Permute to (batch, seq, vocab) to match expected output format
    lm_logits = permutedims(lm_logits_permuted, (3, 2, 1))

    return entity_logits, relation_logits, lm_logits, encoder_output
end

"""
    get_relation_ids_for_sequence(graph, relation_types)

Generate relation IDs for the sequence to allow vectorized embedding lookup.
Returns vector of Ints (1-based indices into relation_embeddings).
Index 1 is reserved for "NO_RELATION" (padding).
"""
function get_relation_ids_for_sequence(
    graph::LeafyChainGraph,
    relation_types::Vector{String},
)::Vector{Int}
    # Calculate sequence length based on graph config
    # Note: graph_to_sequence uses max_sequence_length
    seq_len = graph.config.max_sequence_length
    
    # Initialize with 1 (padding/no relation)
    rel_ids = ones(Int, seq_len)
    
    num_roots = graph.config.num_roots
    num_leaves = graph.config.num_leaves_per_root
    
    # Iterate roots to find relations injected into leaves
    for r in 1:num_roots
        rel_sym = graph.leaf_relations[r, 1]
        
        if rel_sym !== nothing
            rel_str = string(rel_sym)
            idx = findfirst(==(rel_str), relation_types)
            
            if idx !== nothing
                # Map to embedding index (idx + 1, since 1 is padding)
                emb_idx = idx + 1
                
                # Apply to all injected leaves
                for l in 1:num_leaves
                    if graph.injected_mask[r, l]
                        # Calculate position (1-based for Julia arrays)
                        # Root tokens are 1..num_roots
                        # Leaves start at num_roots + 1
                        pos = num_roots + (r-1)*num_leaves + l
                        
                        if pos <= seq_len
                            rel_ids[pos] = emb_idx
                        end
                    end
                end
            end
        end
    end
    
    return rel_ids
end

"""
    inject_relation_types!(embeddings, relation_embeddings, graph, relation_types)

Deprecated: Use get_relation_ids_for_sequence and vector addition instead.
Kept for backward compatibility if needed, but not Zygote-safe.
"""
function inject_relation_types!(
    embeddings::AbstractArray{Float32, 3},
    relation_embeddings::Embedding,
    graph::LeafyChainGraph,
    relation_types::Vector{String},
)
    batch_size = size(embeddings, 1)
    num_roots = graph.config.num_roots
    num_leaves = graph.config.num_leaves_per_root

    for r in 1:num_roots
        # Check if this root has a relation (all injected leaves share it)
        rel_sym = graph.leaf_relations[r, 1]
        
        if rel_sym !== nothing
            rel_str = string(rel_sym)
            rel_id = findfirst(==(rel_str), relation_types)
            
            if rel_id !== nothing
                # Get embedding vector (hidden_dim,)
                rel_vec = relation_embeddings(rel_id)
                
                # Apply to all injected leaves of this root
                for l in 1:num_leaves
                    if graph.injected_mask[r, l]
                        # Calculate position
                        pos = num_roots + (r-1)*num_leaves + l
                        
                        # Add relation embedding to existing token embedding
                        for b in 1:batch_size
                            embeddings[b, pos, :] .+= rel_vec
                        end
                    end
                end
            end
        end
    end
end

"""
    compute_relation_logits(hgat_output::Matrix{Float32}, leafy_chain_graph::LeafyChainGraph,
                           relation_classifier::Dense)

Compute relation logits for all possible entity pairs.
"""
function compute_relation_logits(
    hgat_output::AbstractArray{Float32, 3},
    leafy_chain_graph::LeafyChainGraph,
    relation_classifier::Dense,
)
    # Placeholder implementation to avoid crashes.
    # The original implementation iterated over `edges` which is not present in LeafyChainGraph.
    # Real implementation should likely iterate over relevant node pairs (e.g. root-leaf).
    
    batch_size, seq_length, hidden_dim = size(hgat_output)
    # Output shape: (batch, 1, num_relations) - just a dummy return for now
    
    return zeros(Float32, batch_size, 1, size(relation_classifier.weight, 1))
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
