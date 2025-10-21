"""
MNM (Masked Node Modeling) Training Demo

This example demonstrates the MNM training objective used in GraphMERT.
MNM trains the model to predict masked semantic tokens (leaf nodes) in the
leafy chain graph structure, enabling learning of semantic space representations.

Key concepts demonstrated:
1. Leaf masking selection based on probability
2. Mask application with 80/10/10 strategy (mask/random/keep)
3. Loss calculation for masked node prediction
4. Integration with the leafy chain graph structure
"""

using GraphMERT
using GraphMERT: MNMConfig, select_leaves_to_mask, apply_mnm_masks, create_empty_chain_graph, inject_triple!, SemanticTriple

println("=== MNM Training Demo ===")

# 1. Create configuration
println("\n1. Creating MNM configuration...")
config = MNMConfig(30522, 512, 7, 0.15, 0.3, 1.0, true, 103)
println("Config: vocab_size=$(config.vocab_size), mask_prob=$(config.mask_probability)")

# 2. Create graph and inject semantic triple
println("\n2. Creating graph with semantic content...")
graph = create_empty_chain_graph()

# Inject a test triple
triple = SemanticTriple("diabetes", nothing, "treats", "metformin", [2156, 23421], 0.95, "test")
success = inject_triple!(graph, triple, 1)
println("Triple injection successful: $success")
println("Injected tokens: $(triple.tail_tokens)")

# 3. Select leaves to mask
println("\n3. Selecting leaves to mask...")
masked_positions = select_leaves_to_mask(graph, config)
println("Selected $(length(masked_positions)) positions to mask")
for pos in masked_positions
    println("  - Root $(pos[1]), Leaf $(pos[2])")
end

# 4. Apply masking
println("\n4. Applying MNM masking...")
original_tokens = apply_mnm_masks(graph, masked_positions, config)
println("Original tokens before masking: $original_tokens")

# Check what masking was applied
for (i, (root_idx, leaf_idx)) in enumerate(masked_positions)
    leaf_node = graph.leaf_nodes[root_idx][leaf_idx]
    println("Position ($root_idx, $leaf_idx): token_id=$(leaf_node.token_id), is_padding=$(leaf_node.is_padding)")
end

# 5. Demonstrate graph structure integration
println("\n5. Graph structure integration...")
sequence = graph_to_sequence(graph)
println("Graph sequence length: $(length(sequence))")
println("First 10 tokens: $(sequence[1:10])")

attention_mask = create_attention_mask(graph)
println("Attention mask allows $(sum(attention_mask)) connections")

# 6. Show MNM training concept
println("\n6. MNM Training Concept:")
println("• Syntactic space (roots): Regular MLM on text tokens")
println("• Semantic space (leaves): MNM on KG triples")
println("• Joint training enables vocabulary transfer")
println("• Graph structure preserves semantic relationships")
println("• Attention decay based on graph distance")

println("\n✅ MNM Training demo complete!")
println("\nNext steps in full implementation:")
println("• Create training batches with multiple graphs")
println("• Implement gradient flow through H-GAT")
println("• Add relation embedding dropout")
println("• Integrate with MLM for joint training")
