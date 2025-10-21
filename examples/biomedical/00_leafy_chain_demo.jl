"""
Leafy Chain Graph Demonstration

This example demonstrates the basic functionality of the leafy chain graph structure
used in GraphMERT for combining syntactic and semantic representations.

The leafy chain graph enables:
1. Unified representation of text tokens (roots) and KG triples (leaves)
2. Sequential encoding for transformer input
3. Graph-aware attention mechanisms
4. Joint training on both syntactic and semantic spaces
"""

using GraphMERT

# Import specific types for the example
using GraphMERT: SemanticTriple

println("=== Leafy Chain Graph Demo ===")

# 1. Create configuration
println("\n1. Creating configuration...")
config = default_chain_graph_config()
println("Config: $(config.num_roots) roots, $(config.num_leaves_per_root) leaves per root")

# 2. Create empty graph
println("\n2. Creating empty graph...")
graph = create_empty_chain_graph(config)
println("Created graph with $(length(graph.root_nodes)) root nodes")
println("Total nodes: $(config.num_roots * (1 + config.num_leaves_per_root))")

# 3. Display adjacency matrix structure
println("\n3. Graph connectivity (adjacency matrix):")
adj = build_adjacency_matrix(graph)
println("Adjacency matrix size: $(size(adj))")
println("Connected root-leaf pairs: $(sum(adj[1:config.num_roots, config.num_roots+1:end]))")

# 4. Compute shortest paths
println("\n4. Computing shortest paths...")
dist = floyd_warshall(graph)
println("Shortest paths matrix size: $(size(dist))")
println("Max distance: $(maximum(dist[dist .> 0]))")

# 5. Create attention mask
println("\n5. Creating attention mask...")
mask = create_attention_mask(graph)
println("Attention mask allows $(sum(mask)) connections")

# 6. Inject a semantic triple
println("\n6. Injecting semantic triple...")
triple = SemanticTriple("diabetes", nothing, "treats", "metformin", [2156, 23421], 0.95, "UMLS")
success = inject_triple!(graph, triple, 1)
println("Injection successful: $success")

if success
  println("Injected tokens: $(triple.tail_tokens)")
  println("Root 1 token ID: $(graph.root_nodes[1].token_id)")
  println("Leaf 1 token ID: $(graph.leaf_nodes[1][1].token_id)")
  println("Leaf 2 token ID: $(graph.leaf_nodes[1][2].token_id)")
end

# 7. Convert to sequence
println("\n7. Converting graph to sequence...")
sequence = graph_to_sequence(graph)
println("Sequence length: $(length(sequence))")
println("First 10 tokens: $(sequence[1:10])")
println("Last 10 tokens: $(sequence[end-9:end])")

# 8. Create graph from text
println("\n8. Creating graph from text...")
text = "Diabetes mellitus is a chronic metabolic disorder"
text_graph = create_leafy_chain_from_text(text, config)
println("Text graph created with $(length(text_graph.root_nodes)) roots")

# 9. Show graph structure
println("\n9. Graph structure summary:")
println("- Roots: $(config.num_roots) (text tokens)")
println("- Leaves per root: $(config.num_leaves_per_root) (semantic tokens)")
println("- Total nodes: $(config.num_roots * (1 + config.num_leaves_per_root))")
println("- Sequence length: $(length(graph_to_sequence(graph)))")
println("- Attention connections: $(sum(mask))")

println("\n✅ Leafy Chain Graph demo complete!")
println("\nThis structure enables GraphMERT to:")
println("• Unify syntactic (text) and semantic (KG) spaces")
println("• Enable joint MLM+MNM training")
println("• Apply graph-aware attention mechanisms")
println("• Transfer vocabulary between spaces")
