#!/usr/bin/env julia

"""
Simple demo of GraphMERT.jl functionality
"""

using Pkg
Pkg.activate(".")

using SparseArrays
using GraphMERT

println("=== GraphMERT.jl Demo ===")
println()

# Test 1: Configuration
println("1. Testing Configuration")
config = default_chain_graph_config()
println("   ChainGraphConfig: num_roots=$(config.num_roots), num_leaves=$(config.num_leaves_per_root), seq_len=$(config.max_sequence_length)")

# Test 2: Empty Graph Creation
println("\n2. Testing Empty Graph Creation")
tokens = [1, 2, 3]
texts = ["diabetes", "mellitus", "is"]
graph = create_empty_chain_graph(tokens, texts, config)
println("   Created graph with $(length(graph.nodes)) nodes")
println("   Root tokens: $(graph.root_tokens[1:3])")
println("   Num injections: $(graph.num_injections)")

# Test 3: Triple Injection
println("\n3. Testing Triple Injection")
inject_triple!(graph, 0, 0, [100, 101], "chronic disease", :isa, "diabetes")
println("   After injection: $(graph.num_injections) triples")
println("   Leaf tokens: $(graph.leaf_tokens[1, 1:2])")

# Test 4: Graph to Sequence
println("\n4. Testing Graph to Sequence")
sequence = graph_to_sequence(graph)
println("   Sequence length: $(length(sequence))")
println("   First 10 tokens: $(sequence[1:10])")

# Test 5: Adjacency Matrix
println("\n5. Testing Adjacency Matrix")
adj = build_adjacency_matrix(config)
println("   Adjacency matrix size: $(size(adj))")
println("   Non-zero entries: $(nnz(adj))")

# Test 6: Tokenizer
println("\n6. Testing Tokenizer")
tokenizer = BioMedTokenizer()
text = "Diabetes mellitus is a disease"
tokens_tok = tokenize(tokenizer, text)
println("   Tokenized: $(join(tokens_tok, " "))")

encoded = encode(tokenizer, text)
println("   Encoded length: $(length(encoded))")

println("\n=== Demo Complete ===")
println("GraphMERT.jl core functionality is working!")
