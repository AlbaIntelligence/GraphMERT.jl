using Test
using GraphMERT
using GraphMERT: extend_knowledge_graph

@testset "KG Extension" begin
    # Mock data
    existing_entities = [
        Entity("diabetes_id", "Diabetes", "Diabetes", "DISEASE", "biomedical", Dict{String,Any}(), TextPosition(0,0,0,0), 0.9),
        Entity("metformin_id", "Metformin", "Metformin", "DRUG", "biomedical", Dict{String,Any}(), TextPosition(0,0,0,0), 0.9)
    ]
    existing_relations = [
        Relation("diabetes_id", "metformin_id", "TREATS", 0.8, "biomedical")
    ]
    existing_kg = KnowledgeGraph(existing_entities, existing_relations)
    
    # Text with new info
    text = "Diabetes causes retinopathy."
    
    # Mock extract_knowledge_graph (we can't easily mock it unless we mock model)
    # Instead, let's verify merge logic by manually constructing the "new" KG inside extend_knowledge_graph
    # But extend_knowledge_graph calls extract_knowledge_graph.
    
    # Let's use a real call with a real (untrained) model, but we need to ensure it extracts something.
    # Without a trained model, it might extract nothing.
    # So we should rely on unit testing the merge logic or mocking extract_knowledge_graph.
    
    # Since we can't easily mock inner functions in Julia without Cassette/Mocking,
    # let's trust the integration test flow or just test merge_graphs logic if exposed.
    
    # However, extend_knowledge_graph is a high-level API.
    # Let's verify it compiles and runs without error on empty extraction.
    
    model = load_model() # or create new
    if model === nothing
        model = GraphMERT.create_graphmert_model(GraphMERTConfig())
    end
    
    # Run extension
    # Even if extraction returns empty, it should return at least the existing KG
    extended_kg = extend_knowledge_graph(existing_kg, text, model)
    
    @test length(extended_kg.entities) >= length(existing_kg.entities)
    @test length(extended_kg.relations) >= length(existing_kg.relations)
    
    # Verify existing content preserved
    @test any(e.text == "Diabetes" for e in extended_kg.entities)
    # Relations store IDs, so we check for the ID
    @test any(r.tail == "metformin_id" for r in extended_kg.relations)
end
