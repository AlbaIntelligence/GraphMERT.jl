using Test
using GraphMERT
using GraphMERT: FActScoreResult, KnowledgeGraph, Entity, Relation

# Note: The types in the test were using old names (KnowledgeEntity/Relation). 
# We need to check what are the actual exported types from GraphMERT.jl
# Based on extraction.jl imports: `using ..Types: KnowledgeGraph, BiomedicalEntity, BiomedicalRelation`
# But extraction.jl also exports `KnowledgeGraph`?
# Let's check `src/types.jl` to be sure.

# For now, let's assume `Entity` and `Relation` are the structs, as seen in `Distillation` module imports.

@testset "Evaluation Metrics" begin

    # --- Setup: Create a mock Knowledge Graph ---
    # Updated Entity constructor based on types.jl:
    # Entity(id, text, label, entity_type, domain, attributes, position, confidence, provenance)
    # Default values allow skipping most.
    # We want: Entity("E1", "Diabetes", "Disease", 1.0)
    # Mapping to: id="E1", text="Diabetes", label="Diabetes", entity_type="Disease", confidence=1.0
    
    e1 = Entity("E1", "Diabetes", "Diabetes", "Disease", "biomedical", Dict{String,Any}(), GraphMERT.TextPosition(0,0,0,0), 1.0, "")
    e2 = Entity("E2", "Metformin", "Metformin", "Drug", "biomedical", Dict{String,Any}(), GraphMERT.TextPosition(0,0,0,0), 1.0, "")
    e3 = Entity("E3", "Insulin", "Insulin", "Drug", "biomedical", Dict{String,Any}(), GraphMERT.TextPosition(0,0,0,0), 1.0, "")
    e4 = Entity("E4", "Glucose", "Glucose", "Substance", "biomedical", Dict{String,Any}(), GraphMERT.TextPosition(0,0,0,0), 1.0, "")
    
    entities = [e1, e2, e3, e4]
    
    # Updated Relation constructor:
    # Relation(head, tail, relation_type, confidence, domain, provenance, evidence, attributes, id)
    # We want: Relation("R1", "E2", "TREATS", "E1", 0.9, "sent1")
    # Mapping: head="E2", tail="E1", relation_type="TREATS", confidence=0.9, provenance="sent1"
    
    r1 = Relation(e2.id, e1.id, "TREATS", 0.9, "biomedical", "sent1", "sent1", Dict{String,Any}(), "R1")
    r2 = Relation(e3.id, e1.id, "TREATS", 0.95, "biomedical", "sent2", "sent2", Dict{String,Any}(), "R2")
    r3 = Relation(e1.id, e4.id, "ASSOCIATED_WITH", 0.8, "biomedical", "sent3", "sent3", Dict{String,Any}(), "R3")
    
    relations = [r1, r2, r3]
    
    # Use domain="general" to avoid requiring domain registration in this unit test.
    kg = KnowledgeGraph(entities, relations, Dict{String,Any}("domain" => "general"))
    
    text = "Metformin treats diabetes. Insulin treats diabetes. Diabetes is associated with glucose levels."

    # --- E1: FActScore* ---
    # We need to test evaluate_factscore
    # But it's not exported by default maybe?
    # It is defined in `src/evaluation/factscore.jl`.
    # Let's import it explicitly if needed or use GraphMERT.evaluate_factscore
    
    @testset "FActScore*" begin
        # Test heuristic evaluation
        # Note: evaluate_factscore expects text for context lookup
        # Keep this unit test registry-free
        result_heuristic = GraphMERT.evaluate_factscore(kg, text; confidence_threshold=0.5, include_domain_metrics=false)
        
        @test result_heuristic isa FActScoreResult
        @test result_heuristic.factscore >= 0.0 && result_heuristic.factscore <= 1.0
        # Given the text perfectly matches the triples, heuristic should find support
        @test result_heuristic.supported_triples > 0
    end
end
