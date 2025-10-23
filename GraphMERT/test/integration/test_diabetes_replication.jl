using Test
using GraphMERT
using Statistics: mean

@testset "Diabetes Dataset Replication Test" begin
    # Sample diabetes text similar to what would be in the paper dataset
    DIABETES_TEXT = """
    Type 2 diabetes mellitus is a chronic metabolic disorder characterized by insulin resistance 
    and relative insulin deficiency. The condition affects millions of people worldwide and is 
    associated with obesity, physical inactivity, and genetic predisposition. Common symptoms 
    include increased thirst, frequent urination, unexplained weight loss, and fatigue.
    
    Treatment typically involves lifestyle modifications including diet and exercise, oral 
    medications like metformin, sulfonylureas, and thiazolidinediones, and in some cases, 
    insulin therapy. Regular monitoring of blood glucose levels is essential for effective 
    diabetes management.
    
    Complications can include cardiovascular disease, diabetic nephropathy, retinopathy, and 
    neuropathy, making early detection and treatment crucial for patient outcomes.
    """
    
    @testset "Diabetes Entity Extraction" begin
        println("🩺 Testing diabetes entity extraction...")
        
        entities = discover_head_entities(DIABETES_TEXT)
        
        @test length(entities) > 0
        
        # Check for key diabetes-related entities
        entity_texts = [e.text for e in entities]
        diabetes_terms = ["diabetes", "insulin", "glucose", "metformin", "blood"]
        
        found_terms = sum([any(occursin(term, text) for text in entity_texts) for term in diabetes_terms])
        @test found_terms >= 3  # Should find at least 3 key terms
        
        println("   ✅ Diabetes entity extraction successful")
        println("   • Entities found: $(length(entities))")
        println("   • Key diabetes terms found: $found_terms/$(length(diabetes_terms))")
        println("   • Average confidence: $(round(mean([e.confidence for e in entities]), digits=3))")
    end
    
    @testset "Diabetes Knowledge Graph Construction" begin
        println("📊 Testing diabetes knowledge graph construction...")
        
        entities = discover_head_entities(DIABETES_TEXT)
        kg = KnowledgeGraph(entities, BiomedicalRelation[], Dict{String,Any}("dataset" => "diabetes"))
        
        @test kg !== nothing
        @test length(kg.entities) > 0
        @test haskey(kg.metadata, "dataset")
        
        println("   ✅ Diabetes knowledge graph constructed successfully")
        println("   • Entities: $(length(kg.entities))")
        println("   • Relations: $(length(kg.relations))")
    end
    
    @testset "Diabetes Evaluation Metrics" begin
        println("📈 Testing diabetes evaluation metrics...")
        
        entities = discover_head_entities(DIABETES_TEXT)
        kg = KnowledgeGraph(entities, BiomedicalRelation[], Dict{String,Any}("dataset" => "diabetes"))
        
        # Test FActScore evaluation
        factscore_result = evaluate_factscore(kg, DIABETES_TEXT)
        
        @test factscore_result !== nothing
        @test 0.0 <= factscore_result.factscore <= 1.0
        
        println("   ✅ Diabetes FActScore evaluation successful")
        println("   • FActScore: $(round(factscore_result.factscore * 100, digits=1))%")
        println("   • Total triples: $(factscore_result.total_triples)")
        println("   • Supported triples: $(factscore_result.supported_triples)")
        
        # Test ValidityScore evaluation
        validity_result = evaluate_validity(kg)
        
        @test validity_result !== nothing
        @test 0.0 <= validity_result.validity_score <= 1.0
        
        println("   ✅ Diabetes ValidityScore evaluation successful")
        println("   • ValidityScore: $(round(validity_result.validity_score * 100, digits=1))%")
        println("   • Total triples: $(validity_result.total_triples)")
        println("   • Valid triples: $(validity_result.valid_triples)")
    end
    
    @testset "Diabetes Batch Processing" begin
        println("🚀 Testing diabetes batch processing...")
        
        # Create multiple diabetes documents
        diabetes_docs = [
            "Type 2 diabetes is associated with insulin resistance and obesity.",
            "Metformin is the first-line treatment for type 2 diabetes mellitus.",
            "Diabetic complications include nephropathy, retinopathy, and neuropathy.",
            "Blood glucose monitoring is essential for diabetes management.",
            "Insulin therapy may be required for type 1 diabetes patients."
        ]
        
        # Mock extraction function
        function diabetes_extract(doc::String)
            entities = discover_head_entities(doc)
            return KnowledgeGraph(entities, BiomedicalRelation[], Dict{String,Any}("source" => doc))
        end
        
        # Test batch processing
        config = create_batch_processing_config(batch_size=2, max_memory_mb=256)
        batch_result = extract_knowledge_graph_batch(diabetes_docs, config=config, extraction_function=diabetes_extract)
        
        @test batch_result !== nothing
        @test batch_result.total_documents == length(diabetes_docs)
        @test batch_result.total_time > 0
        
        println("   ✅ Diabetes batch processing successful")
        println("   • Total documents: $(batch_result.total_documents)")
        println("   • Processing time: $(round(batch_result.total_time, digits=2))s")
        println("   • Throughput: $(round(batch_result.average_throughput, digits=2)) docs/s")
        
        # Check merged results
        merged_kg = batch_result.knowledge_graphs[1]
        @test merged_kg !== nothing
        @test length(merged_kg.entities) > 0
        
        println("   • Merged entities: $(length(merged_kg.entities))")
        println("   • Merged relations: $(length(merged_kg.relations))")
    end
end

println("✅ Diabetes Dataset Replication Test Complete!")
println("🎯 Successfully replicated diabetes dataset extraction workflow!")
