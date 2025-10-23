using Test
using GraphMERT
using Dates

@testset "Serialization Tests" begin
    @testset "export_knowledge_graph" begin
        # Create test knowledge graph
        entity1 = BiomedicalEntity("diabetes", "DISEASE", "C001", 0.8, TextPosition(1, 8, 1, 8))
        entity2 = BiomedicalEntity("insulin", "DRUG", "C002", 0.9, TextPosition(10, 16, 10, 16))
        relation1 = BiomedicalRelation("diabetes", "insulin", "treats", 0.8)
        
        kg = KnowledgeGraph([entity1, entity2], [relation1], Dict{String,Any}("source" => "test"), now())
        
        # Test JSON export
        json_result = export_knowledge_graph(kg, "json")
        @test json_result !== nothing
        @test occursin("diabetes", json_result)
        @test occursin("insulin", json_result)
        @test occursin("treats", json_result)
        
        # Test CSV export
        csv_result = export_knowledge_graph(kg, "csv")
        @test csv_result !== nothing
        @test occursin("entities", csv_result)
        @test occursin("relations", csv_result)
        
        # Test RDF export
        rdf_result = export_knowledge_graph(kg, "rdf")
        @test rdf_result !== nothing
        @test occursin("<?xml", rdf_result)
        @test occursin("rdf:RDF", rdf_result)
        
        # Test Turtle export
        ttl_result = export_knowledge_graph(kg, "ttl")
        @test ttl_result !== nothing
        @test occursin("@prefix", ttl_result)
        @test occursin("graphmert:", ttl_result)
        
        # Test unsupported format
        @test_throws ErrorException export_knowledge_graph(kg, "unsupported")
    end
    
    @testset "export_to_json" begin
        entity = BiomedicalEntity("test", "TYPE", "C123", 0.7, TextPosition(1, 4, 1, 4))
        kg = KnowledgeGraph([entity], BiomedicalRelation[], Dict{String,Any}("test" => true), now())
        
        json_result = export_to_json(kg)
        @test json_result !== nothing
        @test occursin("test", json_result)
        @test occursin("TYPE", json_result)
        @test occursin("C123", json_result)
        @test occursin("0.7", json_result)
    end
    
    @testset "export_to_csv" begin
        entity1 = BiomedicalEntity("entity1", "TYPE1", "C001", 0.8, TextPosition(1, 7, 1, 7))
        entity2 = BiomedicalEntity("entity2", "TYPE2", "C002", 0.9, TextPosition(8, 14, 8, 14))
        relation = BiomedicalRelation("entity1", "entity2", "relates", 0.7)
        
        kg = KnowledgeGraph([entity1, entity2], [relation], Dict{String,Any}("test" => true), now())
        
        csv_result = export_to_csv(kg, "test_export")
        @test csv_result !== nothing
        @test occursin("test_export_entities.csv", csv_result)
        @test occursin("test_export_relations.csv", csv_result)
    end
    
    @testset "export_to_rdf" begin
        entity = BiomedicalEntity("test_entity", "DISEASE", "C999", 0.6, TextPosition(1, 11, 1, 11))
        kg = KnowledgeGraph([entity], BiomedicalRelation[], Dict{String,Any}("test" => true), now())
        
        rdf_result = export_to_rdf(kg)
        @test rdf_result !== nothing
        @test occursin("<?xml", rdf_result)
        @test occursin("rdf:RDF", rdf_result)
        @test occursin("test_entity", rdf_result)
        @test occursin("DISEASE", rdf_result)
    end
    
    @testset "export_to_ttl" begin
        entity = BiomedicalEntity("test_entity", "DISEASE", "C999", 0.6, TextPosition(1, 11, 1, 11))
        kg = KnowledgeGraph([entity], BiomedicalRelation[], Dict{String,Any}("test" => true), now())
        
        ttl_result = export_to_ttl(kg)
        @test ttl_result !== nothing
        @test occursin("@prefix", ttl_result)
        @test occursin("graphmert:", ttl_result)
        @test occursin("test_entity", ttl_result)
        @test occursin("DISEASE", ttl_result)
    end
    
    @testset "Edge Cases" begin
        # Test empty knowledge graph
        empty_kg = KnowledgeGraph(BiomedicalEntity[], BiomedicalRelation[], Dict{String,Any}("empty" => true), now())
        
        json_result = export_to_json(empty_kg)
        @test json_result !== nothing
        @test occursin("entities", json_result)
        @test occursin("relations", json_result)
        
        csv_result = export_to_csv(empty_kg)
        @test csv_result !== nothing
        @test occursin("entities", csv_result)
        @test occursin("relations", csv_result)
    end
end

println("✅ Serialization Tests Complete!")
