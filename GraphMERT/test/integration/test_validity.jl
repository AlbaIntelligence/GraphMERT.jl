"""
Integration tests for validate_kg and ValidityReport (US2).

- validate_kg(kg, domain) returns ValidityReport with score, total_triples, valid_count.
- When domain is not registered, returns report with ontology_id=nothing (graceful degradation).
"""

using Test
using Dates
using GraphMERT

function _minimal_kg_with_relations(n::Int)
  entities = [
    GraphMERT.KnowledgeEntity("e$i", "Entity$i", "Entity$i", 0.9, GraphMERT.TextPosition(1, 1, 1, 1), Dict{String,Any}(), now())
    for i in 1:max(2, n)
  ]
  relations = [
    GraphMERT.KnowledgeRelation("e1", "e2", "REL", 0.8, Dict{String,Any}(), now())
    for _ in 1:n
  ]
  GraphMERT.KnowledgeGraph(entities, relations, Dict{String,Any}(), now())
end

@testset "validate_kg returns ValidityReport" begin
  kg = _minimal_kg_with_relations(3)
  report = GraphMERT.validate_kg(kg, "biomedical")
  @test report isa GraphMERT.ValidityReport
  @test report.total_triples >= 0
  @test report.valid_count >= 0
  @test 0.0 <= report.score <= 1.0 || (report.ontology_id === nothing && report.score == 0.0)
end

@testset "validate_kg with missing domain (graceful degradation)" begin
  kg = _minimal_kg_with_relations(2)
  report = GraphMERT.validate_kg(kg, "nonexistent_domain_xyz")
  @test report isa GraphMERT.ValidityReport
  @test report.ontology_id === nothing
  @test report.total_triples == length(kg.relations)
  @test report.valid_count == 0
  @test report.score == 0.0
end

@testset "ValidityReport fields" begin
  kg = _minimal_kg_with_relations(1)
  report = GraphMERT.validate_kg(kg, "biomedical")
  @test hasfield(GraphMERT.ValidityReport, :score)
  @test hasfield(GraphMERT.ValidityReport, :total_triples)
  @test hasfield(GraphMERT.ValidityReport, :valid_count)
  @test hasfield(GraphMERT.ValidityReport, :ontology_id)
end
