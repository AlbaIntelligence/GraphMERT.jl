"""
Tests for seed KG injection functionality (EPIC 2).

This test suite validates the training data preparation pipeline,
including entity linking, triple retrieval, and seed injection algorithms.
"""

using Test
using GraphMERT

@testset "EPIC 2: Seed KG Injection" begin

    @testset "SapBERT Entity Linking" begin
        @testset "Basic Interface" begin
            entities = ["diabetes mellitus", "insulin"]
            result = link_entities_sapbert(entities)

            @test length(result) == 2
            @test result[1][:entity_text] == "diabetes mellitus"
            @test haskey(result[1], :cui)
            @test haskey(result[1], :similarity_score)
            @test 0.0 <= result[1][:similarity_score] <= 1.0
        end
    end

end
