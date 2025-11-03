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

        @testset "Edge Cases" begin
            @testset "Empty Input" begin
                result = link_entities_sapbert(String[])
                @test isempty(result)
            end

            @testset "Single Entity" begin
                result = link_entities_sapbert(["cancer"])
                @test length(result) == 1
                @test result[1][:entity_text] == "cancer"
                @test haskey(result[1], :cui)
                @test haskey(result[1], :similarity_score)
            end

            @testset "Reproducibility" begin
                entities = ["diabetes", "hypertension"]
                result1 = link_entities_sapbert(entities)
                result2 = link_entities_sapbert(entities)

                # Results should be identical (same random seed)
                @test result1 == result2
            end

            @testset "Score Range Validation" begin
                entities = ["test entity 1", "test entity 2", "test entity 3"]
                result = link_entities_sapbert(entities)

                for item in result
                    @test 0.5 <= item[:similarity_score] <= 0.95  # Mock range
                end
            end
        end
    end

end
