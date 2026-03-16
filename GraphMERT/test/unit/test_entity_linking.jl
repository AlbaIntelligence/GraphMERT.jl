using Test
using GraphMERT
using LinearAlgebra
using Statistics

# Include the file to test
include(joinpath(@__DIR__, "../../src/domains/biomedical/entity_linking.jl"))

@testset "Entity Linker" begin
    @testset "Mock Entity Linker" begin
        linker = MockEntityLinker()
        
        # Test known mapping
        results = link_entity(linker, "diabetes")
        @test length(results) == 1
        @test results[1][1] == "C0011849"
        @test results[1][2] == 1.0
        
        # Test fallback
        results_fallback = link_entity(linker, "unknown_entity")
        @test length(results_fallback) == 1
        @test startswith(results_fallback[1][1], "C")
        @test results_fallback[1][2] == 0.8
    end

    @testset "SapBERT Linker Stub" begin
        linker = SapBERTLinker("path/to/model", "path/to/index")
        
        # Expect warning but valid return for now (stub behavior)
        results = link_entity(linker, "diabetes")
        @test length(results) == 1
        @test startswith(results[1][1], "C")
    end
end
