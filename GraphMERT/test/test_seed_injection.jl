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

    @testset "UMLS Triple Retrieval" begin
        @testset "Basic Interface" begin
            cui = "C0011849"  # Mock diabetes mellitus CUI
            triples = get_umls_triples(cui)

            @test isa(triples, Vector{SemanticTriple})
            @test length(triples) >= 0  # May return empty for mock

            if !isempty(triples)
                triple = triples[1]
                @test triple.head_cui == cui || triple.tail_cui == cui
                @test !isempty(triple.head)
                @test !isempty(triple.tail)
                @test !isempty(triple.relation)
                @test 0.0 <= triple.score <= 1.0
                @test !isempty(triple.source)
            end
        end

        @testset "Edge Cases" begin
            @testset "Unknown CUI" begin
                triples = get_umls_triples("C999999")
                @test isempty(triples)
            end

            @testset "Empty CUI" begin
                triples = get_umls_triples("")
                @test isempty(triples)
            end

            @testset "Multiple Triples per CUI" begin
                triples = get_umls_triples("C0011849")  # Diabetes
                @test length(triples) >= 2  # Should have multiple triples

                # All triples should involve this CUI
                for triple in triples
                    @test triple.head_cui == "C0011849" || triple.tail_cui == "C0011849"
                end
            end

            @testset "Triple Field Validation" begin
                triples = get_umls_triples("C0011849")

                for triple in triples
                    @test isa(triple, SemanticTriple)
                    @test !isempty(triple.head)
                    @test !isempty(triple.tail)
                    @test !isempty(triple.relation)
                    @test !isempty(triple.tail_tokens)
                    @test 0.0 <= triple.score <= 1.0
                    @test triple.source == "mock_umls"
                end
            end
        end
    end

    @testset "Seed Injection Algorithm" begin
        @testset "Basic Interface" begin
            # Create mock triples
            triples = [
                SemanticTriple("diabetes mellitus", "C0011849", "treated_by", "metformin", [123, 456], 0.85, "mock_umls"),
                SemanticTriple("diabetes mellitus", "C0011849", "associated_with", "obesity", [789, 101], 0.72, "mock_umls"),
                SemanticTriple("hypertension", "C0020615", "treated_by", "lisinopril", [141, 159], 0.82, "mock_umls"),
            ]
            result = inject_seed_triples(triples)

            @test isa(result, Vector{SemanticTriple})
            @test length(result) <= length(unique([t.head_cui for t in triples if t.head_cui !== nothing]))
        end

        @testset "Score Thresholding" begin
            # Test threshold filtering
            triples = [
                SemanticTriple("entity1", "C001", "rel1", "tail1", [1], 0.8, "mock"),
                SemanticTriple("entity2", "C002", "rel2", "tail2", [2], 0.6, "mock"),  # Below threshold
            ]

            # With default threshold 0.7
            result = inject_seed_triples(triples)
            @test length(result) == 1
            @test result[1].score >= 0.7

            # With lower threshold
            result_low = inject_seed_triples(triples, 0.5)
            @test length(result_low) == 2
        end

        @testset "One Triple Per Entity" begin
            # Multiple triples for same CUI, should select highest scoring one
            triples = [
                SemanticTriple("diabetes", "C0011849", "treated_by", "drug1", [1], 0.8, "mock"),
                SemanticTriple("diabetes", "C0011849", "causes", "condition1", [2], 0.9, "mock"),  # Higher score
                SemanticTriple("diabetes", "C0011849", "associated_with", "condition2", [3], 0.75, "mock"),
                SemanticTriple("hypertension", "C0020615", "treated_by", "drug2", [4], 0.85, "mock"),
            ]

            result = inject_seed_triples(triples)
            @test length(result) == 2  # Two unique CUIs

            # Check that for C0011849, we got the highest scoring triple (0.9)
            cui1_triples = filter(t -> t.head_cui == "C0011849", result)
            @test length(cui1_triples) == 1
            @test cui1_triples[1].score == 0.9
        end

        @testset "Edge Cases" begin
            @testset "Empty Input" begin
                result = inject_seed_triples(SemanticTriple[])
                @test isempty(result)
            end

            @testset "No Triples Meet Threshold" begin
                triples = [
                    SemanticTriple("entity1", "C001", "rel1", "tail1", [1], 0.6, "mock"),
                    SemanticTriple("entity2", "C002", "rel2", "tail2", [2], 0.5, "mock"),
                ]
                result = inject_seed_triples(triples, 0.7)
                @test isempty(result)
            end

            @testset "Nothing CUI Handling" begin
                triples = [
                    SemanticTriple("entity1", nothing, "rel1", "tail1", [1], 0.8, "mock"),
                    SemanticTriple("entity2", "C002", "rel2", "tail2", [2], 0.8, "mock"),
                ]
                result = inject_seed_triples(triples)
                @test length(result) == 1  # Only the one with valid CUI
                @test result[1].head_cui == "C002"
            end
        end
    end

end
