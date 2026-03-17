using Test
using GraphMERT
using JLD2
using FileIO

@testset "SapBERT Index Persistence" begin
    # Setup temporary paths
    temp_dir = mktempdir()
    index_path = joinpath(temp_dir, "sapbert_index.jld2")
    
    # Create mock data for index
    embeddings = rand(Float32, 128, 5) # 5 entities, 128 dim
    cui_list = ["C001", "C002", "C003", "C004", "C005"]
    name_list = ["Entity 1", "Entity 2", "Entity 3", "Entity 4", "Entity 5"]
    
    linker = SapBERTLinker("mock_model_path", index_path;
                          embeddings=embeddings,
                          cui_list=cui_list,
                          name_list=name_list)
                          
    @testset "Save Index" begin
        save_index(linker, index_path)
        @test isfile(index_path)
    end
    
    @testset "Load Index" begin
        # Create a new linker without data
        new_linker = SapBERTLinker("mock_model_path", index_path)
        
        # It should have auto-loaded via constructor
        @test new_linker.embeddings !== nothing
        @test new_linker.cui_list !== nothing
        @test new_linker.name_list !== nothing
        
        @test size(new_linker.embeddings) == (128, 5)
        @test length(new_linker.cui_list) == 5
        @test new_linker.cui_list == cui_list
        @test new_linker.name_list == name_list
        @test new_linker.embeddings ≈ embeddings
    end
    
    @testset "Manual Load" begin
        empty_linker = SapBERTLinker("mock_model_path", "non_existent_path")
        load_index!(empty_linker, index_path)
        
        @test empty_linker.cui_list == cui_list
        @test empty_linker.embeddings ≈ embeddings
    end

    # Clean up
    rm(temp_dir, recursive=true)
end
