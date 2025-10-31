"""
Test suite for BioMedBERT tokenizer
"""

using Test
using ..GraphMERT

@testset "BioMedTokenizer Tests" begin

    @testset "Tokenizer Configuration" begin
        config = BioMedTokenizerConfig()
        @test config.vocab_size == 30522
        @test config.max_sequence_length == 1024
        @test config.pad_token == "<pad>"
        @test config.unk_token == "<unk>"
        @test config.cls_token == "<cls>"
        @test config.sep_token == "<sep>"
        @test config.mask_token == "<mask>"
    end

    @testset "Tokenizer Creation" begin
        config = BioMedTokenizerConfig()
        tokenizer = BioMedTokenizer(config)

        @test tokenizer.config.vocab_size == 30522
        @test haskey(tokenizer.vocab.token_to_id, "<pad>")
        @test haskey(tokenizer.vocab.token_to_id, "<unk>")
        @test haskey(tokenizer.vocab.token_to_id, "<mask>")
    end

    @testset "Tokenization" begin
        tokenizer = BioMedTokenizer()

        text = "Diabetes mellitus is a disease"
        tokens = tokenize(tokenizer, text)

        @test isa(tokens, Vector{String})
        @test length(tokens) > 0
        @test all(token -> isa(token, String), tokens)
    end

    @testset "Token ID Conversion" begin
        tokenizer = BioMedTokenizer()

        tokens = ["diabetes", "mellitus", "is", "a", "disease"]
        token_ids = convert_tokens_to_ids(tokenizer, tokens)

        @test isa(token_ids, Vector{Int})
        @test length(token_ids) == length(tokens)
        @test all(id -> isa(id, Int), token_ids)

        # Convert back
        recovered_tokens = convert_ids_to_tokens(tokenizer, token_ids)
        @test length(recovered_tokens) == length(tokens)
    end

    @testset "Encoding" begin
        tokenizer = BioMedTokenizer()

        text = "Diabetes mellitus"
        token_ids = encode(tokenizer, text)

        @test isa(token_ids, Vector{Int})
        @test length(token_ids) > 0
        @test all(id -> isa(id, Int), token_ids)
    end

    @testset "Decoding" begin
        tokenizer = BioMedTokenizer()

        token_ids = [1, 2, 3]  # Some token IDs
        text = decode(tokenizer, token_ids)

        @test isa(text, String)
        @test length(text) > 0
    end

    @testset "Batch Encoding" begin
        tokenizer = BioMedTokenizer()

        texts = ["Diabetes mellitus", "Type 2 diabetes", "Insulin resistance"]
        batch = batch_encode(tokenizer, texts)

        @test isa(batch, Vector{Vector{Int}})
        @test length(batch) == length(texts)
        @test all(ids -> isa(ids, Vector{Int}), batch)
    end

    @testset "Attention Mask Creation" begin
        tokenizer = BioMedTokenizer()

        token_ids = [1, 2, 3, 0, 0]  # Last two are padding
        attention_mask = create_attention_mask(token_ids, 0)

        @test isa(attention_mask, Vector{Int})
        @test length(attention_mask) == length(token_ids)
        @test attention_mask[1:3] == [1, 1, 1]
        @test attention_mask[4:5] == [0, 0]
    end

    @testset "Position IDs Creation" begin
        seq_len = 5
        position_ids = create_position_ids(seq_len)

        @test isa(position_ids, Vector{Int})
        @test length(position_ids) == seq_len
        @test position_ids == [0, 1, 2, 3, 4]
    end

end
