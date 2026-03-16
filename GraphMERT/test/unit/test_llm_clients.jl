using Test
using GraphMERT
using HTTP
using JSON3

# Mock response struct for generic request_fn mocking
struct MockResponse
    status::Int
    body::Vector{UInt8}
end

@testset "LLM Client Types" begin
    
    @testset "MockLLMClient" begin
        # Create client with predefined responses
        responses = Dict("Hello" => "World")
        client = create_mock_llm_client(responses)
        
        # Test exact match
        resp = GraphMERT.make_llm_request(client, "Hello")
        @test resp.success
        @test resp.content == "World"
        
        # Test fallback
        resp = GraphMERT.make_llm_request(client, "Unknown prompt")
        @test resp.success
        @test resp.content == "Default mock response"
        
        # Test keywords
        resp = GraphMERT.make_llm_request(client, "Please Extract biomedical entities from this text")
        @test occursin("Diabetes", resp.content)
    end
    
    @testset "GeminiClient" begin
        # Mock request function
        function mock_gemini_request(url, headers, body; kwargs...)
            @test occursin("generativelanguage.googleapis.com", url)
            @test occursin("key=test_key", url)
            
            # Verify body format
            json_body = JSON3.read(String(body))
            @test haskey(json_body, "contents")
            prompt = json_body["contents"][1]["parts"][1]["text"]
            
            # Return canned response
            response_json = """
            {
              "candidates": [
                {
                  "content": {
                    "parts": [
                      {
                        "text": "Gemini response to: $prompt"
                      }
                    ]
                  }
                }
              ]
            }
            """
            return MockResponse(200, Vector{UInt8}(response_json))
        end
        
        client = create_gemini_client("test_key"; request_fn=mock_gemini_request)
        
        resp = GraphMERT.make_llm_request(client, "Hello Gemini")
        @test resp.success
        @test resp.content == "Gemini response to: Hello Gemini"
        
        # Test error handling (mock failure)
        fail_client = create_gemini_client("bad_key"; request_fn=(u,h,b;k...) -> MockResponse(500, Vector{UInt8}("{}")))
        resp = GraphMERT.make_llm_request(fail_client, "Hello")
        @test !resp.success
        @test resp.http_status == 500
    end
end
