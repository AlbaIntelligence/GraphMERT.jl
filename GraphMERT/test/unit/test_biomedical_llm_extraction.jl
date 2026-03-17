using Test
using GraphMERT
using GraphMERT.GraphMERT: extract_biomedical_entities, Entity, ProcessingOptions, BiomedicalDomain, load_biomedical_domain, create_mock_llm_client, MockLLMClient, extract_entities, create_prompt

@testset "Biomedical LLM Extraction" begin
    # Setup domain
    domain = load_biomedical_domain()
    
    # Sample text
    text = "Patient has Diabetes Mellitus and takes Metformin for Hyperglycemia. Insulin levels are low."
    
    # 1. Create the expected prompt to set up the mock response
    context = Dict{String, Any}("text" => text, "task_type" => :entity_discovery)
    expected_prompt = create_prompt(domain, :entity_discovery, context)
    
    # 2. Mock LLM response with "Entity | Type" format
    mock_response = """
    Diabetes Mellitus | DISEASE
    Metformin | DRUG
    Insulin | PROTEIN
    Hyperglycemia | SYMPTOM
    """
    
    # 3. Create mock client
    responses = Dict(expected_prompt => mock_response)
    llm_client = create_mock_llm_client(responses)
    
    # Test 1: Extraction with LLM enabled
    config = ProcessingOptions(
        domain="biomedical",
        use_helper_llm=true,
        confidence_threshold=0.5
    )
    
    entities = extract_entities(domain, text, config, llm_client)
    
    @test length(entities) == 4
    
    # Check types and source
    types = Dict(e.text => e.entity_type for e in entities)
    sources = Dict(e.text => e.attributes["source"] for e in entities)
    
    @test types["Diabetes Mellitus"] == "DISEASE"
    @test sources["Diabetes Mellitus"] == "llm"
    
    @test types["Metformin"] == "DRUG"
    @test sources["Metformin"] == "llm"
    
    @test types["Insulin"] == "PROTEIN"
    @test sources["Insulin"] == "llm"
    
    @test types["Hyperglycemia"] == "SYMPTOM"
    @test sources["Hyperglycemia"] == "llm"
    
    # Test 2: Extraction with LLM disabled (regex fallback)
    config_no_llm = ProcessingOptions(
        domain="biomedical",
        use_helper_llm=false
    )
    
    # Note: Regex patterns might not catch "Diabetes Mellitus" exactly or assign different types.
    # "Diabetes" -> DISEASE (regex)
    # "Metformin" -> DRUG (regex)
    # "Insulin" -> DRUG? (regex might classify differently)
    # Just verify they are extracted and source is regex.
    
    entities_regex = extract_entities(domain, text, config_no_llm, llm_client)
    
    @test length(entities_regex) > 0
    @test any(e.text == "Diabetes" || e.text == "Diabetes Mellitus" for e in entities_regex)
    @test all(haskey(e.attributes, "source") && e.attributes["source"] == "regex" for e in entities_regex)
    
    # Test 3: LLM failure fallback (empty response triggers fallback)
    empty_responses = Dict(expected_prompt => "")
    empty_client = create_mock_llm_client(empty_responses)
    
    entities_fallback = extract_entities(domain, text, config, empty_client)
    
    @test length(entities_fallback) > 0
    @test all(e.attributes["source"] == "regex" for e in entities_fallback)
end
