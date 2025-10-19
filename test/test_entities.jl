"""
Test suite for biomedical entity extraction and classification
============================================================

This test suite provides comprehensive testing for the biomedical entity
extraction and classification functionality in GraphMERT.jl.
"""

using Test
using Pkg
Pkg.activate("../")

using GraphMERT
using Logging

# Configure logging for tests
Logging.configure(level=Logging.Warn)

@testset "Biomedical Entity Types" begin
    @testset "Entity Type Enumeration" begin
        # Test that all expected entity types exist
        expected_types = [
            DISEASE, DRUG, PROTEIN, GENE, ANATOMY, SYMPTOM, PROCEDURE,
            ORGANISM, CHEMICAL, CELL_TYPE, MOLECULAR_FUNCTION,
            BIOLOGICAL_PROCESS, CELLULAR_COMPONENT, UNKNOWN
        ]
        
        for entity_type in expected_types
            @test entity_type isa BiomedicalEntityType
        end
        
        # Test that we can get all supported types
        supported_types = get_supported_entity_types()
        @test length(supported_types) == 13  # Excluding UNKNOWN
        @test UNKNOWN ∉ supported_types
    end
    
    @testset "Entity Type Parsing" begin
        # Test valid entity type parsing
        @test parse_entity_type("DISEASE") == DISEASE
        @test parse_entity_type("drug") == DRUG
        @test parse_entity_type("Protein") == PROTEIN
        @test parse_entity_type("GENE") == GENE
        
        # Test invalid entity type parsing
        @test parse_entity_type("INVALID") == UNKNOWN
        @test parse_entity_type("") == UNKNOWN
        @test parse_entity_type("random_text") == UNKNOWN
    end
    
    @testset "Entity Type Names" begin
        # Test getting entity type names
        @test get_entity_type_name(DISEASE) == "DISEASE"
        @test get_entity_type_name(DRUG) == "DRUG"
        @test get_entity_type_name(PROTEIN) == "PROTEIN"
        @test get_entity_type_name(UNKNOWN) == "UNKNOWN"
    end
    
    @testset "Entity Type Descriptions" begin
        # Test getting entity type descriptions
        descriptions = [
            get_entity_type_description(DISEASE),
            get_entity_type_description(DRUG),
            get_entity_type_description(PROTEIN),
            get_entity_type_description(GENE)
        ]
        
        for desc in descriptions
            @test isa(desc, String)
            @test length(desc) > 0
        end
    end
end

@testset "Entity Classification" begin
    @testset "Rule-based Classification" begin
        # Test disease classification
        @test classify_by_rules("alzheimer's disease") == DISEASE
        @test classify_by_rules("diabetes mellitus") == DISEASE
        @test classify_by_rules("cancer") == DISEASE
        
        # Test drug classification
        @test classify_by_rules("aspirin") == DRUG
        @test classify_by_rules("insulin") == DRUG
        @test classify_by_rules("penicillin") == DRUG
        
        # Test protein classification
        @test classify_by_rules("insulin protein") == PROTEIN
        @test classify_by_rules("hemoglobin") == PROTEIN
        @test classify_by_rules("collagen") == PROTEIN
        
        # Test gene classification
        @test classify_by_rules("BRCA1 gene") == GENE
        @test classify_by_rules("p53") == GENE
        @test classify_by_rules("APOE") == GENE
        
        # Test anatomy classification
        @test classify_by_rules("heart muscle") == ANATOMY
        @test classify_by_rules("brain tissue") == ANATOMY
        @test classify_by_rules("liver organ") == ANATOMY
        
        # Test unknown classification
        @test classify_by_rules("random text") == UNKNOWN
        @test classify_by_rules("") == UNKNOWN
    end
    
    @testset "Entity Classification with UMLS" begin
        # Test classification with no UMLS client (fallback)
        @test classify_entity("alzheimer's disease") == DISEASE
        @test classify_entity("aspirin") == DRUG
        @test classify_entity("insulin protein") == PROTEIN
        
        # Test classification with UMLS client (mock)
        umls_client = nothing  # In real tests, this would be a mock client
        @test classify_entity("alzheimer's disease"; umls_client=umls_client) == DISEASE
    end
end

@testset "Entity Validation" begin
    @testset "Basic Validation" begin
        # Test valid entities
        @test validate_biomedical_entity("alzheimer's disease", DISEASE) == true
        @test validate_biomedical_entity("aspirin", DRUG) == true
        @test validate_biomedical_entity("insulin", PROTEIN) == true
        
        # Test invalid entities
        @test validate_biomedical_entity("", DISEASE) == false
        @test validate_biomedical_entity("a", DISEASE) == false  # Too short
        @test validate_biomedical_entity("random text", DISEASE) == false  # Wrong type
    end
    
    @testset "Length Validation" begin
        # Test length constraints
        @test validate_biomedical_entity("a"^200, DISEASE) == false  # Too long
        @test validate_biomedical_entity("ab", DISEASE) == true  # Valid length
        @test validate_biomedical_entity("a"^100, DISEASE) == true  # Valid length
    end
end

@testset "Entity Normalization" begin
    @testset "Text Normalization" begin
        # Test basic normalization
        @test normalize_entity_text("  alzheimer's disease  ") == "alzheimer's disease"
        @test normalize_entity_text("The Alzheimer's Disease") == "Alzheimer's Disease"
        @test normalize_entity_text("alzheimer's disease condition") == "alzheimer's disease"
        
        # Test case normalization
        @test normalize_entity_text("DNA") == "DNA"
        @test normalize_entity_text("RNA") == "RNA"
        @test normalize_entity_text("ATP") == "ATP"
    end
end

@testset "Entity Confidence Scoring" begin
    @testset "Confidence Calculation" begin
        # Test confidence calculation
        @test 0.0 <= calculate_entity_confidence("alzheimer's disease", DISEASE) <= 1.0
        @test 0.0 <= calculate_entity_confidence("aspirin", DRUG) <= 1.0
        @test 0.0 <= calculate_entity_confidence("insulin", PROTEIN) <= 1.0
        
        # Test confidence for invalid entities
        @test calculate_entity_confidence("invalid", DISEASE) == 0.0
        @test calculate_entity_confidence("", DISEASE) == 0.0
    end
end

@testset "Entity Extraction" begin
    @testset "Pattern-based Extraction" begin
        # Test extraction from sample text
        sample_text = """
        Alzheimer's disease is a neurodegenerative disorder. Aspirin is a 
        common drug used to treat pain. Insulin is a protein hormone.
        """
        
        entities = extract_entities_from_text(sample_text)
        
        @test length(entities) > 0
        
        # Check that we found expected entities
        entity_texts = [e[1] for e in entities]
        @test any(contains.(entity_texts, "Alzheimer"))
        @test any(contains.(entity_texts, "Aspirin"))
        @test any(contains.(entity_texts, "Insulin"))
        
        # Check that all entities have valid types and confidence
        for (text, entity_type, confidence) in entities
            @test entity_type isa BiomedicalEntityType
            @test 0.0 <= confidence <= 1.0
        end
    end
    
    @testset "Entity Type Filtering" begin
        sample_text = "Alzheimer's disease and aspirin drug"
        
        # Test extraction with specific entity types
        disease_entities = extract_entities_from_text(sample_text; entity_types=[DISEASE])
        drug_entities = extract_entities_from_text(sample_text; entity_types=[DRUG])
        
        @test all(e[2] == DISEASE for e in disease_entities)
        @test all(e[2] == DRUG for e in drug_entities)
    end
end

@testset "Entity Patterns" begin
    @testset "Pattern Generation" begin
        # Test pattern generation for different entity types
        disease_patterns = get_entity_patterns(DISEASE)
        drug_patterns = get_entity_patterns(DRUG)
        protein_patterns = get_entity_patterns(PROTEIN)
        
        @test length(disease_patterns) > 0
        @test length(drug_patterns) > 0
        @test length(protein_patterns) > 0
        
        # Test that patterns are valid regex
        for pattern in vcat(disease_patterns, drug_patterns, protein_patterns)
            @test isa(pattern, Regex)
        end
    end
end

@testset "Edge Cases" begin
    @testset "Empty Input" begin
        @test extract_entities_from_text("") == []
        @test extract_entities_from_text("   ") == []
        @test classify_entity("") == UNKNOWN
    end
    
    @testset "Special Characters" begin
        special_text = "Alzheimer's disease (AD) is a neurodegenerative disorder."
        entities = extract_entities_from_text(special_text)
        @test length(entities) > 0
    end
    
    @testset "Very Long Text" begin
        long_text = "Alzheimer's disease " * "is a neurodegenerative disorder. " * 100
        entities = extract_entities_from_text(long_text)
        @test length(entities) > 0
    end
end

@testset "Performance" begin
    @testset "Extraction Speed" begin
        sample_text = "Alzheimer's disease is a neurodegenerative disorder. " * 10
        
        # Time the extraction
        start_time = time()
        entities = extract_entities_from_text(sample_text)
        end_time = time()
        
        extraction_time = end_time - start_time
        
        @test extraction_time < 1.0  # Should complete within 1 second
        @test length(entities) > 0
    end
end

println("✅ All entity tests passed!")
