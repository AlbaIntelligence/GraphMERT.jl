"""
Test suite for biomedical relation extraction and classification
==============================================================

This test suite provides comprehensive testing for the biomedical relation
extraction and classification functionality in GraphMERT.jl.
"""

using Test
using Pkg
Pkg.activate("../")

using GraphMERT
using Logging

# Configure logging for tests
Logging.configure(level=Logging.Warn)

@testset "Biomedical Relation Types" begin
    @testset "Relation Type Enumeration" begin
        # Test that all expected relation types exist
        expected_types = [
            TREATS, CAUSES, ASSOCIATED_WITH, PREVENTS, INHIBITS, ACTIVATES,
            BINDS_TO, INTERACTS_WITH, REGULATES, EXPRESSES, LOCATED_IN,
            PART_OF, DERIVED_FROM, SYNONYMOUS_WITH, CONTRAINDICATED_WITH,
            INDICATES, MANIFESTS_AS, ADMINISTERED_FOR, TARGETS, METABOLIZED_BY,
            TRANSPORTED_BY, SECRETED_BY, PRODUCED_BY, CONTAINS, COMPONENT_OF,
            UNKNOWN_RELATION
        ]
        
        for relation_type in expected_types
            @test relation_type isa BiomedicalRelationType
        end
        
        # Test that we can get all supported types
        supported_types = get_supported_relation_types()
        @test length(supported_types) == 25  # Excluding UNKNOWN_RELATION
        @test UNKNOWN_RELATION ∉ supported_types
    end
    
    @testset "Relation Type Parsing" begin
        # Test valid relation type parsing
        @test parse_relation_type("TREATS") == TREATS
        @test parse_relation_type("causes") == CAUSES
        @test parse_relation_type("Associated_With") == ASSOCIATED_WITH
        @test parse_relation_type("PREVENTS") == PREVENTS
        
        # Test invalid relation type parsing
        @test parse_relation_type("INVALID") == UNKNOWN_RELATION
        @test parse_relation_type("") == UNKNOWN_RELATION
        @test parse_relation_type("random_text") == UNKNOWN_RELATION
    end
    
    @testset "Relation Type Names" begin
        # Test getting relation type names
        @test get_relation_type_name(TREATS) == "TREATS"
        @test get_relation_type_name(CAUSES) == "CAUSES"
        @test get_relation_type_name(ASSOCIATED_WITH) == "ASSOCIATED_WITH"
        @test get_relation_type_name(UNKNOWN_RELATION) == "UNKNOWN_RELATION"
    end
    
    @testset "Relation Type Descriptions" begin
        # Test getting relation type descriptions
        descriptions = [
            get_relation_type_description(TREATS),
            get_relation_type_description(CAUSES),
            get_relation_type_description(ASSOCIATED_WITH),
            get_relation_type_description(PREVENTS)
        ]
        
        for desc in descriptions
            @test isa(desc, String)
            @test length(desc) > 0
        end
    end
end

@testset "Relation Classification" begin
    @testset "Rule-based Classification" begin
        # Test treatment relations
        @test classify_by_rules("aspirin", "headache", "aspirin treats headache") == TREATS
        @test classify_by_rules("insulin", "diabetes", "insulin treats diabetes") == TREATS
        
        # Test causal relations
        @test classify_by_rules("smoking", "lung cancer", "smoking causes lung cancer") == CAUSES
        @test classify_by_rules("virus", "disease", "virus causes disease") == CAUSES
        
        # Test association relations
        @test classify_by_rules("gene", "protein", "gene is associated with protein") == ASSOCIATED_WITH
        @test classify_by_rules("symptom", "disease", "symptom is related to disease") == ASSOCIATED_WITH
        
        # Test prevention relations
        @test classify_by_rules("vaccine", "disease", "vaccine prevents disease") == PREVENTS
        @test classify_by_rules("exercise", "obesity", "exercise prevents obesity") == PREVENTS
        
        # Test inhibition relations
        @test classify_by_rules("drug", "enzyme", "drug inhibits enzyme") == INHIBITS
        @test classify_by_rules("inhibitor", "protein", "inhibitor blocks protein") == INHIBITS
        
        # Test activation relations
        @test classify_by_rules("hormone", "receptor", "hormone activates receptor") == ACTIVATES
        @test classify_by_rules("activator", "gene", "activator stimulates gene") == ACTIVATES
        
        # Test binding relations
        @test classify_by_rules("ligand", "receptor", "ligand binds to receptor") == BINDS_TO
        @test classify_by_rules("drug", "protein", "drug interacts with protein") == BINDS_TO
        
        # Test regulation relations
        @test classify_by_rules("transcription factor", "gene", "transcription factor regulates gene") == REGULATES
        @test classify_by_rules("hormone", "metabolism", "hormone controls metabolism") == REGULATES
        
        # Test expression relations
        @test classify_by_rules("gene", "protein", "gene expresses protein") == EXPRESSES
        @test classify_by_rules("DNA", "RNA", "DNA produces RNA") == EXPRESSES
        
        # Test location relations
        @test classify_by_rules("organ", "body", "organ is located in body") == LOCATED_IN
        @test classify_by_rules("protein", "cell", "protein is found in cell") == LOCATED_IN
        
        # Test part-of relations
        @test classify_by_rules("heart", "circulatory system", "heart is part of circulatory system") == PART_OF
        @test classify_by_rules("cell", "tissue", "cell is component of tissue") == PART_OF
        
        # Test derivation relations
        @test classify_by_rules("protein", "gene", "protein is derived from gene") == DERIVED_FROM
        @test classify_by_rules("RNA", "DNA", "RNA originates from DNA") == DERIVED_FROM
        
        # Test unknown relations
        @test classify_by_rules("entity1", "entity2", "random text") == UNKNOWN_RELATION
        @test classify_by_rules("", "", "") == UNKNOWN_RELATION
    end
    
    @testset "Relation Classification with UMLS" begin
        # Test classification with no UMLS client (fallback)
        @test classify_relation("aspirin", "headache", "aspirin treats headache") == TREATS
        @test classify_relation("smoking", "lung cancer", "smoking causes lung cancer") == CAUSES
        
        # Test classification with UMLS client (mock)
        umls_client = nothing  # In real tests, this would be a mock client
        @test classify_relation("aspirin", "headache", "aspirin treats headache"; umls_client=umls_client) == TREATS
    end
end

@testset "Relation Validation" begin
    @testset "Basic Validation" begin
        # Test valid relations
        @test validate_biomedical_relation("aspirin", "headache", TREATS) == true
        @test validate_biomedical_relation("smoking", "lung cancer", CAUSES) == true
        @test validate_biomedical_relation("gene", "protein", ASSOCIATED_WITH) == true
        
        # Test invalid relations
        @test validate_biomedical_relation("", "headache", TREATS) == false
        @test validate_biomedical_relation("aspirin", "", TREATS) == false
        @test validate_biomedical_relation("aspirin", "aspirin", TREATS) == false  # Self-relation
    end
    
    @testset "Type-specific Validation" begin
        # Test treatment validation
        @test validate_treats_relation("aspirin drug", "headache disease") == true
        @test validate_treats_relation("random text", "headache disease") == false
        
        # Test causal validation
        @test validate_causes_relation("smoking factor", "lung cancer disease") == true
        @test validate_causes_relation("random text", "lung cancer disease") == false
        
        # Test association validation (always valid)
        @test validate_associated_relation("any", "entity") == true
        
        # Test prevention validation
        @test validate_prevents_relation("vaccine drug", "disease condition") == true
        @test validate_prevents_relation("random text", "disease condition") == false
        
        # Test inhibition validation
        @test validate_inhibits_relation("aspirin drug", "enzyme protein") == true
        @test validate_inhibits_relation("random text", "enzyme protein") == false
        
        # Test activation validation
        @test validate_activates_relation("hormone drug", "receptor protein") == true
        @test validate_activates_relation("random text", "receptor protein") == false
        
        # Test binding validation
        @test validate_binds_relation("ligand drug", "receptor protein") == true
        @test validate_binds_relation("random text", "receptor protein") == false
        
        # Test interaction validation (always valid)
        @test validate_interacts_relation("any", "entity") == true
        
        # Test regulation validation
        @test validate_regulates_relation("transcription factor gene", "target gene") == true
        @test validate_regulates_relation("random text", "target gene") == false
        
        # Test expression validation
        @test validate_expresses_relation("BRCA1 gene", "BRCA1 protein") == true
        @test validate_expresses_relation("random text", "BRCA1 protein") == false
        
        # Test location validation
        @test validate_located_relation("heart organ", "chest location") == true
        @test validate_located_relation("random text", "chest location") == false
        
        # Test part-of validation
        @test validate_part_of_relation("heart component", "circulatory system") == true
        @test validate_part_of_relation("random text", "circulatory system") == false
        
        # Test derivation validation
        @test validate_derived_relation("protein product", "gene source") == true
        @test validate_derived_relation("random text", "gene source") == false
        
        # Test synonymy validation (always valid)
        @test validate_synonymous_relation("any", "entity") == true
        
        # Test contraindication validation
        @test validate_contraindicated_relation("aspirin drug", "bleeding condition") == true
        @test validate_contraindicated_relation("random text", "bleeding condition") == false
        
        # Test indication validation
        @test validate_indicates_relation("fever symptom", "infection disease") == true
        @test validate_indicates_relation("random text", "infection disease") == false
        
        # Test manifestation validation
        @test validate_manifests_relation("diabetes disease", "thirst symptom") == true
        @test validate_manifests_relation("random text", "thirst symptom") == false
        
        # Test administration validation
        @test validate_administered_relation("insulin drug", "diabetes disease") == true
        @test validate_administered_relation("random text", "diabetes disease") == false
        
        # Test target validation
        @test validate_targets_relation("aspirin drug", "COX enzyme") == true
        @test validate_targets_relation("random text", "COX enzyme") == false
        
        # Test metabolism validation
        @test validate_metabolized_relation("aspirin drug", "CYP450 enzyme") == true
        @test validate_metabolized_relation("random text", "CYP450 enzyme") == false
        
        # Test transport validation
        @test validate_transported_relation("glucose molecule", "GLUT transporter") == true
        @test validate_transported_relation("random text", "GLUT transporter") == false
        
        # Test secretion validation
        @test validate_secreted_relation("insulin hormone", "pancreas gland") == true
        @test validate_secreted_relation("random text", "pancreas gland") == false
        
        # Test production validation
        @test validate_produced_relation("insulin protein", "beta cell") == true
        @test validate_produced_relation("random text", "beta cell") == false
        
        # Test containment validation
        @test validate_contains_relation("cell organelle", "protein molecule") == true
        @test validate_contains_relation("random text", "protein molecule") == false
        
        # Test component validation
        @test validate_component_relation("protein component", "cell structure") == true
        @test validate_component_relation("random text", "cell structure") == false
    end
end

@testset "Relation Confidence Scoring" begin
    @testset "Confidence Calculation" begin
        # Test confidence calculation
        @test 0.0 <= calculate_relation_confidence("aspirin", "headache", TREATS, "aspirin treats headache") <= 1.0
        @test 0.0 <= calculate_relation_confidence("smoking", "lung cancer", CAUSES, "smoking causes lung cancer") <= 1.0
        @test 0.0 <= calculate_relation_confidence("gene", "protein", ASSOCIATED_WITH, "gene is associated with protein") <= 1.0
        
        # Test confidence for invalid relations
        @test calculate_relation_confidence("invalid", "headache", TREATS, "") == 0.0
        @test calculate_relation_confidence("aspirin", "invalid", TREATS, "") == 0.0
    end
end

@testset "UMLS Relation Mapping" begin
    @testset "UMLS Relation Mapping" begin
        # Test mapping UMLS relations to our types
        @test map_umls_relation_to_type("treats") == TREATS
        @test map_umls_relation_to_type("therapeutic") == TREATS
        @test map_umls_relation_to_type("causes") == CAUSES
        @test map_umls_relation_to_type("etiology") == CAUSES
        @test map_umls_relation_to_type("associated") == ASSOCIATED_WITH
        @test map_umls_relation_to_type("related") == ASSOCIATED_WITH
        @test map_umls_relation_to_type("prevents") == PREVENTS
        @test map_umls_relation_to_type("prophylaxis") == PREVENTS
        @test map_umls_relation_to_type("inhibits") == INHIBITS
        @test map_umls_relation_to_type("suppresses") == INHIBITS
        @test map_umls_relation_to_type("activates") == ACTIVATES
        @test map_umls_relation_to_type("stimulates") == ACTIVATES
        @test map_umls_relation_to_type("binds") == BINDS_TO
        @test map_umls_relation_to_type("interacts") == BINDS_TO
        @test map_umls_relation_to_type("regulates") == REGULATES
        @test map_umls_relation_to_type("modulates") == REGULATES
        @test map_umls_relation_to_type("expresses") == EXPRESSES
        @test map_umls_relation_to_type("produces") == EXPRESSES
        @test map_umls_relation_to_type("located") == LOCATED_IN
        @test map_umls_relation_to_type("found") == LOCATED_IN
        @test map_umls_relation_to_type("part") == PART_OF
        @test map_umls_relation_to_type("component") == PART_OF
        @test map_umls_relation_to_type("derived") == DERIVED_FROM
        @test map_umls_relation_to_type("originates") == DERIVED_FROM
        @test map_umls_relation_to_type("synonymous") == SYNONYMOUS_WITH
        @test map_umls_relation_to_type("equivalent") == SYNONYMOUS_WITH
        @test map_umls_relation_to_type("contraindicated") == CONTRAINDICATED_WITH
        @test map_umls_relation_to_type("contraindication") == CONTRAINDICATED_WITH
        @test map_umls_relation_to_type("indicates") == INDICATES
        @test map_umls_relation_to_type("suggests") == INDICATES
        @test map_umls_relation_to_type("manifests") == MANIFESTS_AS
        @test map_umls_relation_to_type("presents") == MANIFESTS_AS
        @test map_umls_relation_to_type("administered") == ADMINISTERED_FOR
        @test map_umls_relation_to_type("used") == ADMINISTERED_FOR
        @test map_umls_relation_to_type("targets") == TARGETS
        @test map_umls_relation_to_type("acts") == TARGETS
        @test map_umls_relation_to_type("metabolized") == METABOLIZED_BY
        @test map_umls_relation_to_type("metabolism") == METABOLIZED_BY
        @test map_umls_relation_to_type("transported") == TRANSPORTED_BY
        @test map_umls_relation_to_type("transport") == TRANSPORTED_BY
        @test map_umls_relation_to_type("secreted") == SECRETED_BY
        @test map_umls_relation_to_type("secretion") == SECRETED_BY
        @test map_umls_relation_to_type("produced") == PRODUCED_BY
        @test map_umls_relation_to_type("production") == PRODUCED_BY
        @test map_umls_relation_to_type("contains") == CONTAINS
        @test map_umls_relation_to_type("includes") == CONTAINS
        @test map_umls_relation_to_type("component") == COMPONENT_OF
        @test map_umls_relation_to_type("constituent") == COMPONENT_OF
        
        # Test unknown relations
        @test map_umls_relation_to_type("unknown") == UNKNOWN_RELATION
        @test map_umls_relation_to_type("") == UNKNOWN_RELATION
        @test map_umls_relation_to_type("random") == UNKNOWN_RELATION
    end
end

@testset "Edge Cases" begin
    @testset "Empty Input" begin
        @test classify_relation("", "", "") == UNKNOWN_RELATION
        @test validate_biomedical_relation("", "", TREATS) == false
    end
    
    @testset "Special Characters" begin
        special_text = "aspirin (ASA) treats headache (HA)."
        @test classify_relation("aspirin", "headache", special_text) == TREATS
    end
    
    @testset "Very Long Text" begin
        long_text = "aspirin treats headache. " * 100
        @test classify_relation("aspirin", "headache", long_text) == TREATS
    end
end

@testset "Performance" begin
    @testset "Classification Speed" begin
        sample_text = "aspirin treats headache. " * 10
        
        # Time the classification
        start_time = time()
        relation_type = classify_relation("aspirin", "headache", sample_text)
        end_time = time()
        
        classification_time = end_time - start_time
        
        @test classification_time < 0.1  # Should complete within 0.1 seconds
        @test relation_type == TREATS
    end
end

println("✅ All relation tests passed!")
