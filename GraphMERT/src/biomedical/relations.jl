"""
Biomedical relation types for GraphMERT.jl

This module defines biomedical relation types and their classification
as specified in the GraphMERT paper for biomedical knowledge graph construction.
"""

using Dates
using Random

# ============================================================================
# Biomedical Relation Types
# ============================================================================

"""
    BiomedicalRelationType

Enumeration of supported biomedical relation types.

"""
@enum BiomedicalRelationType begin
    # Biomedical relation types
    TREATS
    CAUSES
    ASSOCIATED_WITH
    PREVENTS
    INHIBITS
    ACTIVATES
    BINDS_TO
    INTERACTS_WITH
    REGULATES
    EXPRESSES
    LOCATED_IN
    PART_OF
    DERIVED_FROM
    SYNONYMOUS_WITH
    CONTRAINDICATED_WITH
    INDICATES
    MANIFESTS_AS
    ADMINISTERED_FOR
    TARGETS
    METABOLIZED_BY
    TRANSPORTED_BY
    SECRETED_BY
    PRODUCED_BY
    CONTAINS
    COMPONENT_OF
    # General knowledge relation types for Wikipedia
    CREATED_BY
    WORKED_AT
    BORN_IN
    DIED_IN
    FOUNDED
    LED
    INFLUENCED
    DEVELOPED
    INVENTED
    DISCOVERED
    WROTE
    PAINTED
    COMPOSED
    DIRECTED
    ACTED_IN
    OCCURRED_IN
    HAPPENED_DURING
    PART_OF_EVENT
    RELATED_TO
    SIMILAR_TO
    OPPOSITE_OF
    PRECEDED_BY
    FOLLOWED_BY
    UNKNOWN_RELATION
end

"""
    get_relation_type_name(relation_type::BiomedicalRelationType)

Get the string name of a relation type.
"""
function get_relation_type_name(relation_type::BiomedicalRelationType)
    return string(relation_type)
end

"""
    parse_relation_type(type_name::String)

Parse a string to a biomedical relation type.
"""
function parse_relation_type(type_name::String)
    type_name_upper = uppercase(type_name)

    if type_name_upper == "TREATS"
        return TREATS
    elseif type_name_upper == "CAUSES"
        return CAUSES
    elseif type_name_upper == "ASSOCIATED_WITH"
        return ASSOCIATED_WITH
    elseif type_name_upper == "PREVENTS"
        return PREVENTS
    elseif type_name_upper == "INHIBITS"
        return INHIBITS
    elseif type_name_upper == "ACTIVATES"
        return ACTIVATES
    elseif type_name_upper == "BINDS_TO"
        return BINDS_TO
    elseif type_name_upper == "INTERACTS_WITH"
        return INTERACTS_WITH
    elseif type_name_upper == "REGULATES"
        return REGULATES
    elseif type_name_upper == "EXPRESSES"
        return EXPRESSES
    elseif type_name_upper == "LOCATED_IN"
        return LOCATED_IN
    elseif type_name_upper == "PART_OF"
        return PART_OF
    elseif type_name_upper == "DERIVED_FROM"
        return DERIVED_FROM
    elseif type_name_upper == "SYNONYMOUS_WITH"
        return SYNONYMOUS_WITH
    elseif type_name_upper == "CONTRAINDICATED_WITH"
        return CONTRAINDICATED_WITH
    elseif type_name_upper == "INDICATES"
        return INDICATES
    elseif type_name_upper == "MANIFESTS_AS"
        return MANIFESTS_AS
    elseif type_name_upper == "ADMINISTERED_FOR"
        return ADMINISTERED_FOR
    elseif type_name_upper == "TARGETS"
        return TARGETS
    elseif type_name_upper == "METABOLIZED_BY"
        return METABOLIZED_BY
    elseif type_name_upper == "TRANSPORTED_BY"
        return TRANSPORTED_BY
    elseif type_name_upper == "SECRETED_BY"
        return SECRETED_BY
    elseif type_name_upper == "PRODUCED_BY"
        return PRODUCED_BY
    elseif type_name_upper == "CONTAINS"
        return CONTAINS
    elseif type_name_upper == "COMPONENT_OF"
        return COMPONENT_OF
    elseif type_name_upper == "CREATED_BY"
        return CREATED_BY
    elseif type_name_upper == "WORKED_AT"
        return WORKED_AT
    elseif type_name_upper == "BORN_IN"
        return BORN_IN
    elseif type_name_upper == "DIED_IN"
        return DIED_IN
    elseif type_name_upper == "FOUNDED"
        return FOUNDED
    elseif type_name_upper == "LED"
        return LED
    elseif type_name_upper == "INFLUENCED"
        return INFLUENCED
    elseif type_name_upper == "DEVELOPED"
        return DEVELOPED
    elseif type_name_upper == "INVENTED"
        return INVENTED
    elseif type_name_upper == "DISCOVERED"
        return DISCOVERED
    elseif type_name_upper == "WROTE"
        return WROTE
    elseif type_name_upper == "PAINTED"
        return PAINTED
    elseif type_name_upper == "COMPOSED"
        return COMPOSED
    elseif type_name_upper == "DIRECTED"
        return DIRECTED
    elseif type_name_upper == "ACTED_IN"
        return ACTED_IN
    elseif type_name_upper == "OCCURRED_IN"
        return OCCURRED_IN
    elseif type_name_upper == "HAPPENED_DURING"
        return HAPPENED_DURING
    elseif type_name_upper == "PART_OF_EVENT"
        return PART_OF_EVENT
    elseif type_name_upper == "RELATED_TO"
        return RELATED_TO
    elseif type_name_upper == "SIMILAR_TO"
        return SIMILAR_TO
    elseif type_name_upper == "OPPOSITE_OF"
        return OPPOSITE_OF
    elseif type_name_upper == "PRECEDED_BY"
        return PRECEDED_BY
    elseif type_name_upper == "FOLLOWED_BY"
        return FOLLOWED_BY
    else
        return UNKNOWN_RELATION
    end
end

# ============================================================================
# Relation Classification
# ============================================================================

"""
    classify_relation(head_entity::String, tail_entity::String, context::String=""; umls_client=nothing)

Classify a biomedical relation between two entities.
"""
function classify_relation(
    head_entity::String,
    tail_entity::String,
    context::String = "";
    umls_client = nothing,
)
    # Try UMLS classification first if client is available
    if umls_client !== nothing
        try
            relation_type = classify_by_umls(head_entity, tail_entity, umls_client)
            if relation_type !== UNKNOWN_RELATION
                return relation_type
            end
        catch e
            @warn "UMLS relation classification failed: $e"
        end
    end

    # Fallback to rule-based classification
    return classify_by_rules(head_entity, tail_entity, context)
end

"""
    classify_by_umls(head_entity::String, tail_entity::String, umls_client)

Classify relation using UMLS semantic relations.
"""
function classify_by_umls(head_entity::String, tail_entity::String, umls_client)
    # Get CUIs for both entities
    head_cui = get_entity_cui(umls_client, head_entity)
    tail_cui = get_entity_cui(umls_client, tail_entity)

    if head_cui === nothing || tail_cui === nothing
        return UNKNOWN_RELATION
    end

    # Get relations for head entity
    head_relations = get_relations(umls_client, head_cui)
    if !head_relations.success
        return UNKNOWN_RELATION
    end

    # Check if tail entity is in relations
    relations_data = get(head_relations.data, "result", Dict{String,Any}())
    relations = get(relations_data, "relations", Vector{Any}())

    for relation in relations
        target_cui = get(relation, "targetCUI", "")
        relation_name = get(relation, "relationName", "")

        if target_cui == tail_cui
            return map_umls_relation_to_type(relation_name)
        end
    end

    return UNKNOWN_RELATION
end

"""
    map_umls_relation_to_type(umls_relation::String)

Map UMLS relation name to our relation type.
"""
function map_umls_relation_to_type(umls_relation::String)
    relation_lower = lowercase(umls_relation)

    if occursin("treats", relation_lower) || occursin("therapeutic", relation_lower)
        return TREATS
    elseif occursin("causes", relation_lower) || occursin("etiology", relation_lower)
        return CAUSES
    elseif occursin("associated", relation_lower) || occursin("related", relation_lower)
        return ASSOCIATED_WITH
    elseif occursin("prevents", relation_lower) || occursin("prophylaxis", relation_lower)
        return PREVENTS
    elseif occursin("inhibits", relation_lower) || occursin("suppresses", relation_lower)
        return INHIBITS
    elseif occursin("activates", relation_lower) || occursin("stimulates", relation_lower)
        return ACTIVATES
    elseif occursin("binds", relation_lower) || occursin("interacts", relation_lower)
        return BINDS_TO
    elseif occursin("regulates", relation_lower) || occursin("modulates", relation_lower)
        return REGULATES
    elseif occursin("expresses", relation_lower) || occursin("produces", relation_lower)
        return EXPRESSES
    elseif occursin("located", relation_lower) || occursin("found", relation_lower)
        return LOCATED_IN
    elseif occursin("part", relation_lower) || occursin("component", relation_lower)
        return PART_OF
    elseif occursin("derived", relation_lower) || occursin("originates", relation_lower)
        return DERIVED_FROM
    elseif occursin("synonymous", relation_lower) || occursin("equivalent", relation_lower)
        return SYNONYMOUS_WITH
    elseif occursin("contraindicated", relation_lower) ||
           occursin("contraindication", relation_lower)
        return CONTRAINDICATED_WITH
    elseif occursin("indicates", relation_lower) || occursin("suggests", relation_lower)
        return INDICATES
    elseif occursin("manifests", relation_lower) || occursin("presents", relation_lower)
        return MANIFESTS_AS
    elseif occursin("administered", relation_lower) || occursin("used", relation_lower)
        return ADMINISTERED_FOR
    elseif occursin("targets", relation_lower) || occursin("acts", relation_lower)
        return TARGETS
    elseif occursin("metabolized", relation_lower) || occursin("metabolism", relation_lower)
        return METABOLIZED_BY
    elseif occursin("transported", relation_lower) || occursin("transport", relation_lower)
        return TRANSPORTED_BY
    elseif occursin("secreted", relation_lower) || occursin("secretion", relation_lower)
        return SECRETED_BY
    elseif occursin("produced", relation_lower) || occursin("production", relation_lower)
        return PRODUCED_BY
    elseif occursin("contains", relation_lower) || occursin("includes", relation_lower)
        return CONTAINS
    elseif occursin("component", relation_lower) || occursin("constituent", relation_lower)
        return COMPONENT_OF
    else
        return UNKNOWN_RELATION
    end
end

"""
    classify_by_rules(head_entity::String, tail_entity::String, context::String)

Classify relation using rule-based patterns.
"""
function classify_by_rules(head_entity::String, tail_entity::String, context::String)
    head_lower = lowercase(head_entity)
    tail_lower = lowercase(tail_entity)
    context_lower = lowercase(context)

    # Treatment relations
    if occursin(
        r"\b(treats?|therapy|therapeutic|medication|drug|medicine)\b",
        context_lower,
    )
        return TREATS
    end

    # Causal relations
    if occursin(r"\b(causes?|leads? to|results? in|induces?|triggers?)\b", context_lower)
        return CAUSES
    end

    # Association relations
    if occursin(
        r"\b(associated with|related to|linked to|correlated with)\b",
        context_lower,
    )
        return ASSOCIATED_WITH
    end

    # Prevention relations
    if occursin(r"\b(prevents?|protects? against|reduces? risk|avoids?)\b", context_lower)
        return PREVENTS
    end

    # Inhibition relations
    if occursin(r"\b(inhibits?|suppresses?|blocks?|reduces?)\b", context_lower)
        return INHIBITS
    end

    # Activation relations
    if occursin(r"\b(activates?|stimulates?|enhances?|promotes?)\b", context_lower)
        return ACTIVATES
    end

    # Binding relations
    if occursin(r"\b(binds? to|interacts? with|attaches? to|connects? to)\b", context_lower)
        return BINDS_TO
    end

    # Regulation relations
    if occursin(r"\b(regulates?|modulates?|controls?|influences?)\b", context_lower)
        return REGULATES
    end

    # Expression relations
    if occursin(r"\b(expresses?|produces?|synthesizes?|generates?)\b", context_lower)
        return EXPRESSES
    end

    # Location relations
    if occursin(r"\b(located in|found in|present in|exists in)\b", context_lower)
        return LOCATED_IN
    end

    # Part-of relations
    if occursin(r"\b(part of|component of|constituent of|element of)\b", context_lower)
        return PART_OF
    end

    # Derivation relations
    if occursin(r"\b(derived from|originates from|comes from|stems from)\b", context_lower)
        return DERIVED_FROM
    end

    # Synonymy relations
    if occursin(r"\b(synonymous with|equivalent to|same as|identical to)\b", context_lower)
        return SYNONYMOUS_WITH
    end

    # Contraindication relations
    if occursin(
        r"\b(contraindicated with|incompatible with|conflicts with|contradicts)\b",
        context_lower,
    )
        return CONTRAINDICATED_WITH
    end

    # Indication relations
    if occursin(r"\b(indicates?|suggests?|points to|signals?)\b", context_lower)
        return INDICATES
    end

    # Manifestation relations
    if occursin(r"\b(manifests as|presents as|appears as|shows as)\b", context_lower)
        return MANIFESTS_AS
    end

    # Administration relations
    if occursin(r"\b(administered for|used for|given for|prescribed for)\b", context_lower)
        return ADMINISTERED_FOR
    end

    # Target relations
    if occursin(r"\b(targets?|acts on|affects?|influences?)\b", context_lower)
        return TARGETS
    end

    # Metabolism relations
    if occursin(
        r"\b(metabolized by|broken down by|processed by|converted by)\b",
        context_lower,
    )
        return METABOLIZED_BY
    end

    # Transport relations
    if occursin(r"\b(transported by|carried by|moved by|transferred by)\b", context_lower)
        return TRANSPORTED_BY
    end

    # Secretion relations
    if occursin(r"\b(secreted by|released by|produced by|generated by)\b", context_lower)
        return SECRETED_BY
    end

    # Production relations
    if occursin(
        r"\b(produced by|manufactured by|synthesized by|created by)\b",
        context_lower,
    )
        return PRODUCED_BY
    end

    # Containment relations
    if occursin(r"\b(contains?|includes?|has|holds)\b", context_lower)
        return CONTAINS
    end

    # Component relations
    if occursin(r"\b(component of|constituent of|element of|part of)\b", context_lower)
        return COMPONENT_OF
    end

    return UNKNOWN_RELATION
end

# ============================================================================
# Relation Validation
# ============================================================================

"""
    validate_biomedical_relation(head_entity::String, tail_entity::String, relation_type::BiomedicalRelationType)

Validate that a relation is semantically valid.
"""
function validate_biomedical_relation(
    head_entity::String,
    tail_entity::String,
    relation_type::BiomedicalRelationType,
)
    if isempty(head_entity) || isempty(tail_entity)
        return false
    end

    if head_entity == tail_entity
        return false  # Self-relations are generally invalid
    end

    # Type-specific validation
    if relation_type == TREATS
        return validate_treats_relation(head_entity, tail_entity)
    elseif relation_type == CAUSES
        return validate_causes_relation(head_entity, tail_entity)
    elseif relation_type == ASSOCIATED_WITH
        return validate_associated_relation(head_entity, tail_entity)
    elseif relation_type == PREVENTS
        return validate_prevents_relation(head_entity, tail_entity)
    elseif relation_type == INHIBITS
        return validate_inhibits_relation(head_entity, tail_entity)
    elseif relation_type == ACTIVATES
        return validate_activates_relation(head_entity, tail_entity)
    elseif relation_type == BINDS_TO
        return validate_binds_relation(head_entity, tail_entity)
    elseif relation_type == INTERACTS_WITH
        return validate_interacts_relation(head_entity, tail_entity)
    elseif relation_type == REGULATES
        return validate_regulates_relation(head_entity, tail_entity)
    elseif relation_type == EXPRESSES
        return validate_expresses_relation(head_entity, tail_entity)
    elseif relation_type == LOCATED_IN
        return validate_located_relation(head_entity, tail_entity)
    elseif relation_type == PART_OF
        return validate_part_of_relation(head_entity, tail_entity)
    elseif relation_type == DERIVED_FROM
        return validate_derived_relation(head_entity, tail_entity)
    elseif relation_type == SYNONYMOUS_WITH
        return validate_synonymous_relation(head_entity, tail_entity)
    elseif relation_type == CONTRAINDICATED_WITH
        return validate_contraindicated_relation(head_entity, tail_entity)
    elseif relation_type == INDICATES
        return validate_indicates_relation(head_entity, tail_entity)
    elseif relation_type == MANIFESTS_AS
        return validate_manifests_relation(head_entity, tail_entity)
    elseif relation_type == ADMINISTERED_FOR
        return validate_administered_relation(head_entity, tail_entity)
    elseif relation_type == TARGETS
        return validate_targets_relation(head_entity, tail_entity)
    elseif relation_type == METABOLIZED_BY
        return validate_metabolized_relation(head_entity, tail_entity)
    elseif relation_type == TRANSPORTED_BY
        return validate_transported_relation(head_entity, tail_entity)
    elseif relation_type == SECRETED_BY
        return validate_secreted_relation(head_entity, tail_entity)
    elseif relation_type == PRODUCED_BY
        return validate_produced_relation(head_entity, tail_entity)
    elseif relation_type == CONTAINS
        return validate_contains_relation(head_entity, tail_entity)
    elseif relation_type == COMPONENT_OF
        return validate_component_relation(head_entity, tail_entity)
    else
        return true  # Unknown relations are always valid
    end
end

# ============================================================================
# Specific Relation Validators
# ============================================================================

function validate_treats_relation(head_entity::String, tail_entity::String)
    head_lower = lowercase(head_entity)
    tail_lower = lowercase(tail_entity)

    # Head should be a drug/treatment, tail should be a disease/condition
    return (
        occursin(r"\b(drug|medication|medicine|treatment|therapy)\b", head_lower) &&
        occursin(r"\b(disease|disorder|syndrome|condition|illness)\b", tail_lower)
    )
end

function validate_causes_relation(head_entity::String, tail_entity::String)
    head_lower = lowercase(head_entity)
    tail_lower = lowercase(tail_entity)

    # Head should be a cause, tail should be an effect
    return (
        occursin(r"\b(virus|bacteria|gene|mutation|factor|condition)\b", head_lower) &&
        occursin(r"\b(disease|symptom|disorder|syndrome|condition)\b", tail_lower)
    )
end

function validate_associated_relation(head_entity::String, tail_entity::String)
    # Association relations are generally valid between any entities
    return true
end

function validate_prevents_relation(head_entity::String, tail_entity::String)
    head_lower = lowercase(head_entity)
    tail_lower = lowercase(tail_entity)

    # Head should be a preventive measure, tail should be a condition
    return (
        occursin(r"\b(vaccine|drug|medication|treatment|prevention)\b", head_lower) &&
        occursin(r"\b(disease|disorder|syndrome|condition|illness)\b", tail_lower)
    )
end

function validate_inhibits_relation(head_entity::String, tail_entity::String)
    head_lower = lowercase(head_entity)
    tail_lower = lowercase(tail_entity)

    # Head should be an inhibitor, tail should be a target
    return (
        occursin(r"\b(drug|medication|protein|enzyme|inhibitor)\b", head_lower) &&
        occursin(r"\b(protein|enzyme|receptor|pathway|process)\b", tail_lower)
    )
end

function validate_activates_relation(head_entity::String, tail_entity::String)
    head_lower = lowercase(head_entity)
    tail_lower = lowercase(tail_entity)

    # Head should be an activator, tail should be a target
    return (
        occursin(r"\b(drug|medication|protein|enzyme|activator)\b", head_lower) &&
        occursin(r"\b(protein|enzyme|receptor|pathway|process)\b", tail_lower)
    )
end

function validate_binds_relation(head_entity::String, tail_entity::String)
    head_lower = lowercase(head_entity)
    tail_lower = lowercase(tail_entity)

    # Head should be a ligand, tail should be a receptor
    return (
        occursin(r"\b(drug|medication|protein|ligand|molecule)\b", head_lower) &&
        occursin(r"\b(receptor|protein|enzyme|target|binding site)\b", tail_lower)
    )
end

function validate_interacts_relation(head_entity::String, tail_entity::String)
    # Interaction relations are generally valid between any entities
    return true
end

function validate_regulates_relation(head_entity::String, tail_entity::String)
    head_lower = lowercase(head_entity)
    tail_lower = lowercase(tail_entity)

    # Head should be a regulator, tail should be a target
    return (
        occursin(r"\b(gene|protein|enzyme|hormone|factor)\b", head_lower) &&
        occursin(r"\b(gene|protein|enzyme|pathway|process)\b", tail_lower)
    )
end

function validate_expresses_relation(head_entity::String, tail_entity::String)
    head_lower = lowercase(head_entity)
    tail_lower = lowercase(tail_entity)

    # Head should be a gene, tail should be a protein
    return (
        occursin(r"\b(gene|genetic|allele|mutation)\b", head_lower) &&
        occursin(r"\b(protein|enzyme|receptor|hormone)\b", tail_lower)
    )
end

function validate_located_relation(head_entity::String, tail_entity::String)
    head_lower = lowercase(head_entity)
    tail_lower = lowercase(tail_entity)

    # Head should be a structure, tail should be a location
    return (
        occursin(r"\b(organ|tissue|cell|protein|molecule)\b", head_lower) &&
        occursin(r"\b(organ|tissue|cell|compartment|location)\b", tail_lower)
    )
end

function validate_part_of_relation(head_entity::String, tail_entity::String)
    head_lower = lowercase(head_entity)
    tail_lower = lowercase(tail_entity)

    # Head should be a component, tail should be a larger structure
    return (
        occursin(r"\b(component|part|element|subunit)\b", head_lower) &&
        occursin(r"\b(organ|tissue|cell|structure|system)\b", tail_lower)
    )
end

function validate_derived_relation(head_entity::String, tail_entity::String)
    head_lower = lowercase(head_entity)
    tail_lower = lowercase(tail_entity)

    # Head should be a product, tail should be a source
    return (
        occursin(r"\b(protein|molecule|compound|product)\b", head_lower) &&
        occursin(r"\b(gene|protein|precursor|source)\b", tail_lower)
    )
end

function validate_synonymous_relation(head_entity::String, tail_entity::String)
    # Synonymy relations are generally valid between any entities
    return true
end

function validate_contraindicated_relation(head_entity::String, tail_entity::String)
    head_lower = lowercase(head_entity)
    tail_lower = lowercase(tail_entity)

    # Head should be a drug, tail should be a condition
    return (
        occursin(r"\b(drug|medication|medicine|treatment)\b", head_lower) &&
        occursin(r"\b(disease|disorder|syndrome|condition|illness)\b", tail_lower)
    )
end

function validate_indicates_relation(head_entity::String, tail_entity::String)
    head_lower = lowercase(head_entity)
    tail_lower = lowercase(tail_entity)

    # Head should be a symptom/sign, tail should be a condition
    return (
        occursin(r"\b(symptom|sign|manifestation|marker)\b", head_lower) &&
        occursin(r"\b(disease|disorder|syndrome|condition|illness)\b", tail_lower)
    )
end

function validate_manifests_relation(head_entity::String, tail_entity::String)
    head_lower = lowercase(head_entity)
    tail_lower = lowercase(tail_entity)

    # Head should be a condition, tail should be a symptom
    return (
        occursin(r"\b(disease|disorder|syndrome|condition|illness)\b", head_lower) &&
        occursin(r"\b(symptom|sign|manifestation|presentation)\b", tail_lower)
    )
end

function validate_administered_relation(head_entity::String, tail_entity::String)
    head_lower = lowercase(head_entity)
    tail_lower = lowercase(tail_entity)

    # Head should be a drug, tail should be a condition
    return (
        occursin(r"\b(drug|medication|medicine|treatment)\b", head_lower) &&
        occursin(r"\b(disease|disorder|syndrome|condition|illness)\b", tail_lower)
    )
end

function validate_targets_relation(head_entity::String, tail_entity::String)
    head_lower = lowercase(head_entity)
    tail_lower = lowercase(tail_entity)

    # Head should be a drug, tail should be a target
    return (
        occursin(r"\b(drug|medication|medicine|treatment)\b", head_lower) &&
        occursin(r"\b(protein|enzyme|receptor|pathway|process)\b", tail_lower)
    )
end

function validate_metabolized_relation(head_entity::String, tail_entity::String)
    head_lower = lowercase(head_entity)
    tail_lower = lowercase(tail_entity)

    # Head should be a drug, tail should be an enzyme
    return (
        occursin(r"\b(drug|medication|medicine|compound)\b", head_lower) &&
        occursin(r"\b(enzyme|protein|cytochrome|metabolizing)\b", tail_lower)
    )
end

function validate_transported_relation(head_entity::String, tail_entity::String)
    head_lower = lowercase(head_entity)
    tail_lower = lowercase(tail_entity)

    # Head should be a molecule, tail should be a transporter
    return (
        occursin(r"\b(molecule|compound|drug|medication)\b", head_lower) &&
        occursin(r"\b(transporter|protein|carrier|pump)\b", tail_lower)
    )
end

function validate_secreted_relation(head_entity::String, tail_entity::String)
    head_lower = lowercase(head_entity)
    tail_lower = lowercase(tail_entity)

    # Head should be a hormone/protein, tail should be a gland/organ
    return (
        occursin(r"\b(hormone|protein|enzyme|molecule)\b", head_lower) &&
        occursin(r"\b(gland|organ|tissue|cell|secretory)\b", tail_lower)
    )
end

function validate_produced_relation(head_entity::String, tail_entity::String)
    head_lower = lowercase(head_entity)
    tail_lower = lowercase(tail_entity)

    # Head should be a product, tail should be a producer
    return (
        occursin(r"\b(protein|hormone|enzyme|molecule)\b", head_lower) &&
        occursin(r"\b(gene|cell|tissue|organ|producer)\b", tail_lower)
    )
end

function validate_contains_relation(head_entity::String, tail_entity::String)
    head_lower = lowercase(head_entity)
    tail_lower = lowercase(tail_entity)

    # Head should be a container, tail should be a component
    return (
        occursin(r"\b(organ|tissue|cell|compartment|structure)\b", head_lower) &&
        occursin(r"\b(component|part|element|molecule|protein)\b", tail_lower)
    )
end

function validate_component_relation(head_entity::String, tail_entity::String)
    head_lower = lowercase(head_entity)
    tail_lower = lowercase(tail_entity)

    # Head should be a component, tail should be a larger structure
    return (
        occursin(r"\b(component|part|element|subunit|molecule)\b", head_lower) &&
        occursin(r"\b(organ|tissue|cell|structure|system)\b", tail_lower)
    )
end

# ============================================================================
# Relation Confidence Scoring
# ============================================================================

"""
    calculate_relation_confidence(head_entity::String, tail_entity::String, relation_type::BiomedicalRelationType, context::String="")

Calculate confidence score for relation classification.
"""
function calculate_relation_confidence(
    head_entity::String,
    tail_entity::String,
    relation_type::BiomedicalRelationType,
    context::String = "",
)
    if !validate_biomedical_relation(head_entity, tail_entity, relation_type)
        return 0.0
    end

    # Base confidence
    confidence = 0.5

    # Context bonus
    if !isempty(context)
        confidence += 0.2
    end

    # Entity length bonus (optimal length range)
    head_length = length(head_entity)
    tail_length = length(tail_entity)

    if 3 <= head_length <= 50 && 3 <= tail_length <= 50
        confidence += 0.1
    end

    # Specificity bonus
    if occursin(r"\b(specific|precise|exact|definitive)\b", lowercase(context))
        confidence += 0.1
    end

    # Medical terminology bonus
    if occursin(r"\b(medical|clinical|biomedical|scientific)\b", lowercase(context))
        confidence += 0.1
    end

    # UMLS integration bonus (if available)
    # This would be added when UMLS client is available

    return min(confidence, 1.0)
end

# ============================================================================
# Utility Functions
# ============================================================================

"""
    get_relation_type_description(relation_type::BiomedicalRelationType)

Get a description of a relation type.
"""
function get_relation_type_description(relation_type::BiomedicalRelationType)
    descriptions = Dict{BiomedicalRelationType,String}(
        TREATS => "Treatment relationship between drugs and diseases",
        CAUSES => "Causal relationship between causes and effects",
        ASSOCIATED_WITH => "Association relationship between entities",
        PREVENTS => "Prevention relationship between preventive measures and conditions",
        INHIBITS => "Inhibition relationship between inhibitors and targets",
        ACTIVATES => "Activation relationship between activators and targets",
        BINDS_TO => "Binding relationship between ligands and receptors",
        INTERACTS_WITH => "Interaction relationship between entities",
        REGULATES => "Regulation relationship between regulators and targets",
        EXPRESSES => "Expression relationship between genes and proteins",
        LOCATED_IN => "Location relationship between structures and locations",
        PART_OF => "Part-of relationship between components and structures",
        DERIVED_FROM => "Derivation relationship between products and sources",
        SYNONYMOUS_WITH => "Synonymy relationship between equivalent entities",
        CONTRAINDICATED_WITH => "Contraindication relationship between drugs and conditions",
        INDICATES => "Indication relationship between symptoms and conditions",
        MANIFESTS_AS => "Manifestation relationship between conditions and symptoms",
        ADMINISTERED_FOR => "Administration relationship between drugs and conditions",
        TARGETS => "Target relationship between drugs and targets",
        METABOLIZED_BY => "Metabolism relationship between drugs and enzymes",
        TRANSPORTED_BY => "Transport relationship between molecules and transporters",
        SECRETED_BY => "Secretion relationship between hormones and glands",
        PRODUCED_BY => "Production relationship between products and producers",
        CONTAINS => "Containment relationship between containers and components",
        COMPONENT_OF => "Component relationship between components and structures",
        UNKNOWN_RELATION => "Unknown or unclassified relation type",
    )

    return get(descriptions, relation_type, "Unknown relation type")
end

"""
    get_supported_relation_types()

Get all supported biomedical relation types.
"""
function get_supported_relation_types()
    return [
        TREATS,
        CAUSES,
        ASSOCIATED_WITH,
        PREVENTS,
        INHIBITS,
        ACTIVATES,
        BINDS_TO,
        INTERACTS_WITH,
        REGULATES,
        EXPRESSES,
        LOCATED_IN,
        PART_OF,
        DERIVED_FROM,
        SYNONYMOUS_WITH,
        CONTRAINDICATED_WITH,
        INDICATES,
        MANIFESTS_AS,
        ADMINISTERED_FOR,
        TARGETS,
        METABOLIZED_BY,
        TRANSPORTED_BY,
        SECRETED_BY,
        PRODUCED_BY,
        CONTAINS,
        COMPONENT_OF,
    ]
end
