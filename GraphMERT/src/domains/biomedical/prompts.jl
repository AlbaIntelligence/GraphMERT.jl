"""
Biomedical Domain Prompts for GraphMERT.jl

This module provides biomedical-specific LLM prompts for entity discovery,
relation matching, and tail entity formation.
"""

"""
    create_biomedical_prompt(task_type::Symbol, context::Dict)

Generate biomedical-specific LLM prompt for a given task type.

# Arguments
- `task_type::Symbol`: Task type (:entity_discovery, :relation_matching, :tail_formation)
- `context::Dict`: Context dictionary containing task-specific information

# Returns
- `String`: Formatted prompt for the LLM
"""
function create_biomedical_prompt(task_type::Symbol, context::Dict{String, Any})
    if task_type == :entity_discovery
        return create_entity_discovery_prompt(context)
    elseif task_type == :relation_matching
        return create_relation_matching_prompt(context)
    elseif task_type == :tail_formation
        return create_tail_formation_prompt(context)
    else
        error("Unknown task type: $task_type")
    end
end

"""
    create_entity_discovery_prompt(context::Dict)

Create prompt for biomedical entity discovery.

# Context keys:
- `text::String`: Input text to extract entities from
"""
function create_entity_discovery_prompt(context::Dict{String, Any})
    text = get(context, "text", "")
    
    prompt = """
You are a biomedical entity extraction expert. Extract all biomedical entities from the following text.

Focus on identifying:
- Diseases, disorders, syndromes, and medical conditions
- Drugs, medications, and therapeutic agents
- Proteins, enzymes, receptors, antibodies, and hormones
- Genes, genetic variants, mutations, and polymorphisms
- Anatomical structures, organs, tissues, and body parts
- Symptoms, signs, and clinical manifestations
- Medical procedures, surgeries, and treatments
- Microorganisms, bacteria, viruses, and pathogens
- Chemical compounds, molecules, and substances
- Cell types, cell lines, and cellular structures

Text:
$text

Instructions:
1. Extract all biomedical entities mentioned in the text
2. Return only the entity names, one per line
3. Use the exact text as it appears in the input
4. Do not include common words or non-medical terms
5. Focus on specific medical terminology and proper nouns

Return the entities:
"""
    return prompt
end

"""
    create_relation_matching_prompt(context::Dict)

Create prompt for biomedical relation matching.

# Context keys:
- `entities::Vector{String}`: List of entity names
- `text::String`: Original text for context
"""
function create_relation_matching_prompt(context::Dict{String, Any})
    entities = get(context, "entities", String[])
    text = get(context, "text", "")
    
    entity_list = join(entities, ", ")
    
    prompt = """
You are a biomedical relation extraction expert. Find relationships between biomedical entities in the following text.

Entities to consider:
$entity_list

Text:
$text

Focus on identifying these biomedical relation types:
- TREATS: A drug or treatment treats a disease or condition
- CAUSES: A disease, condition, or agent causes another condition
- ASSOCIATED_WITH: Entities are associated or related
- PREVENTS: A drug or treatment prevents a condition
- INHIBITS: A drug or substance inhibits a process or entity
- ACTIVATES: A drug or substance activates a process or entity
- BINDS_TO: A molecule binds to a target
- INTERACTS_WITH: Entities interact biologically
- REGULATES: An entity regulates another entity or process
- EXPRESSES: A gene or cell expresses a protein
- LOCATED_IN: An entity is located in an anatomical structure
- PART_OF: An entity is part of a larger structure
- INDICATES: A symptom or sign indicates a condition
- TARGETS: A drug targets a specific entity
- METABOLIZED_BY: A drug is metabolized by an enzyme or organ

Instructions:
1. Identify all relationships between the given entities
2. For each relationship, specify: Entity1, Relation, Entity2
3. Use the exact entity names as provided
4. Return relationships in the format: Entity1 | Relation | Entity2
5. One relationship per line
6. Only include relationships explicitly mentioned in the text

Return the relationships:
"""
    return prompt
end

"""
    create_tail_formation_prompt(context::Dict)

Create prompt for forming tail entities from predicted tokens.

# Context keys:
- `tokens::Vector{Tuple{Int, Float64}}`: Predicted tokens with probabilities
- `text::String`: Original text for context
"""
function create_tail_formation_prompt(context::Dict{String, Any})
    tokens = get(context, "tokens", Tuple{Int, Float64}[])
    text = get(context, "text", "")
    
    tokens_str = join(["(token: $t, prob: $(round(p, digits=3)))" for (t, p) in tokens], ", ")
    
    prompt = """
You are a biomedical entity formation expert. Form coherent biomedical entity names from predicted tokens.

Predicted tokens with probabilities:
$tokens_str

Original text context:
$text

Instructions:
1. Form a coherent biomedical entity name from the predicted tokens
2. The entity should be a valid biomedical term (disease, drug, protein, gene, etc.)
3. Use the token probabilities to guide your selection
4. Return only the entity name, nothing else
5. Ensure the entity name makes sense in the biomedical context
6. The entity should be consistent with the original text

Formed entity name:
"""
    return prompt
end

# Export
export create_biomedical_prompt
export create_entity_discovery_prompt, create_relation_matching_prompt, create_tail_formation_prompt
