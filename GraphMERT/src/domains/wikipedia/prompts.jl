"""
Wikipedia Domain Prompts for GraphMERT.jl

This module provides Wikipedia-specific LLM prompts for entity discovery,
relation matching, and tail entity formation focused on general knowledge.
"""

"""
    create_wikipedia_prompt(task_type::Symbol, context::Dict)

Generate Wikipedia-specific LLM prompt for a given task type.

# Arguments
- `task_type::Symbol`: Task type (:entity_discovery, :relation_matching, :tail_formation)
- `context::Dict`: Context dictionary containing task-specific information

# Returns
- `String`: Formatted prompt for the LLM
"""
function create_wikipedia_prompt(task_type::Symbol, context::Dict{String, Any})
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

Create prompt for Wikipedia entity discovery.

# Context keys:
- `text::String`: Input text to extract entities from
"""
function create_entity_discovery_prompt(context::Dict{String, Any})
    text = get(context, "text", "")
    
    prompt = """
You are a general knowledge entity extraction expert. Extract all entities from the following text that would be found in Wikipedia.

Focus on identifying:
- People: Historical figures, scientists, artists, writers, politicians, etc.
- Organizations: Companies, institutions, universities, governments, etc.
- Locations: Cities, countries, landmarks, geographic features, etc.
- Concepts: Ideas, theories, principles, abstract concepts, etc.
- Events: Historical events, periods, movements, etc.
- Technologies: Systems, platforms, inventions, tools, etc.
- Artworks: Books, paintings, films, music, etc.
- Periods: Time periods, eras, epochs, etc.
- Theories: Scientific theories, philosophical concepts, etc.
- Methods: Techniques, approaches, methodologies, etc.
- Institutions: Universities, museums, libraries, etc.
- Countries: Nations, states, territories, etc.

Text:
$text

Instructions:
1. Extract all entities mentioned in the text
2. Return only the entity names, one per line
3. Use the exact text as it appears in the input
4. Do not include common words or non-entity terms
5. Focus on proper nouns, capitalized terms, and specific named entities
6. Include both full names and common abbreviations

Return the entities:
"""
    return prompt
end

"""
    create_relation_matching_prompt(context::Dict)

Create prompt for Wikipedia relation matching.

# Context keys:
- `entities::Vector{String}`: List of entity names
- `text::String`: Original text for context
"""
function create_relation_matching_prompt(context::Dict{String, Any})
    entities = get(context, "entities", String[])
    text = get(context, "text", "")
    
    entity_list = join(entities, ", ")
    
    prompt = """
You are a general knowledge relation extraction expert. Find relationships between entities in the following text.

Entities to consider:
$entity_list

Text:
$text

Focus on identifying these Wikipedia relation types:
- BORN_IN: A person was born in a location
- DIED_IN: A person died in a location
- WORKED_AT: A person worked at an organization
- FOUNDED: A person founded an organization
- LED: A person led an organization or movement
- INFLUENCED: A person or concept influenced another
- INVENTED: A person invented a technology or concept
- DEVELOPED: A person or organization developed something
- DISCOVERED: A person discovered something
- WROTE: A person wrote a book or work
- PAINTED: A person painted an artwork
- COMPOSED: A person composed music
- DIRECTED: A person directed a film or work
- ACTED_IN: A person acted in a production
- OCCURRED_IN: An event occurred in a location
- HAPPENED_DURING: An event happened during a period
- CREATED_BY: Something was created by a person
- RELATED_TO: Entities are related or connected
- SIMILAR_TO: Entities are similar

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
You are a general knowledge entity formation expert. Form coherent entity names from predicted tokens that would be found in Wikipedia.

Predicted tokens with probabilities:
$tokens_str

Original text context:
$text

Instructions:
1. Form a coherent entity name from the predicted tokens
2. The entity should be a valid general knowledge term (person, place, organization, concept, etc.)
3. Use the token probabilities to guide your selection
4. Return only the entity name, nothing else
5. Ensure the entity name makes sense in the general knowledge context
6. The entity should be consistent with the original text
7. Consider Wikipedia naming conventions (proper capitalization, etc.)

Formed entity name:
"""
    return prompt
end

# Export
export create_wikipedia_prompt
export create_entity_discovery_prompt, create_relation_matching_prompt, create_tail_formation_prompt
