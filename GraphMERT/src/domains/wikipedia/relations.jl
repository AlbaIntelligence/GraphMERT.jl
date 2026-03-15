"""
Wikipedia Relation Extraction for GraphMERT.jl

This module provides Wikipedia-specific relation extraction using entity-agnostic
patterns (aligned with reference_facts.jl relation types) and optional LLM-based
relation matching. See reports/RELATION_EXTRACTION_IMPROVEMENT_PROPOSAL.md.
"""

using Graphs

# Multi-word name capture: words (letters, digits, apostrophe) separated by spaces.
# Used to build regexes that resolve to entities via find_entity().
const _NAME = raw"([A-Za-z][A-Za-z0-9']*(?:\s+[A-Za-z0-9][A-Za-z0-9']*)*)"

# (regex, relation_type, confidence). Order matters: more specific patterns first.
const _RELATION_PATTERNS = [
    # Marriage / spouse (both directions)
    (Regex("$(_NAME)\\s+married\\s+$(_NAME)", "i"), "SPOUSE_OF", 0.95),
    (Regex("$(_NAME)\\s+was\\s+married\\s+to\\s+$(_NAME)", "i"), "SPOUSE_OF", 0.95),
    (Regex("marriage\\s+of\\s+$(_NAME)\\s+and\\s+$(_NAME)", "i"), "SPOUSE_OF", 0.9),
    (Regex("$(_NAME)\\s+and\\s+$(_NAME)\\s+married", "i"), "SPOUSE_OF", 0.9),
    # Parent / child
    (Regex("$(_NAME)\\s+(?:was\\s+)?(?:the\\s+)?(?:father|mother|parent)\\s+of\\s+$(_NAME)", "i"), "PARENT_OF", 0.95),
    (Regex("$(_NAME)\\s+(?:was\\s+)?(?:the\\s+)?(?:son|daughter|child)\\s+of\\s+$(_NAME)", "i"), "CHILD_OF", 0.95),
    (Regex("(?:father|mother|parent)\\s+of\\s+$(_NAME)\\s+was\\s+$(_NAME)", "i"), "PARENT_OF", 0.9),
    (Regex("$(_NAME)\\s+,\\s+(?:the\\s+)?(?:father|mother)\\s+of\\s+$(_NAME)", "i"), "PARENT_OF", 0.9),
    # Succession
    (Regex("$(_NAME)\\s+(?:reigned|ruled|succeeded)\\s+(?:after|following)\\s+$(_NAME)", "i"), "SUCCESSOR_OF", 0.95),
    (Regex("$(_NAME)\\s+succeeded\\s+$(_NAME)", "i"), "SUCCESSOR_OF", 0.95),
    (Regex("$(_NAME)\\s+was\\s+succeeded\\s+by\\s+$(_NAME)", "i"), "SUCCESSOR_OF", 0.95),
    (Regex("successor\\s+of\\s+$(_NAME)\\s+was\\s+$(_NAME)", "i"), "SUCCESSOR_OF", 0.9),
    (Regex("$(_NAME)\\s+(?:predecessor|preceded)\\s+$(_NAME)", "i"), "PREDECESSOR_OF", 0.9),
    (Regex("$(_NAME)\\s+preceded\\s+$(_NAME)", "i"), "PREDECESSOR_OF", 0.9),
    # Birth / death location
    (Regex("$(_NAME)\\s+(?:was\\s+)?born\\s+in\\s+$(_NAME)", "i"), "BORN_IN", 0.95),
    (Regex("$(_NAME)\\s+(?:was\\s+)?died\\s+in\\s+$(_NAME)", "i"), "DIED_IN", 0.95),
    (Regex("birth\\s+of\\s+$(_NAME)\\s+in\\s+$(_NAME)", "i"), "BORN_IN", 0.9),
    (Regex("$(_NAME)\\s+in\\s+$(_NAME)\\s+(?:at\\s+)?(?:birth|born)", "i"), "BORN_IN", 0.85),
    # Reign (entity reigned from/until - tail can be date or entity; we only resolve entity tail here)
    (Regex("$(_NAME)\\s+reigned\\s+from\\s+$(_NAME)", "i"), "REIGNED_FROM", 0.9),
    (Regex("$(_NAME)\\s+reigned\\s+until\\s+$(_NAME)", "i"), "REIGNED_UNTIL", 0.9),
    (Regex("$(_NAME)\\s+ruled\\s+(?:from\\s+)?$(_NAME)", "i"), "REIGNED", 0.85),
    # Dynasty / title
    (Regex("$(_NAME)\\s+(?:of\\s+)?(?:the\\s+)?(?:house|dynasty)\\s+of\\s+$(_NAME)", "i"), "MEMBER_OF_DYNASTY", 0.9),
    (Regex("$(_NAME)\\s+(?:was\\s+)?(?:a\\s+)?(?:king|queen|monarch)\\s+of\\s+$(_NAME)", "i"), "RULED", 0.9),
]

function _build_entity_lookup(entities::Vector{Entity})
    entity_lookup = Dict{String, Entity}()
    for ent in entities
        entity_lookup[lowercase(strip(ent.text))] = ent
        words = split(ent.text, " ")
        for i in 1:length(words)
            partial = strip(join(words[1:i], " "))
            isempty(partial) && continue
            entity_lookup[lowercase(partial)] = ent
        end
    end
    return entity_lookup
end

function _find_entity(name::String, entity_lookup::Dict{String, Entity})
    name_clean = strip(name)
    isempty(name_clean) && return nothing
    name_lower = lowercase(name_clean)
    if haskey(entity_lookup, name_lower)
        return entity_lookup[name_lower]
    end
    for (key, ent) in entity_lookup
        if startswith(key, name_lower) || occursin(name_lower, key) || startswith(name_lower, key)
            return ent
        end
    end
    return nothing
end

function _add_relation!(
    relations::Vector{Relation},
    head_ent::Entity,
    tail_ent::Entity,
    relation_type::String,
    confidence::Float64,
    evidence::String,
    domain_name::String,
    seen::Set{Tuple{String,String,String}},
)
    key = (head_ent.id, relation_type, tail_ent.id)
    key in seen && return
    push!(seen, key)
    push!(relations, Relation(
        head_ent.id,
        tail_ent.id,
        relation_type,
        confidence,
        domain_name,
        "",
        evidence,
        Dict{String, Any}("head_type" => head_ent.entity_type, "tail_type" => tail_ent.entity_type),
    ))
end

"""
    extract_wikipedia_relations(entities::Vector{Entity}, text::String, config::ProcessingOptions, domain::WikipediaDomain; llm_client=nothing)

Extract Wikipedia relations between entities using entity-agnostic patterns and optional LLM.

When `llm_client` is provided and `config.use_local` is true, also runs LLM-based relation
matching and merges results (deduplicated by head, relation_type, tail).
"""
function extract_wikipedia_relations(
    entities::Vector{Entity},
    text::String,
    config::ProcessingOptions,
    domain::WikipediaDomain;
    llm_client = nothing,
)
    relations = Vector{Relation}()
    if length(entities) < 2
        return relations
    end

    entity_lookup = _build_entity_lookup(entities)
    find_ent(name) = _find_entity(String(name), entity_lookup)
    seen = Set{Tuple{String,String,String}}()

    # Sentence-level pattern matching (entity-agnostic)
    sentences = split(text, r"[.!?]\s*")
    for sentence in sentences
        sentence = strip(sentence)
        isempty(sentence) && continue
        for (pattern, relation_type, conf) in _RELATION_PATTERNS
            for m in eachmatch(pattern, sentence)
                length(m.captures) >= 2 || continue
                head_ent = find_ent(m.captures[1])
                tail_ent = find_ent(m.captures[2])
                if head_ent !== nothing && tail_ent !== nothing && head_ent.id != tail_ent.id
                    _add_relation!(
                        relations,
                        head_ent,
                        tail_ent,
                        relation_type,
                        conf,
                        String(m.match),
                        "wikipedia",
                        seen,
                    )
                end
            end
        end
    end

    # Optional LLM-based relation matching
    if llm_client !== nothing && config.use_local
        llm_relations = _extract_relations_with_llm(entities, text, config, domain, llm_client, find_ent, seen)
        append!(relations, llm_relations)
    end

    # Optional Wikidata enrichment: merge relations from Wikidata for entities that have a QID
    if domain.wikidata_client !== nothing
        wikidata_rels = _enrich_relations_from_wikidata(entities, domain, find_ent, seen)
        append!(relations, wikidata_rels)
    end

    return relations
end

function _enrich_relations_from_wikidata(
    entities::Vector{Entity},
    domain::WikipediaDomain,
    find_ent,
    seen::Set{Tuple{String,String,String}},
)
    out = Relation[]
    for ent in entities
        qid = get(ent.attributes, "wikidata_qid", nothing)
        qid === nothing && continue
        qid = string(qid)
        isempty(qid) && continue
        resp = get_wikidata_relations(qid, domain.wikidata_client)
        if !resp.success || !haskey(resp.data, "result")
            continue
        end
        rel_list = get(get(resp.data, "result", Dict()), "relations", Any[])
        for r in rel_list
            isa(r, Dict) || continue
            prop = get(r, "property", "")
            target_qid = get(r, "targetQID", "")
            target_val = get(r, "targetValue", "")
            isempty(prop) && continue
            rel_type = map_wikidata_property_to_relation_type(prop)
            tail_ent = nothing
            if !isempty(target_val)
                tail_ent = find_ent(target_val)
            end
            if tail_ent === nothing && !isempty(target_qid)
                try
                    label = get_wikidata_label(target_qid, domain.wikidata_client)
                    tail_ent = find_ent(label)
                catch
                    tail_ent = nothing
                end
            end
            tail_ent === nothing && continue
            tail_ent.id == ent.id && continue
            _add_relation!(out, ent, tail_ent, rel_type, 0.85, "wikidata", "wikipedia", seen)
        end
    end
    return out
end

function _extract_relations_with_llm(
    entities::Vector{Entity},
    text::String,
    config::ProcessingOptions,
    domain::WikipediaDomain,
    llm_client,
    find_ent,
    seen::Set{Tuple{String,String,String}},
)
    out = Relation[]
    try
        context = Dict{String, Any}("entities" => [e.text for e in entities], "text" => text)
        prompt = GraphMERT.create_prompt(domain, :relation_matching, context)
        response = _call_llm(llm_client, prompt, config)
        isempty(response) && return out
        for line in split(response, '\n')
            line = strip(line)
            isempty(line) && continue
            # Format: Entity1 | Relation | Entity2
            parts = split(line, '|'; limit = 3)
            length(parts) >= 3 || continue
            e1 = strip(parts[1])
            rel_type = strip(parts[2])
            e2 = strip(parts[3])
            isempty(rel_type) && continue
            head_ent = find_ent(e1)
            tail_ent = find_ent(e2)
            head_ent === nothing && continue
            tail_ent === nothing && continue
            head_ent.id == tail_ent.id && continue
            # Normalize relation type to uppercase with underscores
            rel_norm = uppercase(replace(rel_type, r"\s+" => "_"))
            _add_relation!(out, head_ent, tail_ent, rel_norm, 0.75, line, "wikipedia", seen)
        end
    catch e
        @warn "Wikipedia LLM relation extraction failed: $e"
    end
    return out
end

function _call_llm(llm_client, prompt::String, config::ProcessingOptions)
    if isa(llm_client, GraphMERT.LocalLLM.LocalLLMClient)
        return GraphMERT.LocalLLM.generate(llm_client, prompt)
    end
    return ""
end

"""
    validate_wikipedia_relation(head::String, relation_type::String, tail::String, context::Dict)

Validate a Wikipedia relation.
"""
function validate_wikipedia_relation(head::String, relation_type::String, tail::String, context::Dict{String, Any} = Dict{String, Any}())
    isempty(head) && return false
    isempty(tail) && return false
    isempty(relation_type) && return false
    return true
end

"""
    calculate_wikipedia_relation_confidence(head::String, relation_type::String, tail::String, context::Dict)

Calculate confidence for a Wikipedia relation.
"""
function calculate_wikipedia_relation_confidence(head::String, relation_type::String, tail::String, context::Dict{String, Any} = Dict{String, Any}())
    return 0.85
end

# Export
export extract_wikipedia_relations
export validate_wikipedia_relation
export calculate_wikipedia_relation_confidence
