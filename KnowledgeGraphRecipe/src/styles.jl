function get_node_colors(kg::KnowledgeGraph, domain::Symbol)
    if domain == :biomedical
        return [get_biomedical_color(e.entity_type) for e in kg.entities]
    elseif domain == :wikipedia
        return [get_wikipedia_color(e.entity_type) for e in kg.entities]
    else
        return distinguishable_colors(length(kg.entities))
    end
end

function get_edge_colors(kg::KnowledgeGraph, domain::Symbol)
    if domain == :biomedical
        return [get_relation_color(r.relation_type) for r in kg.relations]
    else
        return colorant"gray"
    end
end

function get_biomedical_color(entity_type::String)
    colors = Dict(
        "DISEASE" => colorant"red",
        "DRUG" => colorant"blue",
        "GENE" => colorant"green",
        "PROTEIN" => colorant"purple"
    )
    get(colors, entity_type, colorant"gray")
end

function get_wikipedia_color(entity_type::String)
    colors = Dict(
        "PERSON" => colorant"blue",
        "LOCATION" => colorant"green",
        "ORGANIZATION" => colorant"orange"
    )
    get(colors, entity_type, colorant"gray")
end

function get_relation_color(relation_type::String)
    colors = Dict(
        "treats" => colorant"blue",
        "causes" => colorant"red",
        "interacts_with" => colorant"green"
    )
    get(colors, relation_type, colorant"gray")
end

function get_node_sizes(kg::KnowledgeGraph, method::Symbol)
    if method == :degree
        degrees = [length([r for r in kg.relations if r.head == e.text || r.tail == e.text])
                  for e in kg.entities]
        return [max(5, d * 2) for d in degrees]
    elseif method == :confidence
        return [max(5, e.confidence * 10) for e in kg.entities]
    else
        return fill(8, length(kg.entities))
    end
end

function get_edge_widths(kg::KnowledgeGraph, method::Symbol)
    if method == :confidence
        return [max(1, r.confidence * 3) for r in kg.relations]
    else
        return fill(1, length(kg.relations))
    end
end
