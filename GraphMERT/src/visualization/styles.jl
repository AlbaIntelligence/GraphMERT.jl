"""
Domain-specific styling for GraphMERT knowledge graph visualization.

Provides color schemes, shapes, and styling appropriate for different domains
(biomedical, Wikipedia, general).
"""

using Colors

# Biomedical domain styling
const BIOMEDICAL_ENTITY_COLORS = Dict(
    "DISEASE" => colorant"#FF6B6B",      # Red
    "DRUG" => colorant"#4ECDC4",         # Teal
    "GENE" => colorant"#45B7D1",         # Blue
    "PROTEIN" => colorant"#96CEB4",      # Green
    "CELL_TYPE" => colorant"#FECA57",    # Yellow
    "MOLECULAR_FUNCTION" => colorant"#FF9FF3",  # Pink
    "BIOLOGICAL_PROCESS" => colorant"#54A0FF",  # Light Blue
    "CELLULAR_COMPONENT" => colorant"#5F27CD",  # Purple
    "ANATOMICAL_STRUCTURE" => colorant"#00D2D3", # Cyan
    "CHEMICAL" => colorant"#FF9F43",      # Orange
)

const BIOMEDICAL_RELATION_COLORS = Dict(
    "TREATS" => colorant"#FF6B6B",       # Red
    "CAUSES" => colorant"#FF3838",       # Dark Red
    "INTERACTS_WITH" => colorant"#4ECDC4", # Teal
    "ASSOCIATED_WITH" => colorant"#A29BFE", # Light Purple
    "PART_OF" => colorant"#45B7D1",      # Blue
    "REGULATES" => colorant"#96CEB4",    # Green
    "LOCATED_IN" => colorant"#FECA57",   # Yellow
    "DERIVES_FROM" => colorant"#FF9FF3", # Pink
    "PRECEDES" => colorant"#54A0FF",     # Light Blue
)

# Wikipedia domain styling
const WIKIPEDIA_ENTITY_COLORS = Dict(
    "PERSON" => colorant"#74B9FF",       # Light Blue
    "ORGANIZATION" => colorant"#A29BFE", # Light Purple
    "LOCATION" => colorant"#FD79A8",     # Pink
    "EVENT" => colorant"#FDCB6E",        # Yellow
    "WORK_OF_ART" => colorant"#E17055", # Orange
    "PRODUCT" => colorant"#00B894",     # Green
    "CONCEPT" => colorant"#6C5CE7",     # Purple
    "DATE" => colorant"#E84393",        # Magenta
    "NUMBER" => colorant"#00CEC9",      # Teal
    "MISC" => colorant"#636E72",        # Gray
)

const WIKIPEDIA_RELATION_COLORS = Dict(
    "BORN_IN" => colorant"#74B9FF",      # Light Blue
    "WORKS_FOR" => colorant"#A29BFE",    # Light Purple
    "LOCATED_IN" => colorant"#FD79A8",   # Pink
    "PART_OF" => colorant"#FDCB6E",      # Yellow
    "CREATED" => colorant"#E17055",     # Orange
    "BELONGS_TO" => colorant"#00B894",  # Green
    "RELATED_TO" => colorant"#6C5CE7",  # Purple
    "HAPPENED_ON" => colorant"#E84393", # Magenta
    "HAS_PROPERTY" => colorant"#00CEC9", # Teal
)

# General domain fallback styling
const GENERAL_ENTITY_COLORS = Dict(
    "DEFAULT" => colorant"#B8D4E3",      # Light Blue-Gray
)

const GENERAL_RELATION_COLORS = Dict(
    "DEFAULT" => colorant"#95A5A6",      # Gray
)

"""
    get_entity_color(entity_type::String, domain::String="general")

Get the appropriate color for an entity type in the given domain.

# Arguments
- `entity_type::String`: The entity type (e.g., "DISEASE", "PERSON")
- `domain::String`: Domain identifier ("biomedical", "wikipedia", "general")

# Returns
- `Color`: RGB color for the entity type
"""
function get_entity_color(entity_type::String, domain::String="general")
    if domain == "biomedical"
        return get(BIOMEDICAL_ENTITY_COLORS, entity_type, BIOMEDICAL_ENTITY_COLORS["DISEASE"])
    elseif domain == "wikipedia"
        return get(WIKIPEDIA_ENTITY_COLORS, entity_type, WIKIPEDIA_ENTITY_COLORS["MISC"])
    else
        return GENERAL_ENTITY_COLORS["DEFAULT"]
    end
end

"""
    get_relation_color(relation_type::String, domain::String="general")

Get the appropriate color for a relation type in the given domain.

# Arguments
- `relation_type::String`: The relation type (e.g., "TREATS", "WORKS_FOR")
- `domain::String`: Domain identifier ("biomedical", "wikipedia", "general")

# Returns
- `Color`: RGB color for the relation type
"""
function get_relation_color(relation_type::String, domain::String="general")
    if domain == "biomedical"
        return get(BIOMEDICAL_RELATION_COLORS, relation_type, BIOMEDICAL_RELATION_COLORS["ASSOCIATED_WITH"])
    elseif domain == "wikipedia"
        return get(WIKIPEDIA_RELATION_COLORS, relation_type, WIKIPEDIA_RELATION_COLORS["RELATED_TO"])
    else
        return GENERAL_RELATION_COLORS["DEFAULT"]
    end
end

"""
    get_domain_color_palette(domain::String, element_type::Symbol)

Get a complete color palette for a domain and element type.

# Arguments
- `domain::String`: Domain identifier
- `element_type::Symbol`: :entity or :relation

# Returns
- `Dict{String, Color}`: Color mapping for the domain
"""
function get_domain_color_palette(domain::String, element_type::Symbol)
    if element_type == :entity
        if domain == "biomedical"
            return copy(BIOMEDICAL_ENTITY_COLORS)
        elseif domain == "wikipedia"
            return copy(WIKIPEDIA_ENTITY_COLORS)
        else
            return copy(GENERAL_ENTITY_COLORS)
        end
    elseif element_type == :relation
        if domain == "biomedical"
            return copy(BIOMEDICAL_RELATION_COLORS)
        elseif domain == "wikipedia"
            return copy(WIKIPEDIA_RELATION_COLORS)
        else
            return copy(GENERAL_RELATION_COLORS)
        end
    else
        error("element_type must be :entity or :relation")
    end
end

"""
    create_color_legend(domain::String, element_type::Symbol)

Create a color legend for visualization.

# Arguments
- `domain::String`: Domain identifier
- `element_type::Symbol`: :entity or :relation

# Returns
- `Vector{Tuple{String, Color}}`: List of (label, color) pairs for legend
"""
function create_color_legend(domain::String, element_type::Symbol)
    palette = get_domain_color_palette(domain, element_type)
    return [(type_name, color) for (type_name, color) in palette]
end

"""
    apply_domain_styling!(mg::MetaGraph, domain::String)

Apply domain-specific styling metadata to a MetaGraph.

Adds color and style information to nodes and edges based on domain.

# Arguments
- `mg::MetaGraph`: MetaGraph to style (modified in-place)
- `domain::String`: Domain identifier
"""
function apply_domain_styling!(mg::MetaGraph, domain::String)
    # Add node styling
    for v in vertices(mg)
        node_props = props(mg, v)
        entity_type = get(node_props, "entity_type", "DEFAULT")
        color = get_entity_color(entity_type, domain)

        # Add styling properties
        node_props["color"] = color
        node_props["domain"] = domain

        set_props!(mg, v, node_props)
    end

    # Add edge styling
    for e in edges(mg)
        edge_props = props(mg, e)
        relation_type = get(edge_props, "relation_type", "DEFAULT")
        color = get_relation_color(relation_type, domain)

        # Add styling properties
        edge_props["color"] = color
        edge_props["domain"] = domain

        set_props!(mg, e, edge_props)
    end
end

"""
    get_styling_config(domain::String)

Get complete styling configuration for a domain.

# Arguments
- `domain::String`: Domain identifier

# Returns
- `Dict{String, Any}`: Styling configuration with colors, shapes, etc.
"""
function get_styling_config(domain::String)
    config = Dict{String, Any}()

    # Color palettes
    config["entity_colors"] = get_domain_color_palette(domain, :entity)
    config["relation_colors"] = get_domain_color_palette(domain, :relation)

    # Default styling parameters
    config["default_node_size"] = 20
    config["default_edge_width"] = 2
    config["node_border_color"] = colorant"white"
    config["node_border_width"] = 2

    # Domain-specific adjustments
    if domain == "biomedical"
        config["default_node_size"] = 25
        config["node_border_color"] = colorant"#2C3E50"
    elseif domain == "wikipedia"
        config["default_node_size"] = 22
        config["edge_width_multiplier"] = 1.2
    end

    return config
end
