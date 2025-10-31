"""
Serialization API for GraphMERT.jl

This module provides functions for exporting knowledge graphs
in various formats including JSON, CSV, and RDF.
"""

using JSON3
using CSV
using Dates
# using DocStringExtensions  # Temporarily disabled

"""
    export_knowledge_graph(kg::KnowledgeGraph, format::String; filepath::String="") -> String

Export knowledge graph in the specified format.

Supported formats:
- "json": JSON format with entities and relations
- "csv": CSV format with separate files for entities and relations
- "rdf": RDF/XML format (basic implementation)
- "ttl": Turtle format (basic implementation)

"""
function export_knowledge_graph(kg::KnowledgeGraph, format::String; filepath::String="")
  if format == "json"
    return export_to_json(kg, filepath)
  elseif format == "csv"
    return export_to_csv(kg, filepath)
  elseif format == "rdf"
    return export_to_rdf(kg, filepath)
  elseif format == "ttl"
    return export_to_ttl(kg, filepath)
  else
    error("Unsupported format: $format. Supported formats: json, csv, rdf, ttl")
  end
end

"""
    export_to_json(kg::KnowledgeGraph, filepath::String="") -> String

Export knowledge graph to JSON format.

"""
function export_to_json(kg::KnowledgeGraph, filepath::String="")
  # Convert to JSON-serializable format
  json_data = Dict(
    "entities" => [
      Dict(
        "text" => e.text,
        "label" => e.label,
        "id" => e.id,
        "confidence" => e.confidence,
        "position" => Dict(
          "start" => e.position.start,
          "stop" => e.position.stop,
        ),
      ) for e in kg.entities
    ],
    "relations" => [
      Dict(
        "relation_type" => r.relation_type,
        "confidence" => r.confidence,
        "head" => Dict(
          "text" => r.head,
          "label" => "",
          "id" => "",
        ),
        "tail" => Dict(
          "text" => r.tail,
          "label" => "",
          "id" => "",
        ),
      ) for r in kg.relations
    ],
    "metadata" => kg.metadata,
    "timestamp" => string(kg.created_at),
  )

  json_string = JSON3.write(json_data, pretty=true)

  if !isempty(filepath)
    open(filepath, "w") do io
      write(io, json_string)
    end
    return "Exported to $filepath"
  end

  return json_string
end

"""
    export_to_csv(kg::KnowledgeGraph, filepath::String="") -> String

Export knowledge graph to CSV format with separate files for entities and relations.

"""
function export_to_csv(kg::KnowledgeGraph, filepath::String="")
  base_path = isempty(filepath) ? "kg_export" : filepath

  # Export entities
  entities_data = [
    (text=e.text, label=e.label, id=e.id, confidence=e.confidence,
      start_pos=e.position.start, stop_pos=e.position.stop)
    for e in kg.entities
  ]

  entities_file = base_path * "_entities.csv"
  CSV.write(entities_file, entities_data)

  # Export relations
  relations_data = [
    (relation_type=r.relation_type, confidence=r.confidence,
      head_text=r.head, head_label="", head_id="",
      tail_text=r.tail, tail_label="", tail_id="")
    for r in kg.relations
  ]

  relations_file = base_path * "_relations.csv"
  CSV.write(relations_file, relations_data)

  return "Exported entities to $entities_file and relations to $relations_file"
end

"""
    export_to_rdf(kg::KnowledgeGraph, filepath::String="") -> String

Export knowledge graph to RDF/XML format (basic implementation).

"""
function export_to_rdf(kg::KnowledgeGraph, filepath::String="")
  rdf_content = """<?xml version="1.0" encoding="UTF-8"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
         xmlns:graphmert="http://graphmert.jl/ontology#">
"""

  # Add entities as resources
  for (i, entity) in enumerate(kg.entities)
    entity_id = "entity_$i"
    rdf_content *= """
      <rdf:Description rdf:about="#$entity_id">
        <rdfs:label>$entity.text</rdfs:label>
        <graphmert:type>$entity.label</graphmert:type>
        <graphmert:confidence>$entity.confidence</graphmert:confidence>
        <graphmert:id>$entity.id</graphmert:id>
      </rdf:Description>
    """
  end

  # Add relations
  for (i, relation) in enumerate(kg.relations)
    relation_id = "relation_$i"
    head_id = "entity_$(findfirst(e -> e.text == relation.head, kg.entities))"
    tail_id = "entity_$(findfirst(e -> e.text == relation.tail, kg.entities))"

    rdf_content *= """
      <rdf:Description rdf:about="#$relation_id">
        <rdf:type rdf:resource="#$relation.relation_type"/>
        <graphmert:confidence>$relation.confidence</graphmert:confidence>
        <graphmert:head rdf:resource="#$head_id"/>
        <graphmert:tail rdf:resource="#$tail_id"/>
      </rdf:Description>
    """
  end

  rdf_content *= "</rdf:RDF>"

  if !isempty(filepath)
    open(filepath, "w") do io
      write(io, rdf_content)
    end
    return "Exported to $filepath"
  end

  return rdf_content
end

"""
    export_to_ttl(kg::KnowledgeGraph, filepath::String="") -> String

Export knowledge graph to Turtle format (basic implementation).

"""
function export_to_ttl(kg::KnowledgeGraph, filepath::String="")
  ttl_content = """@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix graphmert: <http://graphmert.jl/ontology#> .

"""

  # Add entities
  for (i, entity) in enumerate(kg.entities)
    entity_id = "entity_$i"
    ttl_content *= """graphmert:$entity_id rdf:type graphmert:Entity ;
    rdfs:label "$(entity.text)" ;
    graphmert:type "$(entity.label)" ;
    graphmert:confidence $(entity.confidence) ;
    graphmert:id "$(entity.id)" .

"""
  end

  # Add relations
  for (i, relation) in enumerate(kg.relations)
    relation_id = "relation_$i"
    head_id = "entity_$(findfirst(e -> e.text == relation.head, kg.entities))"
    tail_id = "entity_$(findfirst(e -> e.text == relation.tail, kg.entities))"

    ttl_content *= """graphmert:$relation_id rdf:type graphmert:$(relation.relation_type) ;
    graphmert:confidence $(relation.confidence) ;
    graphmert:head graphmert:$head_id ;
    graphmert:tail graphmert:$tail_id .

"""
  end

  if !isempty(filepath)
    open(filepath, "w") do io
      write(io, ttl_content)
    end
    return "Exported to $filepath"
  end

  return ttl_content
end
