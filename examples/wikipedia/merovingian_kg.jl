"""
Knowledge Graph Extraction from Wikipedia Data: Merovingian Kings

This example demonstrates knowledge graph extraction from simulated Wikipedia data
about Merovingian kings - the ruling dynasty of the Franks from c. 481–751 AD.
It shows how to:

1. Process multiple historical text sources (Wikipedia pages)
2. Extract knowledge graphs from each source
3. Merge knowledge graphs from multiple sources
4. Export the combined knowledge graph to JSON

The example uses historically accurate placeholder Wikipedia data about key Merovingian
kings including Clovis I (founder), Chlothar I (unifier), Dagobert I (last powerful king),
Theudebert I (first to mint coins), and Sigebert I (patron of learning).
"""

using GraphMERT

"""
    fetch_wikipedia_page(title::String)

Mock function to simulate fetching Wikipedia page content.
In a real implementation, this would use an HTTP client to fetch actual Wikipedia data.

# Arguments
- `title::String`: Wikipedia page title

# Returns
- `String`: Simulated page content
"""
function fetch_wikipedia_page(title::String)
    # Mock Wikipedia content for Merovingian kings (based on historical facts)
    mock_pages = Dict(
        "Clovis I" => """
        Clovis I (c. 466 – 511) was the first king of the Franks to unite all the Frankish tribes under one ruler.
        He founded the Merovingian dynasty and converted to Christianity, becoming the first Catholic king of the Franks.
        Clovis I defeated the last Roman ruler in Gaul, Syagrius, at the Battle of Soissons in 486.
        He went on to defeat the Alemanni at Tolbiac (496), conquered the Visigothic kingdom of Toulouse (507),
        and annexed Burgundy (534). His unification created the largest and most powerful state in Western Europe
        following the fall of the Western Roman Empire.
        """,
        "Chlothar I" => """
        Chlothar I (c. 497 – 561), also known as Clothar I or Lothar I, was one of the four sons of Clovis I.
        He became king of the Franks after reuniting the kingdom in 558 following the deaths of his brothers.
        Chlothar I conquered the Thuringians in 531 and extended Frankish influence into Saxony.
        He was known for his military campaigns, administrative reforms, and was the last Merovingian king
        to rule over a united Frankish kingdom until his death.
        """,
        "Dagobert I" => """
        Dagobert I (c. 603 – 639) was the king of Austrasia (623–634), king of all the Franks (629–634),
        and king of Neustria and Burgundy (629–639). He was the last king of the unified Merovingian kingdom
        and is considered the last truly powerful Merovingian monarch. Dagobert I moved his capital to Paris,
        reformed the legal system, and was known for his patronage of arts, culture, and the Church.
        He founded several monasteries and promoted Christianity throughout his realm. After his death,
        the kingdom entered a period of decline with increasingly ceremonial kings.
        """,
        "Theudebert I" => """
        Theudebert I (c. 500 – 547/548) was the king of Austrasia from 533 to 548.
        He was the eldest son of Theuderic I and a grandson of Clovis I. Theudebert I extended Frankish
        influence into northern Italy and was the first Frankish king to issue his own coinage.
        He maintained diplomatic relations with the Byzantine Empire and expanded Frankish territory
        through military campaigns against the Gepids and Lombards.
        """,
        "Sigebert I" => """
        Sigebert I (c. 535 – c. 575) was the king of Austrasia from 561 to 575.
        He was the son of Chlothar I and one of the four kings who divided the Frankish kingdom after their father's death.
        Sigebert I married Brunhilda of Visigothic royal blood, establishing an important alliance.
        He founded the city of Nancy and was known for his patronage of learning and the Church.
        He was assassinated during a civil war with his brother Chilperic I.
        """
    )

    return get(mock_pages, title, "Page not found: $title")
end

"""
    merge_graphs(graph1::KnowledgeGraph, graph2::KnowledgeGraph)

Merge two knowledge graphs by combining their entities and relations.

# Arguments
- `graph1::KnowledgeGraph`: First knowledge graph
- `graph2::KnowledgeGraph`: Second knowledge graph

# Returns
- `KnowledgeGraph`: Merged knowledge graph
"""
function merge_graphs(graph1::KnowledgeGraph, graph2::KnowledgeGraph)
    # Combine entities (simple append, no deduplication for this example)
    merged_entities = vcat(graph1.entities, graph2.entities)

    # Combine relations
    merged_relations = vcat(graph1.relations, graph2.relations)

    # Merge metadata
    merged_metadata = merge(graph1.metadata, graph2.metadata)
    merged_metadata["total_entities"] = length(merged_entities)
    merged_metadata["total_relations"] = length(merged_relations)
    merged_metadata["source"] = "merged_wikipedia_graphs"

    return KnowledgeGraph(merged_entities, merged_relations, merged_metadata)
end

println("=== Merovingian Kings Knowledge Graph Extraction ===")

# 1. Define list of Merovingian kings
kings = ["Clovis I", "Chlothar I", "Dagobert I", "Theudebert I", "Sigebert I"]

println("Processing knowledge graphs for $(length(kings)) Merovingian kings:")
println(join(kings, ", "))
println()

# 2. Initialize empty knowledge graph
global main_graph = KnowledgeGraph(
    Vector{KnowledgeEntity}(),
    Vector{KnowledgeRelation}(),
    Dict{String,Any}("source" => "wikipedia_merovingians", "total_entities" => 0, "total_relations" => 0)
)

# 3. Process each king's Wikipedia page
for king in kings
    println("Processing: $king")

    # Fetch simulated Wikipedia content
    page_content = fetch_wikipedia_page(king)
    println("  Page length: $(length(page_content)) characters")

    # Extract knowledge graph
    king_graph = extract_knowledge_graph(page_content)
    println("  Extracted: $(length(king_graph.entities)) entities, $(length(king_graph.relations)) relations")

    # Merge into main graph
    global main_graph = merge_graphs(main_graph, king_graph)

    println()
end

# 4. Display final statistics
println("=== Final Knowledge Graph Statistics ===")
println("Total entities: $(length(main_graph.entities))")
println("Total relations: $(length(main_graph.relations))")

if !isempty(main_graph.entities)
    println("\nEntities found:")
    for entity in main_graph.entities
        println("  - $(entity.text) ($(entity.label))")
    end
end

if !isempty(main_graph.relations)
    println("\nRelations found:")
    for relation in main_graph.relations
        println("  - $(relation.head) --[$(relation.relation_type)]--> $(relation.tail)")
    end
end

# 5. Export the knowledge graph
output_file = "merovingian_kings_graph.json"
export_to_json(main_graph, output_file)
println("\n✅ Knowledge graph exported to: $output_file")

println("\nNext steps:")
println("• Analyze royal succession patterns and territorial divisions")
println("• Build historical timeline of Frankish expansion")
println("• Cross-reference with Gregory of Tours' chronicles")
println("• Visualize the Merovingian family tree and territorial changes")
println("• Study the transition from Merovingian to Carolingian rule")
