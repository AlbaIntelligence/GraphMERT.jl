"""
Wikipedia Example 1: Entity Extraction from Wikipedia Pages
===========================================================

This example demonstrates entity extraction from Wikipedia pages to show
the performance of GraphMERT methods on non-biomedical text. We'll use
Wikipedia articles about various topics to extract entities and relations.

This demonstrates the generalizability of the GraphMERT approach beyond
biomedical domains.
"""

using Pkg
Pkg.activate("../../")

using GraphMERT
using Logging
using HTTP
using JSON

# Configure logging
Logging.configure(level=Logging.Info)

function main()
    println("="^60)
    println("GraphMERT Wikipedia Example 1: Entity Extraction")
    println("="^60)
    
    # Sample Wikipedia-style texts (simplified versions)
    wikipedia_texts = [
        """
        Artificial Intelligence is a branch of computer science that aims to 
        create machines capable of intelligent behavior. Machine learning is 
        a subset of AI that enables computers to learn without being explicitly 
        programmed. Deep learning uses neural networks with multiple layers 
        to process data and make predictions.
        """,
        """
        Climate change refers to long-term shifts in global temperatures and 
        weather patterns. Greenhouse gases such as carbon dioxide and methane 
        trap heat in the atmosphere, causing global warming. Renewable energy 
        sources like solar and wind power can help reduce greenhouse gas 
        emissions and mitigate climate change.
        """,
        """
        The Renaissance was a period of cultural, artistic, and intellectual 
        rebirth in Europe from the 14th to 17th centuries. Artists like 
        Leonardo da Vinci and Michelangelo created masterpieces that continue 
        to influence art today. The period also saw advances in science, 
        literature, and philosophy.
        """,
        """
        Quantum computing is a type of computation that uses quantum mechanical 
        phenomena such as superposition and entanglement. Unlike classical 
        computers that use bits, quantum computers use quantum bits or qubits. 
        This technology has the potential to solve certain problems much faster 
        than classical computers.
        """
    ]
    
    println("\n📄 Processing $(length(wikipedia_texts)) Wikipedia-style texts...")
    
    # Initialize clients (using fallback methods for Wikipedia)
    umls_client = nothing
    llm_client = nothing
    
    println("\n⚠️  Using fallback methods for Wikipedia text processing")
    
    # Process each text
    all_results = Vector{Dict{String,Any}}()
    
    for (i, text) in enumerate(wikipedia_texts)
        println("\n📖 Processing text $i/$(length(wikipedia_texts))...")
        println("Text preview: $(text[1:min(100, length(text)))...")
        
        # Extract entities using general entity types
        # For Wikipedia, we'll use a broader set of entity types
        entity_types = [
            DISEASE, DRUG, PROTEIN, GENE, ANATOMY, SYMPTOM, PROCEDURE, 
            ORGANISM, CHEMICAL, CELL_TYPE, MOLECULAR_FUNCTION, 
            BIOLOGICAL_PROCESS, CELLULAR_COMPONENT
        ]
        
        entities = extract_entities_from_text(text; entity_types=entity_types)
        
        # Also try to extract general entities (people, places, concepts)
        general_entities = extract_general_entities(text)
        
        # Combine results
        all_entities = vcat(entities, general_entities)
        
        # Extract relations
        relations = extract_wikipedia_relations(all_entities, text)
        
        # Store results
        result = Dict(
            "text_id" => i,
            "text_preview" => text[1:min(100, length(text))],
            "entities" => all_entities,
            "relations" => relations,
            "entity_count" => length(all_entities),
            "relation_count" => length(relations)
        )
        
        push!(all_results, result)
        
        # Display results for this text
        println("  Found $(length(all_entities)) entities and $(length(relations)) relations")
        
        # Show top entities
        if !isempty(all_entities)
            println("  Top entities:")
            for (j, (text_entity, entity_type, confidence)) in enumerate(all_entities[1:min(5, length(all_entities))])
                println("    $j. \"$text_entity\" -> $entity_type ($(round(confidence, digits=2)))")
            end
        end
    end
    
    # Aggregate results
    println("\n📊 Aggregated Results:")
    total_entities = sum(r["entity_count"] for r in all_results)
    total_relations = sum(r["relation_count"] for r in all_results)
    
    println("  Total entities across all texts: $total_entities")
    println("  Total relations across all texts: $total_relations")
    println("  Average entities per text: $(round(total_entities / length(wikipedia_texts), digits=1))")
    println("  Average relations per text: $(round(total_relations / length(wikipedia_texts), digits=1))")
    
    # Analyze entity types across all texts
    println("\n🏷️  Entity Type Distribution (All Texts):")
    all_entity_types = Dict{String, Int}()
    for result in all_results
        for (text_entity, entity_type, confidence) in result["entities"]
            type_name = get_entity_type_name(entity_type)
            all_entity_types[type_name] = get(all_entity_types, type_name, 0) + 1
        end
    end
    
    for (type_name, count) in sort(collect(all_entity_types), by=x->x[2], rev=true)
        println("  $type_name: $count")
    end
    
    # Analyze relation types
    println("\n🔗 Relation Type Distribution (All Texts):")
    all_relation_types = Dict{String, Int}()
    for result in all_results
        for relation in result["relations"]
            rel_type = relation["relation_type"]
            all_relation_types[rel_type] = get(all_relation_types, rel_type, 0) + 1
        end
    end
    
    for (rel_type, count) in sort(collect(all_relation_types), by=x->x[2], rev=true)
        println("  $rel_type: $count")
    end
    
    # Calculate confidence statistics
    all_confidences = Float64[]
    for result in all_results
        for (text_entity, entity_type, confidence) in result["entities"]
            push!(all_confidences, confidence)
        end
    end
    
    if !isempty(all_confidences)
        avg_confidence = mean(all_confidences)
        max_confidence = maximum(all_confidences)
        min_confidence = minimum(all_confidences)
        
        println("\n📊 Confidence Statistics:")
        println("  Average: $(round(avg_confidence, digits=3))")
        println("  Maximum: $(round(max_confidence, digits=3))")
        println("  Minimum: $(round(min_confidence, digits=3))")
    end
    
    # Show sample relations
    println("\n🔗 Sample Relations:")
    sample_relations = Vector{Dict{String,Any}}()
    for result in all_results
        append!(sample_relations, result["relations"])
    end
    
    for (i, relation) in enumerate(sample_relations[1:min(10, length(sample_relations))])
        println("  $i. $(relation["head_entity"]) --[$(relation["relation_type"])]--> $(relation["tail_entity"])")
    end
    
    println("\n" * "="^60)
    println("✅ Wikipedia Example 1 completed successfully!")
    println("Next: Run 02_wikipedia_knowledge_graph.jl")
    println("="^60)
end

# Function to extract general entities (people, places, concepts)
function extract_general_entities(text::String)
    entities = Vector{Tuple{String, BiomedicalEntityType, Float64}}()
    
    # Simple patterns for general entities
    patterns = [
        (r"\b[A-Z][a-z]+ [A-Z][a-z]+\b", "PERSON", 0.6),  # Names
        (r"\b[A-Z][a-z]+ (?:University|College|Institute|School)\b", "ORGANIZATION", 0.7),  # Institutions
        (r"\b[A-Z][a-z]+ (?:City|State|Country|Nation)\b", "LOCATION", 0.7),  # Places
        (r"\b[A-Z][a-z]+ (?:Theory|Method|Algorithm|Technique)\b", "CONCEPT", 0.6),  # Concepts
        (r"\b[A-Z][a-z]+ (?:War|Revolution|Movement|Period)\b", "EVENT", 0.6),  # Events
    ]
    
    for (pattern, entity_type_str, confidence) in patterns
        entity_type = parse_entity_type(entity_type_str)
        matches = eachmatch(pattern, text, overlap=false)
        for match in matches
            push!(entities, (match.match, entity_type, confidence))
        end
    end
    
    return entities
end

# Function to extract relations from Wikipedia text
function extract_wikipedia_relations(entities::Vector{Tuple{String, BiomedicalEntityType, Float64}}, text::String)
    relations = Vector{Dict{String,Any}}()
    
    if length(entities) < 2
        return relations
    end
    
    # Find relations between entities
    for i in 1:length(entities)
        for j in (i+1):length(entities)
            head_entity = entities[i][1]
            tail_entity = entities[j][1]
            
            # Classify relation using rule-based approach
            relation_type = classify_relation(head_entity, tail_entity, text)
            
            if relation_type != UNKNOWN_RELATION
                confidence = calculate_relation_confidence(head_entity, tail_entity, relation_type, text)
                
                if validate_biomedical_relation(head_entity, tail_entity, relation_type)
                    push!(relations, Dict(
                        "head_entity" => head_entity,
                        "tail_entity" => tail_entity,
                        "relation_type" => get_relation_type_name(relation_type),
                        "confidence" => confidence,
                        "context" => text
                    ))
                end
            end
        end
    end
    
    return relations
end

# Run the example
main()
