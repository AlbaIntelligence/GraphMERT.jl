"""
Reference facts dataset for French monarchy knowledge graph testing.

These facts are historically verified and used as ground truth for evaluating
the quality of extracted knowledge graphs.
"""

# Reference facts: (subject, predicate, object, verified)
# These represent known historical facts about French monarchy

const FRENCH_MONARCHY_FACTS = [
    # Louis XIV - facts about the Sun King
    ("Louis XIV", "reigned_from", "1643", true),
    ("Louis XIV", "reigned_until", "1715", true),
    ("Louis XIV", "dynasty", "Bourbon", true),
    ("Louis XIV", "title", "Sun King", true),
    ("Louis XIV", "parent_of", "Louis XV", true),
    ("Louis XIV", "spouse", "Maria Theresa of Spain", true),
    ("Louis XIV", "born_in", "Saint-Germain-en-Laye", true),
    ("Louis XIV", "died_in", "Versailles", true),
    ("Louis XIV", "predecessor", "Louis XIII", true),
    ("Louis XIV", "successor", "Louis XV", true),
    
    # Louis XIII - father of Louis XIV
    ("Louis XIII", "reigned_from", "1610", true),
    ("Louis XIII", "reigned_until", "1643", true),
    ("Louis XIII", "dynasty", "Bourbon", true),
    ("Louis XIII", "parent_of", "Louis XIV", true),
    ("Louis XIII", "spouse", "Anne of Austria", true),
    ("Louis XIII", "predecessor", "Henry IV", true),
    
    # Louis XV - great-grandson of Louis XIV
    ("Louis XV", "reigned_from", "1715", true),
    ("Louis XV", "reigned_until", "1774", true),
    ("Louis XV", "dynasty", "Bourbon", true),
    ("Louis XV", "parent_of", "Louis XVI", true),
    ("Louis XV", "grandparent_of", "Louis XVII", true),
    ("Louis XV", "predecessor", "Louis XIV", true),
    ("Louis XV", "successor", "Louis XVI", true),
    
    # Louis XVI - last absolute king
    ("Louis XVI", "reigned_from", "1774", true),
    ("Louis XVI", "reigned_until", "1792", true),
    ("Louis XVI", "dynasty", "Bourbon", true),
    ("Louis XVI", "parent_of", "Louis XVII", true),
    ("Louis XVI", "spouse", "Marie Antoinette", true),
    ("Louis XVI", "predecessor", "Louis XV", true),
    ("Louis XVI", "executed", "1793", true),
    
    # Louis XVII - died in prison
    ("Louis XVII", "reigned_from", "1793", true),
    ("Louis XVII", "reigned_until", "1795", true),
    ("Louis XVII", "dynasty", "Bourbon", true),
    ("Louis XVII", "parent", "Louis XVI", true),
    ("Louis XVII", "parent", "Marie Antoinette", true),
    
    # Henry IV - first Bourbon king
    ("Henry IV", "reigned_from", "1589", true),
    ("Henry IV", "reigned_until", "1610", true),
    ("Henry IV", "dynasty", "Bourbon", true),
    ("Henry IV", "title", "Henry the Great", true),
    ("Henry IV", "parent_of", "Louis XIII", true),
    ("Henry IV", "spouse", "Margaret of Valois", true),
    ("Henry IV", "spouse", "Marie de' Medici", true),
    ("Henry IV", "born_in", "Pau", true),
    ("Henry IV", "died_in", "Paris", true),
    ("Henry IV", "predecessor", "Henry III", true),
    
    # Louis XIII - continued
    ("Louis XIII", "spouse", "Anne of Austria", true),
    
    # Marie de' Medici - second wife of Henry IV
    ("Marie de' Medici", "spouse", "Henry IV", true),
    ("Marie de' Medici", "parent_of", "Louis XIII", true),
    
    # Marie Antoinette
    ("Marie Antoinette", "spouse", "Louis XVI", true),
    ("Marie Antoinette", "parent_of", "Louis XVII", true),
    ("Marie Antoinette", "born_in", "Vienna", true),
    ("Marie Antoinette", "executed", "1793", true),
    ("Marie Antoinette", "nationality", "Austrian", true),
    
    # French Revolution
    ("French Revolution", "started", "1789", true),
    ("French Revolution", "ended", "1799", true),
    
    # Key locations
    ("Versailles", "type", "palace", true),
    ("Paris", "type", "capital", true),
    ("Saint-Germain-en-Laye", "type", "château", true),
    ("Pau", "type", "city", true),
    
    # Dynasties
    ("Capetian", "type", "dynasty", true),
    ("House of Valois", "type", "dynasty", true),
    ("Bourbon", "type", "dynasty", true),
]

# Relation types we expect to extract
const EXPECTED_RELATION_TYPES = [
    "reigned_from",
    "reigned_until",
    "parent_of",
    "spouse",
    "spouse_of",
    "born_in",
    "died_in",
    "predecessor",
    "successor",
    "dynasty",
    "title",
    "nationality",
    "type",
    "started",
    "ended",
    "executed",
    "grandparent_of",
    "parent"
]

# Entity types we expect to extract
const EXPECTED_ENTITY_TYPES = [
    "PERSON",
    "LOCATION", 
    "TITLE",
    "ORGANIZATION",
    "DATE",
    "EVENT"
]

"""
Get all reference facts as a vector
"""
function get_all_reference_facts()
    return FRENCH_MONARCHY_FACTS
end

"""
Get reference facts for a specific monarch
"""
function get_reference_facts_for_monarch(name::String)
    return [f for f in FRENCH_MONARCHY_FACTS if f[1] == name]
end

"""
Get expected relation types
"""
function get_expected_relation_types()
    return EXPECTED_RELATION_TYPES
end

"""
Get expected entity types
"""
function get_expected_entity_types()
    return EXPECTED_ENTITY_TYPES
end
