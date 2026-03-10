"""
Test utilities for Wikipedia domain testing in GraphMERT.jl

Provides testing functions and mock data for French monarchy testing.
"""

using Test
using Random
using Dates

# French monarchy test articles (abbreviated for testing)
const LOUIS_XIV_ARTICLE = """
Louis XIV (5 September 1638 – 1 September 1715), known as the Sun King, was King of France from 1643 until his death in 1715. His reign of 72 years and 110 days is the longest of any major European monarch.

Louis XIV was born at the Château de Saint-Germain-en-Laye. He became king at the age of four under the regency of his mother, Anne of Austria. His father, Louis XIII, had died in 1643.

Louis XIV married Maria Theresa of Spain in 1660. They had several children including Louis, Grand Dauphin, who was the father of Louis XV.

Louis XIV's reign was marked by wars of expansion, including the War of the Spanish Succession. He built the Palace of Versailles and centralized French government power.
"""

const HENRY_IV_ARTICLE = """
Henry IV (13 December 1553 – 14 May 1610), also known as Henry the Great, was King of France from 1589 to his death in 1610. He was the first Bourbon king of France.

Born in Pau, Henry was originally a Huguenot leader. He converted to Catholicism in 1593, famously stating that "Paris is well worth a mass."

Henry IV married Margaret of Valois in 1572. He later married Marie de' Medici in 1600. Their son Louis XIII succeeded him.

Henry IV's reign ended when he was assassinated by François Ravaillac. His successors included Louis XIII and the Bourbon dynasty continued through Louis XIV.
"""

const MARIE_ANTOINETTE_ARTICLE = """
Marie Antoinette (2 November 1755 – 16 October 1793) was the last Queen of France before the French Revolution. She was born Archduchess Maria Theresa of Austria and married Louis XVI in 1770.

Marie Antoinette and Louis XVI had four children: Marie-Thérèse Charlotte, Louis-Joseph, Louis-Charles (Dauphin), and Sophie. Louis-Charles was the father of Louis XVII.

During the Revolution, Marie Antoinette was nicknamed "Madame Deficit" and accused of extravagance. After the flight to Varennes in 1791, tensions increased.

Marie Antoinette was executed by guillotine in Paris. Her son Louis-Charles died in prison. Her daughter Marie-Thérèse survived and later became Duchess of Angoulême.
"""

const FRENCH_KINGS = [
    "Louis IX", "Louis X", "Louis XI", "Louis XII", "Louis XIII",
    "Louis XIV", "Louis XV", "Louis XVI", "Louis XVII", "Louis XVIII",
    "Charles V", "Charles VI", "Charles VII", "Charles VIII", "Charles IX",
    "Henry II", "Henry III", "Henry IV",
    "Francis I", "Francis II"
]

const FRENCH_QUEENS = [
    "Catherine de' Medici", "Marie de' Medici", "Anne of Austria",
    "Maria Theresa of Spain", "Marie Leszczyńska", "Marie Antoinette"
]

const FRENCH_LOCATIONS = [
    "Paris", "Versailles", "Louvre", "Saint-Germain-en-Laye",
    "Fontainebleau", "Blois", "Pau", "Reims"
]

const FRENCH_DYNASTIES = [
    "Capetian", "House of Valois", "Bourbon", "Orléans"
]

# Test configuration
const TEST_RANDOM_SEED = 42
const TEST_CONFIDENCE_THRESHOLD = 0.5

function get_french_monarchy_articles()
    """Return dictionary of French monarchy test articles"""
    return Dict(
        "louis_xiv" => LOUIS_XIV_ARTICLE,
        "henry_iv" => HENRY_IV_ARTICLE,
        "marie_antoinette" => MARIE_ANTOINETTE_ARTICLE
    )
end

function get_expected_entities(article_name::String)
    """Return expected entities for a given article"""
    if article_name == "louis_xiv"
        return [
            ("Louis XIV", "PERSON", 0.9),
            ("France", "LOCATION", 0.85),
            ("Sun King", "TITLE", 0.8),
            ("Louis XIII", "PERSON", 0.85),
            ("Anne of Austria", "PERSON", 0.8),
            ("Maria Theresa of Spain", "PERSON", 0.85),
            ("Louis", "PERSON", 0.7),
            ("Grand Dauphin", "TITLE", 0.75),
            ("Louis XV", "PERSON", 0.8),
            ("Palace of Versailles", "LOCATION", 0.85),
            ("War of the Spanish Succession", "EVENT", 0.7)
        ]
    elseif article_name == "henry_iv"
        return [
            ("Henry IV", "PERSON", 0.9),
            ("Henry the Great", "TITLE", 0.8),
            ("France", "LOCATION", 0.85),
            ("Bourbon", "ORGANIZATION", 0.8),
            ("Paris", "LOCATION", 0.85),
            ("Margaret of Valois", "PERSON", 0.8),
            ("Marie de' Medici", "PERSON", 0.8),
            ("Louis XIII", "PERSON", 0.8),
            ("François Ravaillac", "PERSON", 0.75)
        ]
    elseif article_name == "marie_antoinette"
        return [
            ("Marie Antoinette", "PERSON", 0.9),
            ("Louis XVI", "PERSON", 0.9),
            ("Maria Theresa of Austria", "PERSON", 0.85),
            ("Marie-Thérèse Charlotte", "PERSON", 0.8),
            ("Louis-Joseph", "PERSON", 0.75),
            ("Louis-Charles", "PERSON", 0.75),
            ("Louis XVII", "PERSON", 0.75),
            ("Paris", "LOCATION", 0.85),
            ("French Revolution", "EVENT", 0.8)
        ]
    end
    return []
end

function get_expected_relations(article_name::String)
    """Return expected relations for a given article"""
    if article_name == "louis_xiv"
        return [
            ("Louis XIV", "parent_of", "Louis XV", 0.8),
            ("Louis XIV", "spouse_of", "Maria Theresa of Spain", 0.85),
            ("Louis XIV", "reigned_after", "Louis XIII", 0.9),
            ("Louis XIII", "parent_of", "Louis XIV", 0.85)
        ]
    elseif article_name == "henry_iv"
        return [
            ("Henry IV", "parent_of", "Louis XIII", 0.8),
            ("Henry IV", "spouse_of", "Marie de' Medici", 0.85),
            ("Henry IV", "spouse_of", "Margaret of Valois", 0.8)
        ]
    elseif article_name == "marie_antoinette"
        return [
            ("Marie Antoinette", "spouse_of", "Louis XVI", 0.9),
            ("Marie Antoinette", "parent_of", "Louis-Charles", 0.8),
            ("Louis XVI", "parent_of", "Louis XVII", 0.8)
        ]
    end
    return []
end

# Reference facts for quality assessment
const REFERENCE_FACTS = [
    # Louis XIV facts
    ("Louis XIV", "reigned_from", "1643", true),
    ("Louis XIV", "reigned_until", "1715", true),
    ("Louis XIV", "parent_of", "Louis XV", true),
    ("Louis XIV", "spouse_of", "Maria Theresa of Spain", true),
    ("Louis XIV", "dynasty", "Bourbon", true),
    
    # Henry IV facts
    ("Henry IV", "reigned_from", "1589", true),
    ("Henry IV", "reigned_until", "1610", true),
    ("Henry IV", "dynasty", "Bourbon", true),
    ("Henry IV", "parent_of", "Louis XIII", true),
    
    # Marie Antoinette facts
    ("Marie Antoinette", "spouse_of", "Louis XVI", true),
    ("Marie Antoinette", "parent_of", "Louis XVII", true),
    ("Marie Antoinette", "executed", "1793", true)
]

# Setup test environment
function setup_test_environment()
    Random.seed!(TEST_RANDOM_SEED)
    @info "Test environment initialized with seed: $(TEST_RANDOM_SEED)"
end

# Calculate precision
function calculate_precision(extracted::Vector, expected::Vector)
    if isempty(expected)
        return isempty(extracted) ? 1.0 : 0.0
    end
    matches = 0
    for ext in extracted
        for exp in expected
            if ext[1] == exp[1]  # Match on name
                matches += 1
                break
            end
        end
    end
    return matches / length(extracted)
end

# Calculate recall
function calculate_recall(extracted::Vector, expected::Vector)
    if isempty(extracted)
        return isempty(expected) ? 1.0 : 0.0
    end
    matches = 0
    for exp in expected
        for ext in extracted
            if ext[1] == exp[1]  # Match on name
                matches += 1
                break
            end
        end
    end
    return matches / length(expected)
end

# Calculate F1 score
function calculate_f1(precision::Float64, recall::Float64)
    if precision + recall == 0.0
        return 0.0
    end
    return 2 * (precision * recall) / (precision + recall)
end
