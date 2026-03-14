"""
Reusable test fixtures for GraphMERT.jl

Provides cross-domain test data that can be shared across different domain tests.
"""

using Random

# ============================================================================
# Generic Test Articles
# ============================================================================

const SAMPLE_PERSON_ARTICLE = """
John Smith is a computer scientist who works at MIT. He was born in Boston in 1985.
John completed his PhD at Stanford University in 2012. His research focuses on
machine learning and natural language processing. He has published over 50 papers
and received several awards for his work.
"""

const SAMPLE_LOCATION_ARTICLE = """
Paris is the capital city of France, located in the Île-de-France region.
With a population of over 2 million people, Paris is one of the largest
cities in Europe. The city is famous for the Eiffel Tower, the Louvre Museum,
and Notre-Dame Cathedral. Paris has been a center of art, culture, and politics
for centuries.
"""

const SAMPLE_ORGANIZATION_ARTICLE = """
Google LLC is an American multinational technology company headquartered in
Mountain View, California. Google was founded in 1998 by Larry Page and Sergey Brin.
The company offers a wide range of products and services including search,
advertising, cloud computing, and software. Alphabet Inc. is the parent company
of Google.
"""

# ============================================================================
# Generic Entity Lists
# ============================================================================

const COMMON_PERSONS = [
    "John Smith", "Jane Doe", "Alice Johnson", "Bob Williams",
    "Mary Brown", "Michael Jones", "Sarah Davis", "David Miller"
]

const COMMON_LOCATIONS = [
    "Boston", "San Francisco", "New York", "London", "Paris",
    "Tokyo", "Berlin", "Sydney", "Toronto", "Mumbai"
]

const COMMON_ORGANIZATIONS = [
    "MIT", "Stanford University", "Harvard University", "Google",
    "Microsoft", "Apple", "Amazon", "Facebook", "IBM"
]

# ============================================================================
# Evaluation Helpers
# ============================================================================

"""
Calculate precision for extracted entities against expected entities.
"""
function calculate_entity_precision(extracted::Vector, expected::Vector)
    if isempty(extracted)
        return isempty(expected) ? 1.0 : 0.0
    end
    matches = 0
    for ext in extracted
        for exp in expected
            if ext[1] == exp[1]
                matches += 1
                break
            end
        end
    end
    return matches / length(extracted)
end

"""
Calculate recall for extracted entities against expected entities.
"""
function calculate_entity_recall(extracted::Vector, expected::Vector)
    if isempty(extracted)
        return isempty(expected) ? 1.0 : 0.0
    end
    matches = 0
    for exp in expected
        for ext in extracted
            if ext[1] == exp[1]
                matches += 1
                break
            end
        end
    end
    return matches / length(expected)
end

"""
Calculate F1 score from precision and recall.
"""
function calculate_f1_score(precision::Float64, recall::Float64)
    if precision + recall == 0.0
        return 0.0
    end
    return 2 * (precision * recall) / (precision + recall)
end

"""
Calculate all metrics at once.
"""
function calculate_metrics(extracted::Vector, expected::Vector)
    precision = calculate_entity_precision(extracted, expected)
    recall = calculate_entity_recall(extracted, expected)
    f1 = calculate_f1_score(precision, recall)
    return (precision=precision, recall=recall, f1=f1)
end
