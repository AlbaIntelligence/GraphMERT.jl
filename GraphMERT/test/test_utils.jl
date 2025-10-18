"""
Test utilities for GraphMERT.jl

Provides common testing functions and mock data for comprehensive testing.
"""

using Test
using Random
using Dates

# Test data generators
function generate_mock_text()
  """Generate mock biomedical text for testing"""
  texts = [
    "Diabetes is a chronic condition that affects blood sugar levels.",
    "Insulin therapy is commonly used to treat type 1 diabetes.",
    "Metformin is an oral medication for type 2 diabetes management.",
    "HbA1c levels indicate long-term blood glucose control.",
    "Diabetic neuropathy affects peripheral nerves in diabetes patients."
  ]
  return texts[rand(1:length(texts))]
end

function generate_mock_entities()
  """Generate mock biomedical entities for testing"""
  return [
    ("diabetes", "DISEASE", 0.95),
    ("insulin", "DRUG", 0.92),
    ("blood sugar", "BIOMARKER", 0.88),
    ("HbA1c", "BIOMARKER", 0.90),
    ("neuropathy", "SYMPTOM", 0.85)
  ]
end

function generate_mock_relations()
  """Generate mock biomedical relations for testing"""
  return [
    ("diabetes", "TREATS", "insulin", 0.89),
    ("diabetes", "AFFECTS", "blood sugar", 0.94),
    ("metformin", "TREATS", "diabetes", 0.91),
    ("diabetes", "CAUSES", "neuropathy", 0.87)
  ]
end

# Test configuration
const TEST_RANDOM_SEED = 42
const TEST_CONFIDENCE_THRESHOLD = 0.8

# Setup test environment
function setup_test_environment()
  """Setup test environment with consistent random seed"""
  Random.seed!(TEST_RANDOM_SEED)
  return nothing
end

# Performance testing utilities
function measure_execution_time(f, args...)
  """Measure execution time of a function"""
  start_time = time()
  result = f(args...)
  end_time = time()
  return result, end_time - start_time
end

function measure_memory_usage(f, args...)
  """Measure memory usage of a function (simplified)"""
  # In practice, this would use more sophisticated memory profiling
  result = f(args...)
  return result, 0  # Placeholder for memory usage
end

# Validation utilities
function validate_entity(entity)
  """Validate that an entity meets GraphMERT requirements"""
  @test hasfield(typeof(entity), :id)
  @test hasfield(typeof(entity), :text)
  @test hasfield(typeof(entity), :label)
  @test hasfield(typeof(entity), :confidence)
  @test 0.0 <= entity.confidence <= 1.0
  return true
end

function validate_relation(relation)
  """Validate that a relation meets GraphMERT requirements"""
  @test hasfield(typeof(relation), :head)
  @test hasfield(typeof(relation), :tail)
  @test hasfield(typeof(relation), :relation_type)
  @test hasfield(typeof(relation), :confidence)
  @test 0.0 <= relation.confidence <= 1.0
  return true
end

function validate_knowledge_graph(graph)
  """Validate that a knowledge graph meets GraphMERT requirements"""
  @test hasfield(typeof(graph), :entities)
  @test hasfield(typeof(graph), :relations)
  @test hasfield(typeof(graph), :confidence_threshold)
  @test 0.0 <= graph.confidence_threshold <= 1.0
  return true
end

# Mock data for testing
const MOCK_BIOMEDICAL_TEXT = """
Diabetes mellitus is a group of metabolic disorders characterized by high blood sugar levels over a prolonged period.
Type 1 diabetes results from the pancreas's failure to produce enough insulin.
Type 2 diabetes begins with insulin resistance, a condition in which cells fail to respond to insulin properly.
"""

const MOCK_UMLS_MAPPINGS = Dict(
  "diabetes" => "C0011884",
  "insulin" => "C0021641",
  "metformin" => "C0025234",
  "HbA1c" => "C0201980"
)

# Test constants
const TEST_PERFORMANCE_TARGETS = (
  tokens_per_second=5000,
  memory_limit_gb=4.0,
  factscore_target=0.698,
  validity_target=0.688
)
