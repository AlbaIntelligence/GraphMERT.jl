"""
Evaluation-related types for GraphMERT.jl.

These were split out from `types.jl` to keep core type definitions smaller
and to make it easier for agents to load only the pieces they need.
"""

module TypesEval

using Dates

export FActScore, ValidityScore, GraphRAG

struct FActScore
  score::Float64
  precision::Float64
  recall::Float64
  f1::Float64
  total_facts::Int
  correct_facts::Int
  incorrect_facts::Int
end

struct ValidityScore
  score::Float64
  valid_relations::Int
  total_relations::Int
  invalid_relations::Int
end

struct GraphRAG
  score::Float64
  retrieval_accuracy::Float64
  generation_quality::Float64
  overall_performance::Float64
end

end # module
