# Wikipedia Domain Test Reproducibility

## Random Seeds

All Wikipedia domain tests use a fixed random seed for reproducibility:

| Test File | Random Seed | Purpose |
|----------|------------|---------|
| `run_entity_extraction.jl` | 42 | Entity extraction validation |
| `run_relation_extraction.jl` | 42 | Relation extraction validation |
| `run_quality_assessment.jl` | 42 | Quality metrics validation |
| `run_export.jl` | 42 | Export functionality validation |

## Reproducibility Notes

1. **Fixed Seed**: All tests use `Random.seed!(TEST_RANDOM_SEED)` where `TEST_RANDOM_SEED = 42`
2. **Deterministic Results**: With the same seed, tests produce consistent results across runs
3. **Julia Version**: Tests are designed for Julia 1.10+
4. **Package Versions**: See `Project.toml` for exact dependency versions

## Running Tests with Different Seeds

To test with different seeds, modify the `TEST_RANDOM_SEED` constant in each test file:

```julia
const TEST_RANDOM_SEED = 123  # Change as needed
```

## Known Non-Deterministic Elements

Some aspects may vary between runs:
- Parallel processing order (if enabled via `ProcessingOptions.parallel_processing=true`)
- External API calls (Wikidata lookups - disabled in test mode)
