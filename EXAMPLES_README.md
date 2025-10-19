# GraphMERT.jl Examples and Test Suite

This directory contains comprehensive examples and test suites for GraphMERT.jl, demonstrating the library's capabilities and ensuring code quality.

## 📚 Examples

### Biomedical Examples (Following Original Paper)

The biomedical examples follow the progression of the original GraphMERT paper, demonstrating each step of the knowledge graph construction pipeline:

1. **`examples/biomedical/01_basic_entity_extraction.jl`**
   - Demonstrates basic biomedical entity extraction
   - Shows entity type classification and validation
   - Includes confidence scoring and statistics
   - Based on GraphMERT paper Section 3.1

2. **`examples/biomedical/02_relation_extraction.jl`**
   - Demonstrates relation extraction between entities
   - Shows relation type classification and validation
   - Includes graph density and network analysis
   - Based on GraphMERT paper Section 3.2

3. **`examples/biomedical/03_knowledge_graph_construction.jl`**
   - Demonstrates complete knowledge graph construction
   - Shows entity merging and deduplication
   - Includes graph analysis and export functionality
   - Based on GraphMERT paper Section 3.3

4. **`examples/biomedical/04_training_pipeline.jl`** (Coming Soon)
   - Will demonstrate MLM and MNM training objectives
   - Shows model training and evaluation
   - Based on GraphMERT paper Section 4

### Wikipedia Examples (General Domain)

The Wikipedia examples demonstrate GraphMERT's generalizability beyond biomedical domains:

1. **`examples/wikipedia/01_wikipedia_entity_extraction.jl`**
   - Demonstrates entity extraction from Wikipedia-style text
   - Shows adaptation to general domain entities
   - Includes performance comparison with biomedical domain

2. **`examples/wikipedia/02_wikipedia_knowledge_graph.jl`** (Coming Soon)
   - Will demonstrate knowledge graph construction from Wikipedia articles
   - Shows cross-domain knowledge graph analysis

## 🧪 Test Suite

### Comprehensive Testing

The test suite provides extensive coverage of GraphMERT functionality:

1. **`test/test_entities.jl`**
   - Tests biomedical entity types and classification
   - Validates entity extraction and normalization
   - Includes performance and edge case testing
   - 100+ individual test cases

2. **`test/test_relations.jl`**
   - Tests biomedical relation types and classification
   - Validates relation extraction and validation
   - Includes UMLS mapping and confidence scoring
   - 150+ individual test cases

3. **`test/run_tests.jl`**
   - Comprehensive test runner
   - Performance benchmarks
   - Memory usage testing
   - Detailed reporting and recommendations

### Test Categories

- **Unit Tests**: Individual function testing
- **Integration Tests**: End-to-end pipeline testing
- **Performance Tests**: Speed and memory benchmarks
- **Edge Case Tests**: Boundary condition testing
- **Regression Tests**: Prevent breaking changes

## 🚀 Running Examples

### Prerequisites

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

### Running Biomedical Examples

```bash
# Run examples in sequence
julia examples/biomedical/01_basic_entity_extraction.jl
julia examples/biomedical/02_relation_extraction.jl
julia examples/biomedical/03_knowledge_graph_construction.jl
```

### Running Wikipedia Examples

```bash
# Run Wikipedia examples
julia examples/wikipedia/01_wikipedia_entity_extraction.jl
```

### Running Tests

```bash
# Run all tests
julia test/run_tests.jl

# Run individual test modules
julia test/test_entities.jl
julia test/test_relations.jl
```

## 📊 Expected Output

### Example Output

Each example provides:
- **Progress indicators** with emojis and clear formatting
- **Detailed statistics** on entities, relations, and graph metrics
- **Performance metrics** including processing time and confidence scores
- **Visual representations** of knowledge graphs and relationships
- **Export capabilities** for further analysis

### Test Output

The test suite provides:
- **Test results** with pass/fail status
- **Performance benchmarks** in entities/second and relations/second
- **Memory usage** measurements
- **Coverage statistics** and recommendations
- **Detailed error reporting** for failed tests

## 🔧 Configuration

### API Keys (Optional)

For full functionality, configure API keys:

```julia
# UMLS API key for biomedical entity linking
umls_client = create_umls_client("your-umls-api-key")

# LLM API key for helper LLM integration
llm_client = create_helper_llm_client("your-llm-api-key")
```

### Fallback Mode

All examples and tests work in fallback mode without API keys, using rule-based methods for entity and relation extraction.

## 📈 Performance Expectations

### Entity Extraction
- **Speed**: 1000+ entities/second
- **Accuracy**: 85%+ for biomedical text
- **Memory**: <10MB for typical documents

### Relation Classification
- **Speed**: 500+ relations/second
- **Accuracy**: 80%+ for biomedical relations
- **Memory**: <5MB for typical batches

### Knowledge Graph Construction
- **Speed**: <1 second for 100 entities
- **Memory**: <50MB for 1000 entities
- **Scalability**: Linear with input size

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure GraphMERT.jl is properly installed
2. **API Errors**: Check API keys and network connectivity
3. **Memory Issues**: Reduce batch sizes for large datasets
4. **Performance Issues**: Enable caching and optimize text preprocessing

### Getting Help

- Check the test output for specific error messages
- Review the example code for usage patterns
- Consult the main GraphMERT.jl documentation
- Report issues with detailed error logs

## 🔄 Continuous Integration

The test suite is designed for CI/CD integration:

```yaml
# Example GitHub Actions workflow
- name: Run GraphMERT Tests
  run: julia test/run_tests.jl
```

## 📝 Contributing

When adding new examples or tests:

1. Follow the existing naming convention
2. Include comprehensive documentation
3. Add performance benchmarks
4. Test edge cases and error conditions
5. Update this README with new examples

## 🎯 Future Enhancements

- [ ] Interactive Jupyter notebook examples
- [ ] Real-time visualization of knowledge graphs
- [ ] Comparative analysis with other methods
- [ ] Large-scale dataset examples
- [ ] Multi-language support examples
- [ ] Advanced training pipeline examples
