# Troubleshooting

This guide helps you resolve common issues when using GraphMERT.jl.

## Installation Issues

### Package Not Found

**Error**: `Package GraphMERT not found`

**Solution**:
```julia
using Pkg
Pkg.update()
Pkg.add("GraphMERT")
```

### Dependency Conflicts

**Error**: `Unsatisfiable requirements detected`

**Solution**:
```julia
using Pkg
Pkg.resolve()
Pkg.build("GraphMERT")
```

### Memory Allocation Errors

**Error**: `OutOfMemoryError`

**Solutions**:
1. Increase heap size:
   ```bash
   julia --heap-size-hint=4G
   ```

2. Reduce batch size:
   ```julia
   config = BatchProcessingConfig(batch_size=5, memory_limit=2.0)
   ```

3. Process smaller chunks:
   ```julia
   # Split long text into chunks
   chunks = split_text_into_chunks(text, max_length=1000)
   graphs = [extract_knowledge_graph(chunk) for chunk in chunks]
   ```

## Runtime Issues

### Low Entity Extraction

**Problem**: Few entities extracted from text

**Solutions**:
1. Lower confidence threshold:
   ```julia
   config = GraphMERTConfig(min_confidence=0.5)
   ```

2. Enable UMLS integration:
   ```julia
   config = GraphMERTConfig(enable_umls=true, umls_threshold=0.7)
   ```

3. Enable LLM assistance:
   ```julia
   config = GraphMERTConfig(enable_llm=true, llm_model="gpt-3.5-turbo")
   ```

### Poor Relation Quality

**Problem**: Incorrect or missing relations

**Solutions**:
1. Improve text quality:
   ```julia
   # Clean and normalize text
   cleaned_text = preprocess_text(text, remove_stopwords=true)
   ```

2. Adjust relation confidence:
   ```julia
   # Filter relations by confidence
   filtered_graph = filter_knowledge_graph(graph, min_confidence=0.8)
   ```

3. Use domain-specific configuration:
   ```julia
   config = GraphMERTConfig(
       min_confidence=0.7,
       enable_umls=true,
       umls_threshold=0.8
   )
   ```

### Performance Issues

**Problem**: Slow processing

**Solutions**:
1. Enable parallel processing:
   ```julia
   config = BatchProcessingConfig(enable_parallel=true)
   ```

2. Optimize batch size:
   ```julia
   # Start with small batches and increase
   config = BatchProcessingConfig(batch_size=10)
   ```

3. Monitor memory usage:
   ```julia
   monitor = create_performance_monitor()
   graph = extract_knowledge_graph(text, monitor=monitor)
   println("Memory: ", get_memory_usage(monitor), " MB")
   ```

## Configuration Issues

### Invalid Configuration

**Error**: `Invalid configuration parameter`

**Solution**:
```julia
# Use default configuration first
config = GraphMERTConfig()
graph = extract_knowledge_graph(text, config)

# Then customize gradually
config = GraphMERTConfig(
    min_confidence=0.7,  # Valid range: 0.0-1.0
    max_entities=100,    # Valid range: 1-1000
    enable_umls=true     # Boolean
)
```

### UMLS Connection Issues

**Error**: `UMLS connection failed`

**Solutions**:
1. Check internet connection
2. Verify UMLS credentials
3. Disable UMLS if not needed:
   ```julia
   config = GraphMERTConfig(enable_umls=false)
   ```

### LLM Integration Issues

**Error**: `LLM API error`

**Solutions**:
1. Check API key configuration
2. Verify model availability
3. Disable LLM if not needed:
   ```julia
   config = GraphMERTConfig(enable_llm=false)
   ```

## Data Issues

### Empty Knowledge Graph

**Problem**: No entities or relations extracted

**Solutions**:
1. Check text quality:
   ```julia
   # Ensure text is not empty
   @assert !isempty(text)
   ```

2. Lower confidence thresholds:
   ```julia
   config = GraphMERTConfig(min_confidence=0.3)
   ```

3. Enable all features:
   ```julia
   config = GraphMERTConfig(
       enable_umls=true,
       enable_llm=true,
       min_confidence=0.5
   )
   ```

### Encoding Issues

**Error**: `StringIndexError` or garbled text

**Solutions**:
1. Check text encoding:
   ```julia
   # Ensure UTF-8 encoding
   text = String(text)
   ```

2. Clean text before processing:
   ```julia
   cleaned_text = preprocess_text(text, normalize_unicode=true)
   ```

### Large File Processing

**Problem**: Memory issues with large files

**Solutions**:
1. Process in chunks:
   ```julia
   function process_large_file(filepath, chunk_size=1000)
       chunks = read_file_in_chunks(filepath, chunk_size)
       graphs = [extract_knowledge_graph(chunk) for chunk in chunks]
       return merge_knowledge_graphs(graphs)
   end
   ```

2. Use streaming processing:
   ```julia
   config = BatchProcessingConfig(
       batch_size=5,
       memory_limit=2.0
   )
   ```

## Debugging

### Enable Debug Logging

```julia
using Logging

# Enable debug logging
Logging.with_logger(Logging.ConsoleLogger(stderr, Logging.Debug)) do
    graph = extract_knowledge_graph(text)
end
```

### Performance Profiling

```julia
using Profile

# Profile performance
Profile.clear()
@profile extract_knowledge_graph(text)
Profile.print()
```

### Memory Profiling

```julia
using Pkg
Pkg.add("ProfileView")

using ProfileView

# Profile memory usage
Profile.clear()
@profile extract_knowledge_graph(text)
ProfileView.view()
```

## Getting Help

### Check Documentation

1. [API Reference](api/core.md): Complete function documentation
2. [User Guide](user_guide/core_concepts.md): Detailed usage instructions
3. [Installation Guide](getting_started/installation.md): Setup instructions

### Community Support

1. **GitHub Issues**: [Report bugs](https://github.com/alba-intelligence/GraphMERT.jl/issues)
2. **GitHub Discussions**: [Ask questions](https://github.com/alba-intelligence/GraphMERT.jl/discussions)
3. **Email Support**: alba.intelligence@gmail.com

### Provide Information

When reporting issues, include:

1. **Julia Version**: `julia --version`
2. **Package Version**: `Pkg.status("GraphMERT")`
3. **Error Message**: Complete error traceback
4. **Reproducible Example**: Minimal code that reproduces the issue
5. **System Information**: OS, memory, CPU details

### Example Bug Report

```julia
# Minimal reproducible example
using GraphMERT

text = "Diabetes is treated with insulin."
try
    graph = extract_knowledge_graph(text)
    println("Success: ", length(graph.entities), " entities")
catch e
    println("Error: ", e)
    # Include full stacktrace
    showerror(stdout, e, catch_backtrace())
end
```
