# Installation

## System Requirements

GraphMERT.jl requires:

- **Julia**: Version 1.10 or higher
- **Memory**: Minimum 4GB RAM (8GB recommended for large datasets)
- **Storage**: 2GB free space for models and dependencies

## Installation Methods

### Method 1: Julia Package Manager (Recommended)

```julia
using Pkg
Pkg.add("GraphMERT")
```

### Method 2: Development Installation

For the latest features and development version:

```julia
using Pkg
Pkg.add(url="https://github.com/alba-intelligence/GraphMERT.jl.git")
```

### Method 3: Local Development

```bash
git clone https://github.com/alba-intelligence/GraphMERT.jl.git
cd GraphMERT.jl
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

## Dependencies

GraphMERT.jl automatically installs the following dependencies:

### Core Dependencies
- **Flux.jl**: Machine learning framework
- **Transformers.jl**: RoBERTa model support
- **Graphs.jl**: Graph processing
- **MetaGraphs.jl**: Graph metadata handling

### Data Processing
- **CSV.jl**: Data import/export
- **JSON3.jl**: JSON processing
- **HTTP.jl**: Web API integration

### Scientific Computing
- **Statistics.jl**: Statistical functions
- **Distributions.jl**: Probability distributions
- **LinearAlgebra.jl**: Linear algebra operations

### Performance
- **BenchmarkTools.jl**: Performance benchmarking
- **TextAnalysis.jl**: Text processing utilities

## Verification

After installation, verify the installation:

```julia
using GraphMERT

# Check version
println("GraphMERT version: ", GraphMERT.version())

# Test basic functionality
text = "Diabetes is a chronic condition."
entities = extract_biomedical_terms(text)
println("Extracted entities: ", length(entities))
```

## Troubleshooting

### Common Issues

**Issue**: `Package not found`
```julia
# Solution: Update package registry
using Pkg
Pkg.update()
Pkg.add("GraphMERT")
```

**Issue**: `MethodError` with dependencies
```julia
# Solution: Rebuild packages
using Pkg
Pkg.build("GraphMERT")
```

**Issue**: Memory allocation errors
```julia
# Solution: Increase memory limit
julia --heap-size-hint=4G
```

### Performance Optimization

For optimal performance:

1. **Use Julia 1.10+**: Latest performance improvements
2. **Enable threading**: `julia -t auto`
3. **Precompile packages**: `julia --project=. -e "using GraphMERT"`
4. **Use optimized BLAS**: Install MKL or OpenBLAS

### Getting Help

If you encounter issues:

1. Check the [Troubleshooting Guide](../troubleshooting.md)
2. Search [GitHub Issues](https://github.com/alba-intelligence/GraphMERT.jl/issues)
3. Join [GitHub Discussions](https://github.com/alba-intelligence/GraphMERT.jl/discussions)
4. Contact: alba.intelligence@gmail.com
