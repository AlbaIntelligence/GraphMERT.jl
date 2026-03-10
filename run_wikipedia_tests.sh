#!/bin/bash
# Wikipedia Domain Test Runner for GraphMERT.jl
#
# Usage:
#   ./run_wikipedia_tests.sh
#
# This script runs the Wikipedia domain tests in the GraphMERT.jl package.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo "Wikipedia Domain Test Runner"
echo "========================================"
echo ""

# Check if Julia is available
if ! command -v julia &>/dev/null; then
	echo "Error: Julia not found. Please install Julia 1.8+"
	exit 1
fi

echo "Julia version: $(julia --version)"
echo ""

# Run the test script
julia --project=GraphMERT run_wikipedia_tests.jl "$@"
