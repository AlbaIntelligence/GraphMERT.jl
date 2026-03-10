# Testing and Validation Guidelines

**Date**: 2026-01-29
**Purpose**: Testing and validation guidelines for Interlisp project

## Core Testing Strategy

- **Fast Smoke Tests**: Use quick test runner by default (user preference)
- **Comprehensive Tests**: Keep separate for thorough validation
- **Test Location**: Tests should be next to scripts in `tests/` folders

## Parity Testing Methodology

- **Cross-Reference C Traces**: Compare execution traces instruction-by-instruction
- **Step-by-Step Validation**: Run emulators with `EMULATOR_MAX_STEPS=N` for controlled testing
- **Canonical Comparison Script**: Use `scripts/compare_emulator_execution.sh` for systematic comparison
- **Unified Trace Format**: Ensure consistent logging format across implementations

## Regression Testing

- **Commit-Hook Validation**: Run parity tests before commits to catch regressions
- **Incremental Testing**: Test after each opcode implementation
- **Boundary Testing**: Validate edge cases and error conditions

## Debugging Integration

- **Systematic Debugging**: Follow `documentation/core/critical-debugging-technique.typ` hierarchy
- **Trace Analysis**: Use comparison tools for divergence identification
- **Performance Profiling**: Monitor execution speed during testing

## Systematic Debugging Workflow

**CRITICAL**: For any implementation divergence or bug, follow this systematic process:

1. **Cross-reference C traces** - Establish baseline with verified C emulator output
2. **Instrument execution** - Add targeted debug prints following technique hierarchy
3. **Validate incrementally** - Test after each code change, not at the end
4. **Document findings** - Update both specifications and implementation docs
5. **Regression test** - Ensure no existing functionality breaks

**Key Principle**: Never implement blindly - always validate against C reference first.

**Performance Considerations**:

- VM emulators require careful optimization - profile memory access patterns
- Stack operations are performance-critical - minimize redundant calculations
- Memory translation overhead should be amortized across operations
- Consider instruction caching for frequently executed opcodes

## Unified Trace Format

**Purpose**: Enable rapid divergence identification between C and Zig emulators

**Format**: Single-line, pipe-delimited columns:

```
LINE#|PC|INSTRUCTION|OPCODE|OPERANDS|REGISTERS|FLAGS|SP_FP|STACK_SUMMARY|MEMORY_CONTEXT|FP_VP_FO_VA|BS_MEM|NOTES
```

**Key Benefits**:

- **Rapid comparison** with awk/Python scripts
- **Comprehensive context** in single line
- **Consistent format** across both emulators
- **Memory issue triage** with dedicated fields

**Comparison Tools**:

- `scripts/compare_unified_traces.awk` - Fast awk-based comparison
- `scripts/compare_unified_traces.py` - Detailed Python analysis

## Centralized Memory Management

**Problem Solved**: Scattered memory logic causing recurring address translation, endianness, and paging issues

**Solution**: Centralized memory management module (`zaiko/src/memory/manager.zig`)

**Components**:

- **AddressManager**: LispPTR ↔ byte conversions, virtual page calculations
- **FPtoVPManager**: File page ↔ virtual page mapping, page OK flags
- **EndiannessManager**: Byte-swapping logic, XOR addressing
- **MemoryAccessManager**: Safe memory reads, bounds checking

**Integration**: Both C and Zig emulators use centralized functions for consistency

## Parity Workflow (C vs Zig) – Canonical

For repeatable execution-trace parity work:

- **Canonical script**: `scripts/compare_emulator_execution.sh`
  - Supports the shared runtime cap knob **`EMULATOR_MAX_STEPS`** (unset/0 → run to completion).
- **Unified trace format**: Single-line column-formatted traces for rapid comparison
  - **C trace**: `c_emulator_unified_trace.txt`
  - **Zig trace**: `zig_emulator_unified_trace.txt`
  - **Comparison scripts**: `scripts/compare_unified_traces.awk`, `scripts/compare_unified_traces.py`
- **Divergence analysis**: `scripts/analyze_execution_divergence.py` (supports LCP skip + `--start-line` resume).
- **Path robustness**: Prefer **absolute sysout paths** (scripts should not depend on cwd).

## Testing Commands

### Build and Test Zig Emulator
```bash
cd zaiko
zig build
zig build test
```

### Run Zig Emulator
```bash
zig build run -- medley/internal/loadups/starter.sysout
```

### Run C Emulator (Baseline)
```bash
cd /home/emmanuel/Sync/Development/Emulation/_gits/Interlisp
EMULATOR_MAX_STEPS=5 ./maiko/linux.x86_64/ldesdl /home/emmanuel/Sync/Development/Emulation/_gits/Interlisp/medley/internal/loadups/starter.sysout
```

### Run Zig Emulator (Comparison)
```bash
cd /home/emmanuel/Sync/Development/Emulation/_gits/Interlisp/zaiko
env ZIG_GLOBAL_CACHE_DIR=zaiko/.zig-cache EMULATOR_MAX_STEPS=5 zig build run -- /home/emmanuel/Sync/Development/Emulation/_gits/Interlisp/medley/internal/loadups/starter.sysout
```

### Compare Traces
```bash
cd /home/emmanuel/Sync/Development/Emulation/_gits/Interlisp
head -3 c_emulator_execution_log.txt
head -3 zaiko/zig_emulator_execution_log.txt
```

### Run Comparison Script
```bash
cd /home/emmanuel/Sync/Development/Emulation/_gits/Interlisp
EMULATOR_MAX_STEPS=100 ./scripts/compare_emulator_execution.sh /home/emmanuel/Sync/Development/Emulation/_gits/Interlisp/medley/internal/loadups/starter.sysout
```

## Environment Variables

- `ZIG_GLOBAL_CACHE_DIR=zaiko/.zig-cache` - Required for Zig builds
- `EMULATOR_MAX_STEPS=N` - Limit execution steps for testing

## Testing Best Practices

### For C Implementation (Production Reference)
1. **Use maiko/ as primary reference** - Contains superior documentation
2. **Leverage comprehensive headers** - Use confidence levels and testing guidance
3. **Preserve documentation quality** - Maintain structured explanations
4. **Cross-reference maiko_untouched/** - Only for historical context

### For Zig Development (Parity Focus)
1. **Address TODO markers systematically** - 245 markers indicate critical gaps
2. **Replace placeholder implementations** - Focus on non-functional stubs
3. **Implement missing functional areas** - Priority: floating point, graphics, I/O
4. **Test comprehensively** - Build coverage as implementations are completed
5. **Document algorithms** - Add explanations during implementation, not after

### General Practices
1. **Always check C implementation first** when implementing new features
2. **Update documentation** before committing (follow `documentation/core/critical-memory.typ`)
3. **Keep files under 500 lines** for better maintainability
4. **Test incrementally** - don't wait until everything is done
5. **Document findings** in `documentation` for future reference
6. **Use descriptive commit messages** with task IDs when applicable
7. **Follow systematic debugging** from `documentation/core/critical-debugging-technique.typ`
8. **Validate against C traces** before considering implementation complete
9. **Handle edge cases** identified during C code analysis
10. **Profile performance** during development, not just at the end

---

**Last Updated**: 2026-01-29
**Status**: Testing infrastructure operational; parity workflow established
