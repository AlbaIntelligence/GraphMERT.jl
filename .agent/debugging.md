# Debugging Guidelines

**Date**: 2026-01-29
**Purpose**: Debugging guidelines and techniques for Interlisp project

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
- Performance is a secondary consideration after correctness

## Common Debugging Gotchas (read before parity fixes)

- **PC units**: traces may show both `PC` (bytes) and `PC/2` (DLword address). When indexing `virtual_memory`, use **byte PC**.
- **FPtoVP units**: FPtoVP "virtual page" values correspond to **512-byte pages** (matches trace `[vpage:...]`), not DLword pages.
- **Byte swap vs XOR addressing**:
  - Sysout pages are typically **32-bit byte-swapped** on load on little-endian hosts.
  - Instruction decode may apply **XOR (`addr ^ 3`)** for BYTESWAP byte access; however, trace logging often prints **raw bytes at PC** (no XOR) to match C.
- **CSTKPTRL/TOPOFSTACK synchronization (CRITICAL)**:
  - The C code's `StackPtrRestore()` macro restores `CSTKPTRL` from `CurrentStackPTR` before each opcode
  - **TOPOFSTACK must be read from memory** (`*(CSTKPTRL - 1)`) after restoring CSTKPTRL, NOT from a cached value
  - This is because operations like GVAR push values (changing TOPOFSTACK), and UNBIND walks CSTKPTRL into the binding stack
  - Without syncing TOPOFSTACK from memory, operations after UNBIND see stale cached values
  - **Fix**: Call `readTopOfStackFromMemory()` after `initCSTKPTRLFromCurrentStackPTR()` in the dispatch loop

## Complex Multi-Step Debugging

For issues requiring systematic investigation:

1. **Isolate the problem** - Reproduce with minimal test case
2. **Establish baseline** - Verify C emulator behavior with traces
3. **Apply debugging hierarchy** - Start with cross-referencing, progress through techniques
4. **Document incrementally** - Update findings as they're discovered
5. **Validate fixes** - Ensure no regressions in existing functionality
6. **Commit comprehensively** - Include documentation updates in commits

## Key Concepts

### Memory Management

- **LispPTR**: 32-bit virtual address (DLword offset from Lisp_world, multiply by 2 for bytes)
- **DLword**: 16-bit unsigned integer
- **Virtual Memory**: Complete Lisp address space, allocated based on `process_size`
- **FPtoVP Table**: Maps file page numbers to virtual page numbers
- **Stack**: Part of virtual memory, grows DOWN

### VM Execution

- **PC (Program Counter)**: Byte offset in virtual memory
- **Stack Frame**: 10 DLwords (20 bytes), contains return address, saved registers, local variables
- **Function Header**: Contains `startpc` (byte offset), `na` (arg count), `pv` (param var count)
- **Dispatch Loop**: Fetches, decodes, and executes bytecode instructions

### SDL2 Integration

- **Display Region**: DLword array where each bit represents a pixel
- **BitBLT**: Converts bit array to pixels (foreground/background colors)
- **Event Translation**: SDL keycodes → Lisp keycodes via keymap (74 entries)
- **Coordinate Translation**: Window coordinates → Display coordinates (divide by pixel_scale)

## Critical Debugging Techniques

### Technique Hierarchy

When debugging implementation divergences, follow this hierarchy:

1. **Cross-reference C traces** - Compare execution traces instruction-by-instruction
2. **Step-by-instruction validation** - Run emulators with `EMULATOR_MAX_STEPS=N`
3. **Debug instrumentation** - Add targeted prints following technique hierarchy
4. **Memory integrity verification** - Check for corruption at critical addresses
5. **Algorithm reverse engineering** - Analyze C code for correct implementation
6. **Hypothesis testing** - Formulate and validate specific bug theories
7. **Documentation updates** - Record findings in specifications and implementations

### Trace Analysis

- Use unified trace format for rapid comparison
- Identify first divergence point
- Analyze memory context at divergence
- Check stack state and register values
- Verify instruction decoding

### Memory Integrity Checks

- Verify address calculations are correct
- Check bounds on all memory accesses
- Validate byte-swapping logic
- Confirm FPtoVP table mappings
- Test XOR addressing for instruction fetch

## Debugging Tools

### Comparison Scripts

- `scripts/compare_emulator_execution.sh` - Main comparison script
- `scripts/compare_unified_traces.awk` - Fast awk-based comparison
- `scripts/compare_unified_traces.py` - Detailed Python analysis
- `scripts/analyze_execution_divergence.py` - Divergence analysis with resume capability

### Debug Output

- C emulator: `c_emulator_execution_log.txt` (native detailed trace format)
- Zig emulator: `zaiko/zig_emulator_execution_log.txt` (unified single-line trace format)
- Unified traces: `c_emulator_unified_trace.txt`, `zig_emulator_unified_trace.txt`

### Environment Variables

- `EMULATOR_MAX_STEPS=N` - Limit execution steps for controlled testing
- `ZIG_GLOBAL_CACHE_DIR=./zaiko/.zig-cache` - Required for Zig builds

## Common Debugging Scenarios

### Stack Corruption

**Symptom**: Invalid stack pointers or TOPOFSTACK values

**Cause**: Incorrect stack manipulation, missing synchronization

**Solution**: Validate stack state after each operation, check CSTKPTRL/TOPOFSTACK sync

### Memory Corruption

**Symptom**: Execution diverges from C traces unexpectedly

**Cause**: Incorrect memory access, bounds violations, or type mismatches

**Solution**: Use memory integrity verification techniques from `documentation/CRITICAL_DEBUGGING_TECHNIQUE.typ`

### Type Mismatches

**Symptom**: Compilation errors or runtime crashes

**Cause**: LispPTR vs native pointer confusion, signed vs unsigned issues

**Solution**: Always cross-reference C code for correct types and casting

### Performance Issues

**Symptom**: Execution too slow for practical use

**Cause**: Inefficient algorithms, excessive memory allocations

**Solution**: Profile with `EMULATOR_MAX_STEPS`, optimize hot paths

### Submodule Issues

**Symptom**: Changes not committed or merged properly

**Cause**: Incorrect submodule workflow

**Solution**: Enter submodule directory for commits, update pointers from parent

## Debugging Best Practices

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

1. **Always execute with a non-hanging timeout**: always use `timeout --kill-after=N+1 Ns`
1. **Always check C implementation first** when implementing new features
1. **Update documentation** before committing (follow `documentation/core/critical-memory.typ`)
1. **Keep files under 500 lines** for better maintainability
1. **Test incrementally** - don't wait until everything is done
1. **Document findings** in `documentation` for future reference
1. **Use descriptive commit messages** with task IDs when applicable
1. **Follow systematic debugging** from `documentation/CRITICAL_DEBUGGING_TECHNIQUE.typ`
1. **Validate against C traces** before considering implementation complete
1. **Handle edge cases** identified during C code analysis
1. **Profile performance** during development, not just at the end

## Critical Debugging Resources

- **Critical Debugging Techniques**: `documentation/CRITICAL_DEBUGGING_TECHNIQUE.typ`
- **Main README**: `documentation/README.md`
- **Index**: `documentation/reference/index.typ`
- **Architecture**: `documentation/components/vm-core.typ`
- **Glossary**: `documentation/reference/glossary.typ`

---

**Last Updated**: 2026-01-29
**Status**: Debugging workflow established; common gotchas documented
