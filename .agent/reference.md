# Reference and Quick Reference

**Date**: 2026-01-29
**Purpose**: Quick reference guide for Interlisp project

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

## Resources

### Documentation

- **Main README**: `documentation/README.md`
- **Critical Debugging Techniques**: `documentation/CRITICAL_DEBUGGING_TECHNIQUE.typ`
- **Index**: `documentation/reference/index.typ`
- **Architecture**: `documentation/components/vm-core.typ`
- **Glossary**: `documentation/reference/glossary.typ`

### Specifications

- **Zig Completion Spec**: `specs/spec.md`
- **Implementation Plan**: `specs/plan.md`
- **Tasks**: `specs/tasks.md`
- **Current State**: `specs/current-state-analysis.md`

### Implementation Notes

- **Zig Implementation**: `documentation/implementations/zig-implementation.typ`
- **C Implementation**: Reference in `maiko/src/`

## Quick Reference

### Important Paths

- Zig source: `zaiko/src/`
- Zig tests: `zaiko/tests/`
- C reference: `maiko/src/`
- Documentation: `documentation/`
- Specs: `specs/`

### Important Constants

- `IFPAGE_KEYVAL`: `0x15e3` (CRITICAL: Must match C implementation)
- `IFPAGE_ADDRESS`: `512` bytes from start of sysout file
- `BYTESPER_PAGE`: `512` bytes (256 DLwords)
- `STK_OFFSET`: `0x00010000` (DLword offset for stack area)
- `FRAMESIZE`: `10` DLwords (20 bytes)

### Build Commands

```bash
# Build Zig emulator
cd zaiko
zig build

# Run Zig emulator
zig build run -- medley/internal/loadups/starter.sysout

# Run tests
zig build test

# Build C emulator (if needed)
cd ../../..
medley/scripts/build/build-c-emulator.sh
```

## Troubleshooting Common Issues

### Memory Corruption

**Symptom**: Execution diverges from C traces unexpectedly

**Cause**: Incorrect memory access, bounds violations, or type mismatches

**Solution**: Use memory integrity verification techniques from `documentation/CRITICAL_DEBUGGING_TECHNIQUE.typ`

### Stack Corruption

**Symptom**: Invalid stack pointers or TOPOFSTACK values

**Cause**: Incorrect stack manipulation, missing synchronization

**Solution**: Validate stack state after each operation, check CSTKPTRL/TOPOFSTACK sync

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

## Best Practices

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

### For Documentation Accuracy

1. **Verify completion claims** - Zig is 60-70% complete, not 89.2%
2. **Cross-reference implementations** - Use C as reference for Zig development
3. **Check for placeholder code** - TODO markers indicate incomplete implementations
4. **Maintain reality checks** - Regular assessment vs documented assumptions

### General Practices

1. **Always check C implementation first** when implementing new features
2. **Update documentation** before committing (follow `documentation/core/critical-memory.typ`)
3. **Keep files under 500 lines** for better maintainability
4. **Test incrementally** - don't wait until everything is done
5. **Document findings** in `documentation` for future reference
6. **Use descriptive commit messages** with task IDs when applicable
7. **Follow systematic debugging** from `documentation/CRITICAL_DEBUGGING_TECHNIQUE.typ`
8. **Validate against C traces** before considering implementation complete
9. **Handle edge cases** identified during C code analysis
10. **Profile performance** during development, not just at the end

## Contact & Support

- **Project Documentation**: See `documentation/README.md`
- **Implementation Status**: See `specs/current-state-analysis.md`
- **Task Tracking**: See `specs/tasks.md`

---

**Last Updated**: 2026-01-29
**Status**: Reference guide complete; quick reference established
