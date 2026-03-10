# Documentation Guidelines

**Date**: 2026-01-29
**Purpose**: Documentation guidelines for Interlisp project

## Critical Documentation (Must-Read Files)

1. Read ./reports/WORK_STATE.md, then ./reports/STEP_COMPARISON_STATUS.md

1. When **compressing or summarizing context**, follow **§7**: Phase 1 (aggressive compression of tool/command/trace/linter outputs only) is mandatory and first; Phase 2 (re-read AGENTS.md, then compress the remainder) only if further compression is needed.

1. **`documentation/core/critical-memory.typ`** - **CRITICAL**: Rules for documentation updates

   - All documentation improvements MUST be emulator-independent in `specifications/`
   - Language-specific details go in `implementations/`
   - **ALWAYS** write documentation using the Typst document format
   - **ALWAYS** update both before committing
   - **ALWAYS** use the command `date` to date entries (when necessary) as YYYY-mm-dd HH:MM

1. **`documentation/core/critical-memory.typ`** - **CRITICAL**: Essential debugging techniques and practices

1. **`documentation/README.md`** - Overview of documentation structure
1. **`specs/plan.md`** - Implementation plan for emulator runner
1. **`specs/tasks.md`** - Task list for emulator runner

## Documentation Updates

**CRITICAL**: Before ANY git commit, follow `documentation/core/critical-memory.typ`:

1. ✅ Review discoveries and learnings
2. ✅ Update general documentation (`specifications/`) with emulator-independent findings
3. ✅ Update language-specific documentation (`implementations/`) with Zig-specific details
4. ✅ Make comprehensive commit with documentation updates

**Example Commit Format**:

```
Update documentation: [Brief description]

CRITICAL: Document [general findings] and [language-specific findings]

General Documentation Updates:
- Updated [file]: [general finding 1]
- Updated [file]: [general finding 2]

Language-Specific Documentation Updates:
- Updated [file]: [language-specific detail 1]
- Updated [file]: [language-specific detail 2]
```

## Documentation Accuracy Warning

### Known Discrepancies

1. **Task Tracking Inaccuracy**: Zig completion shown as 89.2% vs actual 60-70%
2. **Placeholder Not Accounted**: Task completion doesn't reflect numerous TODO implementations
3. **Quality vs Quantity**: Completed tasks may have non-functional implementations
4. **Testing Insufficient**: Completion tracking doesn't assess test coverage

### Verification Requirements

1. **Always inspect source code** before trusting documentation claims
2. **Check for TODO/FIXME markers** as incompleteness indicators
3. **Verify functionality** rather than relying on completion percentages
4. **Use C implementation** as reference for Zig development priorities

## Documentation Structure

### Specifications (Emulator-Independent)

Location: `documentation/specifications/`

These documents describe the Interlisp system and emulator behavior in an implementation-independent way:

- **Data Structures**: Arrays, atoms, cons cells, function headers, number types, sysout format
- **I/O**: File system, keyboard protocol, mouse protocol, network protocol
- **Memory**: Address translation, centralized memory design, garbage collection, memory layout, virtual memory
- **Platform Abstraction**: Implementation choices, required behaviors
- **Validation**: Compatibility criteria, reference behaviors
- **VM Core**: Execution model, execution trace, function calls, interrupt handling, stack management, trace and logging formats, type checking, VM architecture

### Implementations (Language-Specific)

Location: `documentation/implementations/`

These documents contain language-specific implementation details and findings:

- **C Implementation**: Detailed analysis of C emulator behavior, tracing results, debugging sessions
- **Zig Implementation**: Zig-specific implementation notes, debugging findings, parity analysis
- **Lisp Implementation**: Common Lisp implementation details (when developed)

### Core Documentation

Location: `documentation/core/`

- **critical-memory.typ**: Rules for documentation updates (MUST READ)
- Other core documentation files

## Documentation Format

### Typst Format

All documentation MUST be written using the Typst document format. This provides:

- Consistent formatting across all documentation
- Better version control (text-based)
- Easy conversion to PDF when needed
- Structured document organization

### Dating Entries

When adding dated entries to documentation, use the format:

```bash
date  # Outputs: YYYY-mm-dd HH:MM
```

Example: `2026-01-29 14:30`

## Documentation Best Practices

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

## Documentation Resources

### Main Documentation Files

- **Main README**: `documentation/README.md`
- **Critical Debugging Techniques**: `documentation/core/critical-memory.typ.typ`
- **Index**: `documentation/reference/index.typ`
- **Architecture**: `documentation/components/vm-core.typ`
- **Glossary**: `documentation/reference/glossary.typ`

### Specification Files

- **Zig Completion Spec**: `specs/spec.md`
- **Implementation Plan**: `specs/plan.md`
- **Tasks**: `specs/tasks.md`
- **Current State**: `specs/current-state-analysis.md`

### Implementation Notes

- **Zig Implementation**: `documentation/implementations/zig-implementation.typ`
- **C Implementation**: Reference in `maiko/src/`

---

**Last Updated**: 2026-01-29
**Status**: Documentation structure established; accuracy warnings noted
