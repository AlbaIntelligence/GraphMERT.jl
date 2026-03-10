# Development Guidelines

**Date**: 2026-01-29
**Purpose**: Development guidelines for the Interlisp project

## Project Overview

This repository contains the **Interlisp** project, which includes:

- **Maiko - C Implementation in @/maiko_untouched**: Historical baseline with sparse comments. Reference for original implementation details only.
- **Maiko - C Implementation in @/maiko**: Production-ready C implementation with comprehensive documentation. Primary reference for development.
- **Zig Implementation**: Alternative implementation of Maiko in Zig (incomplete)
- **Common Lisp Implementation**: Alternative implementation of Maiko in Common Lisp (targetting Sbcl) (in progress - will be developed after Zig)

## Project Structure

```
Interlisp/
├── documentation/             # Comprehensive but may overstate Zig completion
│   ├── specifications/        # Technical specs (verify against actual state)
│   ├── implementations/       # Implementation-specific notes
│   └── core/
│       └── critical-memory.typ # ⚠️ MUST READ AND ABIDE BY - Documentation rules
├── maiko/                     # C Implementation (PRODUCTION-READY)
│   ├── src/                   # Enhanced with superior documentation
│   └── inc/                   # Production headers
├── maiko_untouched/           # Historical baseline (sparse comments)
├── zaiko/                     # Zig Implementation (INCOMPLETE)
│   ├── src/                   # 60-70% complete, many placeholders
│   └── tests/                 # Basic test structure
├── specs/                     # Task tracking (89.2% - inaccurate)
└── medley/                    # Medley Interlisp system
```

## Current Status (2026-01-29)

### C Implementation State: PRODUCTION READY

- **Status**: Fully functional and production-ready
- **Comments**: Enhanced with comprehensive documentation (superior to maiko_untouched/)
- **Completeness**: 94.5% opcode coverage (242/256 opcodes implemented)
- **Quality**: Production-grade with extensive testing framework

### Zig Implementation State: INCOMPLETE

- **Actual Coverage**: 60-70% (despite 89.2% task completion)
- **Critical Issues**: 245 TODO/FIXME markers (8x more than C)
- **Missing Features**: Floating point (completely stubbed), advanced graphics, I/O subsystems
- **Quality**: Development-grade with numerous placeholder implementations

### Documentation Accuracy Gap

- **Task Tracking**: Overstates Zig completion by 20-30%
- **Specifications**: Assume higher Zig implementation than actually exists
- **Issue**: Completion percentages do not reflect implementation quality

## Implementation Reality Assessment

### C Codebase: Production Reference

**Comment State Analysis**:

- **maiko/**: Contains superior comprehensive documentation with structured headers, confidence levels, and algorithm explanations
- **maiko_untouched/**: Sparse historical baseline for reference only
- **Recommendation**: Use maiko/ as primary development reference

### Zig Codebase: Development Reality

**Completion Gap**:

- **Task Tracking**: Claims 89.2% completion
- **Actual Implementation**: 60-70% coverage with major gaps
- **Evidence**: 245 TODO/FIXME markers indicate significant immaturity vs 31 in C
- **Critical Gaps**: Floating point operations completely stubbed, graphics operations incomplete, I/O systems missing substantial functionality

**Quality Indicators**:

- Placeholder implementations return without action
- Minimal error handling in incomplete modules
- Insufficient test coverage for production use

## Parity Development Guidance

### Priority Areas for Zig Parity

**Critical Missing Functionality**:

1. **Floating Point Operations**: All opcodes currently stubbed
2. **Graphics Pipeline**: Essential BitBLT and drawing operations missing
3. **I/O Subsystems**: File system, device handling, network operations incomplete
4. **Placeholder Elimination**: Replace non-functional implementations

**Quality Requirements for Parity**:

1. **Comprehensive Testing**: Build test coverage matching C implementation
2. **Error Handling**: Implement robust error management across all modules
3. **Performance**: Achieve comparable execution speed to C reference
4. **Documentation**: Add algorithm explanations for completed implementations

## Working Guidelines

### 1. Code Organization

- **File Size Limit**: Keep files under 500 lines (user preference)
- **Modular Structure**: Split large files into logical modules
- **Naming**: Follow existing conventions (see `zaiko/src/`)

### 1.1 Submodules (CRITICAL)

This repository includes git submodules (notably `maiko/` and `medley/`).

- **Default**: Treat submodules as **read-only** unless the user explicitly asks to modify them.
- **Committing**:
  - **Do not commit inside submodules** unless the user explicitly requests it.
  - Superproject commits may update submodule pointers _only if explicitly intended_.
- **Staging**: Paths inside submodules cannot be staged from the superproject (e.g. `git add maiko/...` will fail). Enter the submodule repo if changes are required.

### 1.2 Generated Artifacts (never commit)

Avoid committing machine/local outputs that create noisy diffs:

- **Execution logs / traces**: `*_execution_log*.txt`, `c_*_trace*.txt`, `zig_*_trace*.txt`
- **Python caches**: `__pycache__/`, `*.pyc`
- **Build outputs**: Zig `zig-out/`, `**/.zig-cache/`, C/CMake build directories
- **Binary documents**: generated PDFs (commit the Typst source instead)

### 1.3 File System Access Restrictions (CRITICAL)

**SECURITY REQUIREMENT**: Agents must not access any files or folders outside the project directory containing this AGENTS.md file.

- **Prohibited**: Accessing `/tmp`, `/var`, `/home`, or any system directories
- **Prohibited**: Creating files outside the project workspace
- **Allowed**: Only files and folders within the Interlisp project directory
- **Reason**: Security, isolation, and reproducibility requirements

**Violation will result in immediate termination of the session.**

### 3. C Implementation as Reference

**CRITICAL RULE**: The C implementation (`maiko/src/`) is the ultimate reference.

- **Use maiko/ as primary reference** - Contains superior documentation to maiko_untouched/
- **Leverage comprehensive headers** - Use confidence levels and testing guidance
- **Preserve documentation quality** - Maintain structured explanations
- **Cross-reference maiko_untouched/** - Only for historical context

- If comments/documentation conflict with C code, **trust the C code**
- If documentation is incorrect, **update the documentation**
- When debugging, use the C emulator with debug statements to understand behavior

## Build System

- **Zig**: Uses Zig build system (`build.zig`)
- **C**: Uses traditional Makefile/CMake. Executable built with `./medley/scripts/build/build-c-emulator.sh --display-backend sdl --build-system cmake --force`
- **Nix**: Project uses Nix for managing dependencies (see `shell.nix`)

## Code Style

- **Indentation**: 2 spaces (R conventions)
- **Python Version**: 3.12
- **Zig Version**: 0.15.2+

## Common Tasks

### Adding New Opcodes

1. **Cross-reference C implementation** - Study `maiko/src/` for algorithm and semantics
2. **Add opcode definition** to `zaiko/src/vm/dispatch/opcode.zig`
3. **Implement handler** in appropriate module in `zaiko/src/vm/opcodes/`
4. **Add to dispatch switch** in `zaiko/src/vm/dispatch/execution.zig`
5. **Systematic testing** - Use parity debugging techniques from `documentation/CRITICAL_DEBUGGING_TECHNIQUE.typ`
6. **Update documentation** in `documentation/specifications/instruction-set/opcodes.typ`
7. **Add regression tests** if applicable

### File Splitting

When a file exceeds 500 lines:

1. Identify logical groupings
2. Create new modules in appropriate subdirectory
3. Update re-export file to import and re-export from new modules
4. Ensure all imports are updated
5. Test compilation

## Contact & Support

- **Project Documentation**: See `documentation/README.md`
- **Implementation Status**: See `specs/current-state-analysis.md`
- **Task Tracking**: See `specs/tasks.md`

---

**Last Updated**: 2026-01-29
**Status**: C implementation production-ready with comprehensive documentation; Zig implementation incomplete (60-70% actual coverage) with significant quality gaps
