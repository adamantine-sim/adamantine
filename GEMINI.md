# Adamantine Development Guide

This document provides foundational mandates for development within the Adamantine codebase. Adhere to these standards to ensure consistency and technical integrity.

## Project Overview
Adamantine is a thermomechanical code for additive manufacturing, leveraging deal.II, ArborX, Trilinos, and Kokkos. It is a high-performance C++ project with support for GPU acceleration (via Kokkos) and distributed memory parallelism (via deal.II/MPI).

## Tech Stack
- **Language**: C++17 or later.
- **Build System**: CMake (minimum version 3.15).
- **Package Management**: Nix (Flakes enabled).
- **Core Libraries**: deal.II, ArborX, Trilinos, Kokkos, Boost.

## Development Mandates

### 1. Build & Environment
- **Environment**: Always use the Nix development shell (`nix develop`) or `direnv allow` to ensure all dependencies are available.
- **Compilation**: Prefer `Ninja` as the generator.

### 2. Coding Standards
- **Formatting**: Rigorously follow the `.clang-format` configuration (LLVM-based, Allman braces). Use the `./indent` script at the root to format the entire codebase.
- **Naming Conventions**:
    - Files: `.cc` for implementation, `.hh` for headers, `.templates.hh` for template implementations.
    - Consistency: Follow existing naming patterns in `source/` (e.g., PascalCase for classes, snake_case for some utility functions).
- **GPU Portability**: Use Kokkos for performance-portable kernels. Be mindful of Host/Device memory spaces.

### 3. Testing
- **Verification**: Every new feature or bug fix MUST include corresponding tests in the `tests/` directory.
- **Execution**: Run tests using `ctest` within the build directory.
- **Test Framework**: The project uses a custom test runner (see `tests/main.cc`). Tests are structured as standalone `.cc` files in the `tests/` directory.

### 4. Project Structure
- `application/`: Main entry point and application-level logic.
- `source/`: Core library implementation.
- `tests/`: Test suite and test data.
- `cmake/`: Custom CMake modules for dependency detection.
- `nix/`: Nix-specific configuration.

## Workflow Instructions for Gemini
- **Research**: When investigating bugs, check `source/` for implementation and `tests/` for reproduction cases.
- **Execution**: 
    - Always verify changes by compiling (e.g., `ninja`) and running `ctest`.
    - Always run `./indent` after modifying C++ files.
    - If adding a new source file, ensure it is added to the appropriate `CMakeLists.txt` in `source/` or `application/`.
