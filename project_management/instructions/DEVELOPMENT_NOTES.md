# Development Notes

This document captures design decisions and architectural choices made during the development of the Gray-Tunneled Hashing project.

## Project Structure

### Source Layout (`src/`)

The project uses a `src/` layout, which is a modern Python packaging best practice. Benefits include:
- Clear separation between source code and tests
- Prevents accidental imports of development files
- Makes it easier to test the installed package
- Aligns with Python packaging standards (PEP 517/518)

### Package Organization

The package is organized into three main subpackages:

1. **`algorithms/`**: Core algorithm implementations
   - Contains the main `GrayTunneledHasher` class
   - Future: Additional algorithm variants and optimizations

2. **`data/`**: Data generation and processing utilities
   - Synthetic data generators for testing
   - Future: Real dataset loaders and preprocessing

3. **`evaluation/`**: Evaluation metrics and analysis tools
   - Metrics for assessing hashing quality
   - Future: Comprehensive evaluation suite

## Design Decisions

### API Design

The `GrayTunneledHasher` class follows a scikit-learn-like API:
- `fit()`: Train the hasher on data
- `encode()`: Transform embeddings to binary codes
- `decode()`: Reconstruct approximate embeddings (future)
- `evaluate()`: Assess encoding quality

This design:
- Is familiar to users of machine learning libraries
- Separates training from inference
- Allows for future extensions (e.g., `fit_transform()`)

### Placeholder Implementation

The current implementation uses placeholder algorithms:
- Encoding uses simple random projection + thresholding
- Decoding returns zeros (stub)
- This allows testing the API and structure before implementing the full algorithm

### Dependencies

- **numpy**: Core numerical computing (required)
- **pytest**: Testing framework (dev dependency)
- **black**: Code formatting (dev dependency)
- **ruff**: Fast linting (dev dependency)

We keep dependencies minimal to reduce complexity and improve portability.

### Testing Strategy

- Tests are in `tests/` directory at the root level
- Tests use pytest for discovery and execution
- Tests cover:
  - Basic functionality (fit, encode, decode, evaluate)
  - Error handling
  - End-to-end workflows
- Future: Add integration tests, performance tests, and property-based tests

## Future Considerations

### Algorithm Implementation

- The full Gray-Tunneled Hashing algorithm will replace the placeholder
- Need to decide on specific implementation details (e.g., Gray code encoding strategy)
- May need additional dependencies for advanced mathematical operations

### Performance

- Current implementation prioritizes correctness over performance
- Future optimizations may require:
  - NumPy optimizations
  - Cython or Numba for hot paths
  - GPU acceleration for large-scale experiments

### Integration

- Design for easy integration with vector DBs
- Consider adapter pattern for different DB interfaces
- Plan for both direct usage and library integration

