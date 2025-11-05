# Contributing Guidelines

Thank you for your interest in contributing to the Gray-Tunneled Hashing project!

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on technical merit

## Development Workflow

### Before Committing

1. **Run Tests**: Always run the test suite before committing:
   ```bash
   pytest
   ```
   Ensure all tests pass.

2. **Code Style**: Follow the project's code style:
   - Use `black` for formatting (configured in `pyproject.toml`)
   - Use `ruff` for linting (configured in `pyproject.toml`)
   - Run `black .` and `ruff check .` before committing

3. **Code Quality**: 
   - Write clear, readable code
   - Add docstrings for public functions and classes
   - Add type hints where appropriate
   - Write tests for new functionality

### Commit Messages

- Use clear, descriptive commit messages
- You may use Conventional Commits style (e.g., `feat:`, `chore:`, `test:`, `docs:`)
- Group related changes into coherent commits
- Never commit code that fails tests or is obviously broken

### Branching Strategy

- Create feature branches for new functionality
- Use descriptive branch names (e.g., `feature/algorithm-optimization`, `fix/hamming-distance-bug`)
- Keep branches focused on a single feature or fix

### Pull Requests

- Provide clear description of changes
- Reference related issues or tasks
- Ensure all tests pass
- Request review from maintainers

## Testing

- Write tests for new functionality
- Maintain or improve test coverage
- Tests should be in the `tests/` directory
- Use descriptive test function names starting with `test_`

## Documentation

- Update documentation when adding new features
- Keep README files up to date
- Add docstrings to new functions and classes
- Update `project_management/sprints/sprint-log.md` when completing sprints

## Questions?

If you have questions, please open an issue or contact the maintainers.

