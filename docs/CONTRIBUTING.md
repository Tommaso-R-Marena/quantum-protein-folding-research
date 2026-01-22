# Contributing to QuantumFold-Advantage

Thank you for considering contributing! This document provides guidelines for contributing to the project.

## Development Setup

### 1. Fork and Clone

```bash
# Fork on GitHub, then clone your fork
git clone https://github.com/YOUR-USERNAME/quantum-protein-folding-research.git
cd quantum-protein-folding-research

# Add upstream remote
git remote add upstream https://github.com/Tommaso-R-Marena/quantum-protein-folding-research.git
```

### 2. Install Development Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### 3. Create Feature Branch

```bash
git checkout -b feature/your-feature-name
```

## Code Standards

### Style Guide

We follow PEP 8 with these specifics:
- **Line length**: 100 characters (relaxed from 79)
- **Quotes**: Double quotes for docstrings, either for code
- **Imports**: Grouped (stdlib, third-party, local) with `isort`
- **Type hints**: Required for public APIs

### Formatting

```bash
# Auto-format with black
black src/ tests/

# Sort imports
isort src/ tests/

# Check style
flake8 src/ tests/

# Type checking
mypy src/
```

### Docstrings

Use Google-style docstrings:

```python
def function_name(param1: int, param2: str) -> bool:
    """Brief description.
    
    Longer description if needed.
    
    Args:
        param1: Description of param1
        param2: Description of param2
    
    Returns:
        Description of return value
    
    Raises:
        ValueError: When input is invalid
    
    Example:
        >>> function_name(42, "test")
        True
    """
```

## Testing

### Writing Tests

- Place tests in `tests/` directory
- Name test files `test_<module>.py`
- Use descriptive test names: `test_<what>_<condition>_<expected>`
- Group related tests in classes
- Use fixtures for common setup

Example:

```python
import pytest

class TestVQESolver:
    """Tests for VQE solver."""
    
    def test_initialization_creates_circuit(self):
        """VQE initialization should create valid circuit."""
        # Arrange
        hamiltonian = SparsePauliOp.from_list([('Z', 1.0)])
        
        # Act
        solver = VQESolver(hamiltonian, n_qubits=1)
        
        # Assert
        assert solver.ansatz is not None
        assert solver.n_params > 0
```

### Running Tests

```bash
# All tests
pytest tests/

# With coverage
pytest --cov=quantum_protein_folding tests/

# Specific file
pytest tests/test_vqe.py -v

# Skip slow tests
pytest -m "not slow" tests/

# Run in parallel
pytest -n auto tests/
```

### Test Markers

```python
@pytest.mark.slow
def test_large_system():
    """This test is slow."""
    pass

@pytest.mark.integration
def test_full_pipeline():
    """End-to-end integration test."""
    pass
```

## Pull Request Process

### 1. Before Submitting

- [ ] All tests pass: `pytest tests/`
- [ ] Code is formatted: `black src/ tests/`
- [ ] No linter errors: `flake8 src/ tests/`
- [ ] Type hints pass: `mypy src/`
- [ ] Docstrings are complete
- [ ] New features have tests (aim for >90% coverage)
- [ ] Update documentation if needed

### 2. Commit Messages

Follow conventional commits:

```
type(scope): brief description

Longer description if needed.

References #issue-number
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Formatting, no code change
- `refactor`: Code restructuring
- `test`: Adding tests
- `perf`: Performance improvement
- `chore`: Maintenance

Examples:
```
feat(vqe): add SPSA optimizer support
fix(hamiltonian): correct sign in contact term
docs(readme): update installation instructions
test(qaoa): add convergence tests
```

### 3. Pull Request Template

```markdown
## Description
Brief description of changes.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
Describe tests added or modified.

## Checklist
- [ ] Tests pass
- [ ] Code is formatted
- [ ] Docstrings updated
- [ ] Documentation updated
```

### 4. Review Process

1. Automated checks must pass (CI/CD)
2. At least one maintainer approval required
3. Address review comments
4. Squash commits before merge (if requested)

## Contributing Areas

### High Priority

1. **Algorithm Improvements**
   - New ansatz designs
   - Improved initialization strategies
   - Noise mitigation techniques

2. **Performance Optimization**
   - Circuit compilation optimizations
   - Parallel execution
   - GPU acceleration for classical parts

3. **New Features**
   - Additional contact potentials
   - Side-chain modeling
   - Hybrid quantum-classical methods

4. **Documentation**
   - Tutorial notebooks
   - API examples
   - Performance benchmarks

### Medium Priority

5. **Testing**
   - Edge case coverage
   - Property-based testing
   - Integration tests

6. **Analysis Tools**
   - New metrics
   - Visualization improvements
   - Statistical analysis

### Good First Issues

Look for issues labeled `good-first-issue`:
- Documentation improvements
- Adding examples
- Test coverage
- Minor bug fixes

## Code Review Guidelines

### For Authors

- Keep PRs focused and small
- Provide context in description
- Respond to comments promptly
- Be open to feedback

### For Reviewers

- Be constructive and specific
- Ask questions rather than demand changes
- Approve once comments are addressed
- Focus on correctness, readability, maintainability

## Release Process

(For maintainers)

1. Update version in `setup.py`
2. Update CHANGELOG.md
3. Tag release: `git tag v1.0.0`
4. Push tags: `git push --tags`
5. Create GitHub release with notes
6. Publish to PyPI (if applicable)

## Questions?

Open an issue with label `question` or email the maintainers.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
