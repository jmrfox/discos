# Contributing to GenCoMo

Thank you for your interest in contributing to GenCoMo! This document provides guidelines for contributing to the project.

## Development Setup

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/yourusername/gencomo.git
   cd gencomo
   ```
3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. Install in development mode:
   ```bash
   pip install -e ".[dev]"
   ```

## Code Style

We follow these coding standards:

- **PEP 8**: Python code style guide
- **Black**: Code formatting (line length 88)
- **Flake8**: Linting
- **Type hints**: Use type annotations where appropriate

Format and check your code:
```bash
black gencomo/ tests/ examples/
flake8 gencomo/ tests/ examples/
mypy gencomo/
```

## Testing

All contributions should include tests. We use pytest for testing.

Run tests:
```bash
pytest tests/
```

Run tests with coverage:
```bash
pytest --cov=gencomo tests/
```

## Documentation

- Use docstrings for all public functions and classes
- Follow Google docstring style
- Update README.md for significant changes
- Add examples for new features

## Submitting Changes

1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes
3. Add tests for new functionality
4. Ensure all tests pass
5. Commit with a clear message:
   ```bash
   git commit -m "Add feature: description of what was added"
   ```
6. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
7. Create a Pull Request on GitHub

## Pull Request Guidelines

- Clearly describe what your PR does
- Reference any related issues
- Ensure all CI checks pass
- Add documentation for new features
- Keep PRs focused and atomic

## Reporting Issues

When reporting bugs or requesting features:

1. Check if the issue already exists
2. Use the issue templates when available
3. Provide minimal reproduction examples
4. Include system information (OS, Python version, etc.)

## Areas for Contribution

We welcome contributions in these areas:

### High Priority
- **Performance optimization**: Faster mesh processing and simulation
- **Memory efficiency**: Reduce memory usage for large meshes
- **Visualization**: Better plotting and 3D visualization tools
- **File format support**: More mesh formats (PLY, OBJ, etc.)

### Medium Priority
- **Advanced biophysics**: More ion channel models
- **Parallel processing**: Multi-core support for large simulations
- **Integration**: Bridges to NEURON, Brian2, or other simulators
- **Examples**: More example scripts and tutorials

### Documentation
- **Tutorials**: Step-by-step guides for common workflows
- **API documentation**: Complete docstring coverage
- **Theory guide**: Mathematical background and algorithms
- **Performance guide**: Optimization tips and benchmarks

## Code Organization

```
gencomo/
├── core.py          # Core data structures
├── mesh.py          # Mesh processing utilities  
├── slicer.py        # Z-axis slicing
├── regions.py       # Region detection
├── graph.py         # Graph construction
├── ode.py           # ODE system
├── simulation.py    # Simulation engine
├── cli.py           # Command-line interface
└── __init__.py      # Package initialization

tests/
├── test_core.py     # Core functionality tests
├── test_mesh.py     # Mesh processing tests
├── test_slicer.py   # Slicing tests
└── ...

examples/
├── basic_example.py    # Basic usage example
├── advanced_example.py # Advanced features
└── ...
```

## Release Process

Maintainers handle releases using semantic versioning:

- **MAJOR** (x.0.0): Breaking API changes
- **MINOR** (0.x.0): New features, backwards compatible
- **PATCH** (0.0.x): Bug fixes, backwards compatible

## Questions?

Feel free to:
- Open an issue for questions
- Start a discussion on GitHub Discussions
- Contact the maintainers directly

Thank you for contributing to GenCoMo!
