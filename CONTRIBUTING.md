# Contributing to TTFEMesh

Thank you for your interest in contributing to TTFEMesh! This document provides guidelines and instructions for contributing to the project.

## Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR-USERNAME/TT-FEMesh.git
   cd TT-FEMesh
   ```
3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. Install development dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```

## Development Workflow

1. Create a new branch for your feature/fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes
3. Run tests:
   ```bash
   make all
   ```
4. Commit your changes with a descriptive commit message
5. Push to your fork
6. Create a Pull Request

## Code Style

- Follow PEP 8 guidelines
- Use type hints for all function parameters and return values
- Write docstrings for all public functions and classes
- Keep functions focused and small
- Write unit tests for new functionality

## Testing

- Write tests for all new functionality
- Ensure all tests pass before submitting a PR
- Maintain or improve test coverage
- Use pytest for testing

## Documentation

- Update docstrings for any modified functions/classes
- Update relevant documentation in the `docs/` directory
- Add examples for new features
- Keep the README.md up to date

## Pull Request Process

1. Update the README.md with details of changes if needed
2. Update the documentation if needed
3. The PR will be merged once you have the sign-off of at least one maintainer
4. Make sure all CI checks pass

## Development Tools

The project uses several tools to maintain code quality:

- `pytest` for testing
- `flake8` for linting
- `mypy` for type checking
- `black` for code formatting
- `isort` for import sorting

You can run all checks using:
```bash
make all
```

## Docker Development Environment

A Docker development environment is provided for consistent development across platforms:

```bash
# Build the development container
docker-compose build

# Run the development container
docker-compose run --rm dev

# Run tests in the container
docker-compose exec dev make all
```

## Questions and Support

If you have any questions or need help, please:
1. Check the existing documentation
2. Open an issue for bugs or feature requests
3. Contact the maintainers

## License

By contributing to TTFEMesh, you agree that your contributions will be licensed under the project's MIT License. 