# Contributing to YFlow

We love your input! We want to make contributing to YFlow as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## Development Process

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

### Pull Requests

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code follows the project style guidelines.
6. Issue that pull request!

### Issues

We use GitHub issues to track public bugs and feature requests. Report a bug by opening a new issue.

## Priority Areas for Contribution

We would especially welcome contributions in these areas:

1. **GPU Testing and Optimization**
   - Testing the existing GPU implementation on various hardware
   - Performance optimization for GPU operations
   - Memory efficiency improvements

2. **Transformer Architecture (YFormers)**
   - Implementation of attention mechanisms
   - Encoder and decoder blocks
   - Position encoding variants
   - Memory-efficient transformer variants

3. **Documentation and Examples**
   - Usage examples for different components
   - Architecture diagrams
   - API documentation
   - Tutorial notebooks

4. **Additional Layer Implementations**
   - Convolutional layers
   - Recurrent layers
   - Advanced pooling operations
   - Custom layers for specific applications

## Critical Contribution Guidelines

### Naming Convention Requirements

- **MANDATORY**: All modules, classes, and major components MUST start with the letter 'Y'
  - Example: `YQuence` for RNN implementation, `Ystm` for LSTM implementation
  - This naming convention is non-negotiable and pull requests that don't follow it will be rejected

### External Library Policy

- **STRICTLY FORBIDDEN**: Usage of any existing deep learning libraries (TensorFlow, PyTorch, JAX, etc.)
  - The purpose of this project is to build everything from scratch
  - No importing from or dependencies on other ML frameworks is allowed
  - Pull requests containing such imports will be automatically rejected

### Code Style

- Use 4 spaces for indentation
- Follow PEP 8 guidelines
- Include docstrings for all public methods and classes
- Write clear, descriptive variable and function names

## License

By contributing, you agree that your contributions will be licensed under the project's MIT License.

## Getting Started

If you're new to the project, consider starting with issues labeled "good first issue" or "help wanted".

Thank you for contributing to YFlow!
