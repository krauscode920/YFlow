# Contributing to YFlow

We love your input! We want to make contributing to YFlow as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## **CRITICAL: Branch Structure and Submission Process**

**ALL CONTRIBUTIONS MUST BE SUBMITTED TO THE CONTRIBUTE BRANCH ONLY**

### **Branch Structure**
- **Main Branch**: https://github.com/krauscode920/YFlow/tree/main (Protected - No direct contributions)
- **Contribute Branch**: https://github.com/krauscode920/YFlow/tree/Contribute (All PRs go here)

### **Contribution Process**
1. **Fork the repository** from the main branch
2. **Create your feature branch** from the **Contribute branch**
3. **Make your changes** following our naming conventions and governance rules
4. **Submit pull request** to the **Contribute branch ONLY**
5. **Code review** and approval by maintainers
6. **Merge to main** branch upon approval

**Pull requests submitted to any branch other than Contribute will be automatically rejected.**

## Architecture Naming Convention & Governance

YFlow follows a strict naming convention and governance model to maintain consistency and prevent architectural fragmentation.

### **Fixed Architecture Names (CANNOT BE CHANGED)**

These architectures have established names that are permanent and cannot be modified:

- **YFormers** - All Transformer architectures (GPT, BERT, T5, etc.)
- **YLSTM** - Long Short-Term Memory networks
- **YQuence** - Standard Recurrent Neural Networks (RNN)
- **BiYQuence** - Bidirectional Recurrent Neural Networks

### **Governance Principles**

#### **No Parallel Architectures Rule**
- There can only be ONE implementation per architecture type
- Contributors **CANNOT** create alternative versions (e.g., "YFormers2", "FastYFormers", "YBert")
- All improvements must be made to the existing architecture

#### **Working with Existing Architectures**

If you want to improve YFormers, YLSTM, YQuence, or BiYQuence:

1. **Work within the existing codebase** - modify the existing architecture, don't create alternatives
2. **Submit pull requests** to the **Contribute branch only**
3. **Pass code review** and maintainer approval
4. **Maintain backward compatibility** where possible
5. **Provide comprehensive tests** and documentation

#### **Creating New Architectures**

Contributors are welcome to create entirely **new architectures**:

- **Complete naming freedom** for new architectures (no Y-prefix requirement)
- Novel architectures are subject to evaluation and approval
- Must demonstrate clear innovation over existing architectures
- Must follow YFlow's core principles and device abstraction

### **Examples**

#### ✅ **Allowed Contributions:**
```python
# Improving existing YFormers
class YFormers(TransformerModel):
    def __init__(self):
        # Add new attention mechanism to existing architecture
        self.improved_attention = NewAttentionVariant()

# Creating new architectures (free naming)
class ConvolutionNet(Model):
    """Novel convolutional architecture"""
    pass

class HybridModel(Model):
    """New hybrid CNN-Transformer architecture"""
    pass

class GraphNet(Model):
    """Graph neural network architecture"""
    pass
```

#### ❌ **Not Allowed:**
```python
# Creating parallel transformer implementations
class YFormers2(Model):  # ❌ No parallel architectures
    pass

class FastYFormers(Model):  # ❌ No alternative implementations
    pass

class YBert(Model):  # ❌ BERT variants belong in YFormers
    pass

class YGpt(Model):  # ❌ GPT variants belong in YFormers
    pass
```

## Critical Contribution Guidelines

### **External Library Policy - STRICTLY ENFORCED**

- **ABSOLUTELY FORBIDDEN**: Usage of any existing deep learning libraries
  - No TensorFlow, PyTorch, JAX, Keras, etc.
  - No scikit-learn for ML operations
  - No importing from or dependencies on other ML frameworks
  - Pull requests containing such imports will be **automatically rejected**

### **Naming Convention Requirements**

#### **Fixed Architecture Names (Protected)**
The following architecture names are **permanently fixed** and cannot be changed:
- **YFormers** - All Transformer architectures
- **YLSTM** - Long Short-Term Memory networks  
- **YQuence** - Standard Recurrent Neural Networks
- **BiYQuence** - Bidirectional Recurrent Neural Networks

#### **New Architecture Naming**
- **Complete freedom** to name new architectures
- No Y-prefix requirement for new architectures
- Choose descriptive, clear names for your implementations
- Names are subject to approval during review process

### **Device Abstraction Requirement**

All new layers and architectures must:
- Use YFlow's device abstraction system
- Support both CPU and GPU execution
- Follow the established device management patterns

## Priority Areas for Contribution

### **Immediate High-Priority Needs**

1. **GPU Testing and Validation**
   - Test YFormers and core layers on actual GPU hardware
   - Performance benchmarking against reference implementations
   - Memory efficiency optimization
   - GPU-specific bug fixes

2. **YFormers Architecture Extensions**
   - Vision Transformers (ViT) within YFormers
   - Sparse attention mechanisms
   - Advanced positional encodings
   - Cross-modal attention variants

3. **Core Architecture Implementation**
   - **YLSTM**: Complete LSTM implementation
   - **YQuence**: RNN architecture
   - **BiYQuence**: Bidirectional RNN
   - New CNN-based architectures

### **Medium Priority**

4. **Advanced Layer Implementations**
   - Convolutional layers and architectures
   - Advanced pooling operations
   - Normalization variants
   - Custom activation functions

5. **Training Infrastructure**
   - Data loaders and preprocessing
   - Training loop utilities
   - Metrics and evaluation
   - Learning rate schedulers

6. **Documentation and Examples**
   - Tutorial notebooks for each architecture
   - Comprehensive API documentation
   - Educational content explaining implementations
   - Performance comparison studies

### **Lower Priority**

7. **Optimization and Performance**
   - Memory usage optimization
   - Computational efficiency improvements
   - Profiling and benchmarking tools

## Development Guidelines

### **Code Style**

- Use 4 spaces for indentation
- Follow PEP 8 guidelines
- Include comprehensive docstrings for all public methods and classes
- Write clear, descriptive variable and function names
- Add type hints where appropriate

### **Testing Requirements**

- All new features must include unit tests
- GPU functionality should include both CPU and GPU tests
- Integration tests for complete architectures
- Performance tests for critical components

### **Documentation Standards**

- Update README.md if adding new major features
- Include docstrings following NumPy/Google style
- Add usage examples for new architectures
- Document any breaking changes clearly

## Specific Contribution Types

### **Bug Fixes**
- Create issue first describing the bug
- Include minimal reproduction case
- Submit fix to Contribute branch
- Include tests that verify the fix

### **New Architecture Proposals**
- Open issue with architecture proposal
- Include motivation and use cases
- Provide implementation plan
- Get approval before starting work

### **Performance Improvements**
- Include benchmarks showing improvement
- Ensure no regression in existing functionality
- Document any trade-offs

## Getting Started

### **Good First Issues**
Look for issues labeled:
- "good first issue"
- "help wanted" 
- "documentation"
- "testing needed"

### **Setting Up Development Environment**

1. Fork the repository from main branch
2. Clone your fork
3. Create branch from Contribute branch:
   ```bash
   git checkout Contribute
   git checkout -b your-feature-branch
   ```
4. Install development dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

## Review Process

1. **Automated Checks**: All PRs run automated tests and style checks
2. **Architecture Review**: New architectures undergo design review
3. **Code Review**: Maintainer review for code quality and adherence to guidelines
4. **Testing**: Comprehensive testing including edge cases
5. **Documentation**: Ensure all changes are properly documented

## License

By contributing, you agree that your contributions will be licensed under the project's MIT License.

## Questions?

- Open an issue for general questions
- Check existing issues and pull requests first
- Tag maintainers for urgent questions

## Version

This contributing guide is for YFlow version 0.2.0 and reflects the unified governance model with YFormers integration.

Thank you for contributing to YFlow! Together we're building a comprehensive, educational deep learning framework from the ground up.
