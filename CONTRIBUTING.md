# Contributing to YFlow

Thank you for your interest in contributing to YFlow! We're building a corporate-free deep learning framework from scratch, and we need talented developers who share our vision of independence and transparency in AI infrastructure.

## **CRITICAL: Submission Process**

### **Branch Structure**

**ALL PULL REQUESTS MUST GO TO THE CONTRIBUTE BRANCH**

- **Main Branch**: https://github.com/krauscode920/YFlow/tree/main  
  → Protected, production code only  
  → No direct contributions accepted

- **Contribute Branch**: https://github.com/krauscode920/YFlow/tree/Contribute  
  → All PRs submitted here  
  → Code review and testing happens here  
  → Merged to main after approval

### **Submission Workflow**

1. **Fork** the repository from main branch
2. **Create feature branch** from the Contribute branch:
   ```bash
   git checkout Contribute
   git checkout -b feature/your-feature-name
   ```
3. **Develop** your feature following our guidelines
4. **Test** thoroughly (include unit tests)
5. **Submit PR** to **Contribute branch ONLY**
6. **Code review** by maintainers
7. **Merge** to main upon approval

**PRs submitted to main or any other branch will be automatically closed.**

## Philosophy and Core Principles

### **Independence Above All**

YFlow exists because we refuse to depend on corporate-owned ML frameworks. This is non-negotiable:

- **ABSOLUTELY FORBIDDEN**: TensorFlow, PyTorch, JAX, Keras, scikit-learn (for ML ops)
- **Why**: We're building completely independent infrastructure
- **Consequence**: PRs with these imports are **immediately rejected** without review

### **Production-Ready, Not Just Educational**

While YFlow's implementation is transparent and educational, it's designed for real production workloads:

- Code must be production-quality (not just "learning examples")
- Performance matters (benchmarks required for critical paths)
- API stability is crucial (no breaking changes without strong justification)

### **Community-Governed Architecture**

We maintain strict governance to prevent fragmentation while encouraging innovation.

## Architecture Governance

### **Fixed Architectures (Cannot Be Duplicated)**

These architectures have permanent names and **only one implementation** is allowed:

- **YFormers**: All transformer variants (GPT, BERT, T5, ViT, etc.)
- **YSTM**: LSTM networks  
- **YQuence**: Standard RNN
- **BiYQuence**: Bidirectional RNN

**What This Means:**

✅ **You CAN**: Improve existing architectures, add features, optimize performance  
❌ **You CANNOT**: Create "YFormers2", "FastYFormers", "YFormersLite", parallel implementations

### **Contributing to Fixed Architectures**

If you want to enhance YFormers, YSTM, YQuence, or BiYQuence:

1. **Work within existing code** - modify the architecture, don't fork it
2. **Maintain backward compatibility** where possible
3. **Add comprehensive tests** for new features
4. **Document changes** thoroughly
5. **Submit to Contribute branch**

Example - Improving YFormers:
```python
# ✅ CORRECT: Enhancing existing YFormers
class YFormers(TransformerModel):
    def __init__(self, ...):
        super().__init__()
        # Add new attention variant
        self.sparse_attention = SparseAttentionHead(...)
        
    def forward_with_sparse_attention(self, x):
        # New method adding functionality
        return self.sparse_attention(x)

# ❌ WRONG: Creating parallel implementation
class YFormers2(TransformerModel):  # Will be rejected
    pass

class BetterYFormers(Model):  # Will be rejected
    pass
```

### **Creating New Architectures**

You have **complete freedom** to create novel architectures:

- **No naming restrictions** (Y-prefix not required)
- Choose descriptive, clear names
- Must demonstrate clear innovation or different use case
- Subject to architecture review for approval

Examples:
```python
# ✅ All acceptable new architectures
class ConvolutionalNet(Model):
    """CNN implementation for YFlow"""
    pass

class GraphNeuralNetwork(Model):
    """GNN for graph-structured data"""
    pass

class HybridTransformerCNN(Model):
    """Novel hybrid architecture"""
    pass

class StateSpaceModel(Model):
    """Mamba-style state space model"""
    pass
```

## Priority Contribution Areas

### **🔥 Critical (Immediate Impact)**

1. **Performance Optimization**
   - CUDA kernel optimization for attention, matmul
   - Memory efficiency improvements
   - Profiling and benchmarking tools
   - Gradient checkpointing implementation

2. **Advanced YFormers Features**
   - Vision Transformer (ViT) within YFormers
   - Sparse attention mechanisms
   - Flash Attention-style optimizations
   - Cross-modal attention variants
   - Efficient KV-cache for generation

3. **Production Infrastructure**
   - Model quantization (int8, int4)
   - ONNX export for deployment
   - Model pruning utilities
   - Distributed training support (DDP)

### **⚡ High Priority**

4. **Training Utilities**
   - Advanced learning rate schedulers (cosine, warmup)
   - Gradient accumulation
   - Mixed precision training (fp16/bf16)
   - Training metrics and logging (WandB integration)
   - Early stopping and model selection

5. **Data Pipeline**
   - Advanced data augmentation
   - Efficient tokenization utilities
   - Multi-worker DataLoader
   - Streaming dataset support

6. **New Architectures**
   - CNN implementations (complete convolutional layers)
   - GRU (alternative to LSTM)
   - Bidirectional LSTM enhancement
   - State space models (Mamba-style)

### **📚 Medium Priority**

7. **Documentation and Examples**
   - Tutorial notebooks for each architecture
   - Complete API documentation
   - Performance comparison studies (vs PyTorch)
   - Production deployment guides
   - Educational content on implementation details

8. **Developer Experience**
   - Model surgery tools (freeze/unfreeze layers)
   - Layer visualization
   - Gradient checking utilities
   - Better error messages

## Technical Requirements

### **Device Abstraction (Mandatory)**

All new code must use YFlow's device abstraction:

```python
from yflow.core.device import DeviceContext, is_gpu_available

class MyNewLayer(Layer):
    def forward(self, x):
        # Use device.xp for operations (auto CPU/GPU)
        xp = self.device.xp
        output = xp.matmul(x, self.weights)
        return output
```

**Rules:**
- Never use `numpy` directly - use `device.xp`
- Never use `cupy` directly - use `device.xp`
- Support both CPU and GPU execution
- Test on both CPU and GPU

### **Code Quality Standards**

#### **Style Guidelines**
- **PEP 8 compliance** (enforced by linters)
- **Type hints** for all function signatures
- **Docstrings** for all public methods and classes (Google/NumPy style)
- **4 spaces** for indentation (no tabs)
- **Descriptive naming** (no single-letter variables except loops)

#### **Testing Requirements**
- **Unit tests** for all new features
- **Integration tests** for complete architectures
- **CPU and GPU tests** (where applicable)
- **Benchmark tests** for performance-critical code
- **Minimum 80% code coverage** for new code

#### **Documentation Requirements**
- Docstrings explaining:
  - What the code does
  - Parameters and return values
  - Example usage
  - Mathematical formulation (for complex operations)
- README updates for major features
- CHANGELOG entries for all changes

### **Performance Standards**

For performance-critical code (attention, matmul, activations):

1. **Benchmark against reference** (if improving existing code)
2. **No performance regressions** allowed
3. **Memory profiling** for large operations
4. **Include performance tests** in PR

Example benchmark structure:
```python
import time
import numpy as np

def benchmark_attention(batch_size, seq_len, d_model):
    # Setup
    query = np.random.randn(batch_size, seq_len, d_model)
    
    # Benchmark
    start = time.time()
    output = attention_layer(query)
    elapsed = time.time() - start
    
    print(f"Time: {elapsed:.4f}s")
    return elapsed
```

## Development Setup

### **Environment Setup**

```bash
# Clone and setup
git clone https://github.com/YOUR_USERNAME/YFlow.git
cd YFlow
git checkout Contribute

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Dev dependencies

# Install YFlow in editable mode
pip install -e .

# Run tests to verify setup
pytest tests/
```

### **Running Tests**

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_yformers.py

# Run with coverage
pytest --cov=yflow tests/

# Run only GPU tests (if GPU available)
pytest -m gpu tests/

# Run only CPU tests
pytest -m cpu tests/
```

### **Code Quality Checks**

```bash
# Format code
black yflow/

# Check style
flake8 yflow/

# Type checking
mypy yflow/

# Run all pre-commit checks
pre-commit run --all-files
```

## Contribution Workflow

### **Before You Start**

1. **Check existing issues** - avoid duplicate work
2. **Open an issue** for major features (discuss approach first)
3. **Comment on issue** to claim it
4. **Get approval** for architecture changes before coding

### **During Development**

1. **Keep PRs focused** - one feature/fix per PR
2. **Write tests first** (TDD encouraged)
3. **Commit frequently** with clear messages
4. **Keep up with Contribute branch** (rebase regularly)

### **Commit Message Format**

```
<type>(<scope>): <subject>

<body>

<footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `perf`: Performance improvement
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

Example:
```
feat(yformers): add flash attention implementation

Implement flash attention for efficient memory usage in
long sequences. Reduces memory from O(n²) to O(n).

Closes #123
```

### **Pull Request Process**

1. **Update documentation** (README, docstrings, CHANGELOG)
2. **Ensure all tests pass**
3. **Write PR description**:
   - What changed
   - Why it changed
   - How to test it
   - Breaking changes (if any)
4. **Request review** from maintainers
5. **Address feedback** promptly
6. **Squash commits** if requested

### **Code Review Standards**

Your PR will be reviewed for:

- **Correctness**: Does it work as intended?
- **Quality**: Clean, readable, maintainable code?
- **Tests**: Adequate test coverage?
- **Performance**: No regressions?
- **Documentation**: Well-documented?
- **Governance**: Follows architecture rules?

## Special Contribution Types

### **Bug Fixes**

1. **Create issue first** with:
   - Bug description
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details
2. **Write failing test** that demonstrates the bug
3. **Fix the bug**
4. **Verify test now passes**
5. **Submit PR** with issue reference

### **New Architecture Proposals**

For major architectural additions:

1. **Open RFC issue** with:
   - Motivation and use case
   - Technical design
   - API proposal
   - Performance characteristics
   - Comparison to alternatives
2. **Get community feedback**
3. **Get maintainer approval**
4. **Implement with tests and docs**
5. **Submit PR**

### **Performance Improvements**

1. **Benchmark current performance**
2. **Implement optimization**
3. **Benchmark new performance**
4. **Document improvement** (% faster, memory saved)
5. **Include benchmarks in PR**

## Future Roadmap Context

### **2027: C++ Core Transition**

YFlow is planned to undergo a C++ core rewrite in 2027 for production-level performance while maintaining the current Python API. This means:

**For Contributors:**
- **Python API will remain stable** - your contributions won't be wasted
- **Focus on architecture and features** in 2026 (current year)
- **Performance optimizations** should be portable to C++ backend
- **Clean abstractions** will make backend transition smoother

**What This Means:**
- Write clean, well-abstracted code
- Don't over-optimize Python code that will be rewritten
- Focus on correctness and API design
- Document design decisions for future C++ implementers

## Getting Help

### **Questions?**

- **General questions**: Open a GitHub issue with `question` label
- **Architecture decisions**: Tag maintainers in issue
- **Urgent questions**: Use Discussions tab

### **Good First Issues**

New contributors should look for:
- `good-first-issue` label
- `help-wanted` label  
- `documentation` label
- `testing-needed` label

### **Community**

- **GitHub Issues**: Technical discussions
- **GitHub Discussions**: General questions, ideas, community
- **Pull Requests**: Code review and collaboration

## Recognition

Contributors who make significant contributions will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Given credit in relevant documentation

## License

By contributing to YFlow, you agree that your contributions will be licensed under the MIT License.

---

## Quick Reference

**Remember:**
1. ❌ No corporate ML libraries (PyTorch, TensorFlow, JAX)
2. ✅ Use device abstraction (`device.xp`)
3. ✅ Submit PRs to **Contribute branch only**
4. ✅ Write tests for everything
5. ✅ One implementation per fixed architecture
6. ✅ Freedom to create new architectures

**Questions?** Open an issue and tag maintainers.

**Ready to contribute?** Check out `good-first-issue` labels and start coding!

---

**Version**: 0.3.0  
**Last Updated**: February 2026

Thank you for helping build the future of independent, corporate-free AI infrastructure! 🚀
