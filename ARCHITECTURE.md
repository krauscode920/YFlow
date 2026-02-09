# YFlow Architecture Documentation

This document explains the architectural decisions, design patterns, and implementation philosophy behind YFlow. Understanding this will help you contribute effectively and maintain consistency with the framework's vision.

## Core Philosophy

### Independence from Corporate Frameworks

**Why we don't use PyTorch/TensorFlow/JAX:**

YFlow exists because we believe in building AI infrastructure that is:
1. **Truly open**: Not controlled by corporate interests
2. **Transparent**: Every line of code is understandable and modifiable
3. **Independent**: No vendor lock-in or dependency on external ML frameworks
4. **Educational**: Implementation details are visible and documented

This isn't just ideological—it's practical. Corporate frameworks can change APIs, deprecate features, or shift priorities based on business needs. YFlow guarantees stability and control.

### Production-Ready, Not Just Educational

While YFlow's transparent implementation makes it excellent for learning, it's designed for real production workloads:

- **Performance**: Optimized for actual use, not just demos
- **Reliability**: Comprehensive testing and validation
- **API Stability**: Semantic versioning and migration guides
- **Scalability**: GPU support and future C++ core

## Architectural Layers

### Layer 1: Device Abstraction

**Location**: `yflow/core/device.py`

The device abstraction is the foundation of YFlow's portability.

**Design Decision**: Use property-based abstraction (`device.xp`)

```python
class Device:
    @property
    def xp(self):
        """Returns numpy for CPU, cupy for GPU"""
        return cupy if self.is_gpu else numpy
```

**Why this approach:**
- Single abstraction point for all numerical operations
- Automatic fallback to CPU when GPU unavailable
- No code changes needed to switch devices
- Future-proof for other backends (TPU, custom accelerators)

**Alternative considered**: Conditional imports throughout codebase
- **Rejected**: Would require changes in hundreds of places
- **Rejected**: Harder to maintain and test

**Usage pattern:**
```python
class MyLayer(Layer):
    def forward(self, x):
        xp = self.device.xp  # Auto-selects numpy or cupy
        return xp.matmul(x, self.weights)
```

### Layer 2: Automatic Differentiation

**Current Implementation**: Manual backward methods in each layer

**Design Decision**: Explicit backward pass

```python
class Dense(Layer):
    def forward(self, x):
        self.input = x  # Cache for backward
        return x @ self.weights + self.bias
    
    def backward(self, grad_output):
        # Manually compute gradients
        self.grad_weights = self.input.T @ grad_output
        self.grad_bias = np.sum(grad_output, axis=0)
        return grad_output @ self.weights.T
```

**Why manual backward:**
- **Control**: Explicit gradient computation is easier to debug
- **Performance**: Can optimize specific layer backward passes
- **Educational**: Makes gradient flow transparent
- **Simplicity**: No computational graph overhead (yet)

**Future Plan (2027)**: Automatic differentiation engine
- Build computational graph during forward pass
- Automatic gradient computation via chain rule
- PyTorch-like autograd experience
- Maintains backward compatibility with current API

**Why deferred:**
- Current approach works well for NLP architectures
- Can focus on architecture quality now
- C++ core transition is better time to add autograd

### Layer 3: Base Abstractions

**Location**: `yflow/core/layer.py`, `yflow/core/model.py`

**Layer Contract**:
```python
class Layer:
    def forward(self, x):
        """Transform input to output"""
        raise NotImplementedError
    
    def backward(self, grad_output):
        """Compute gradients and return grad_input"""
        raise NotImplementedError
    
    def parameters(self):
        """Return trainable parameters"""
        return []
```

**Design Decision**: Simple, explicit contract

**Why:**
- Easy to understand and implement
- Clear responsibilities
- Minimal magic or hidden behavior
- Composable by design

**Model as Layer Container**:
```python
class Model:
    def __init__(self):
        self.layers = []
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
```

**Why simple container:**
- Explicit layer ordering
- Easy to inspect and debug
- Clear data flow
- Can be extended with training loop utilities

### Layer 4: Specialized Architectures

**YFormers Design Pattern**:

**Governance Decision**: One implementation per architecture type

```
yformers/
├── attention.py       # Attention mechanisms (shared)
├── embeddings.py      # Token and positional embeddings (shared)
├── encoder.py         # Encoder blocks and stack
├── decoder.py         # Decoder blocks and stack
├── model.py           # Complete models (3 variants)
└── utils.py           # Utilities and masking
```

**Why unified YFormers:**
- **Prevents fragmentation**: No "YFormers2" or competing implementations
- **Code reuse**: Shared attention, embeddings, utilities
- **Consistency**: Same API across all transformer variants
- **Maintainability**: One place to fix bugs and add features

**Three model variants in one architecture:**

1. **TransformerModel** (Encoder-Decoder): Translation, summarization
2. **EncoderOnlyModel** (BERT-style): Classification, feature extraction
3. **DecoderOnlyModel** (GPT-style): Language modeling, generation

**Why three variants:**
- Different use cases require different architectures
- Shared components reduce duplication
- Users can choose the right tool for their task
- Still one cohesive architecture (not three competing ones)

## Design Patterns

### Pattern 1: Cache for Backward Pass

**Problem**: Backward pass needs values from forward pass

**Solution**: Cache in instance variables

```python
class Attention(Layer):
    def forward(self, query, key, value):
        # Compute attention
        scores = query @ key.T / sqrt(d_k)
        weights = softmax(scores)
        output = weights @ value
        
        # Cache for backward
        self.cache = {
            'query': query,
            'key': key,
            'value': value,
            'weights': weights,
            'scores': scores
        }
        return output
    
    def backward(self, grad_output):
        # Use cached values to compute gradients
        weights = self.cache['weights']
        # ... gradient computation
```

**Why caching:**
- Avoids recomputation in backward pass
- Makes gradients easier to compute correctly
- Clear separation between forward and backward

**Trade-off**: Memory usage
- For large sequences, cache can be significant
- Future optimization: gradient checkpointing

### Pattern 2: Device-Aware Initialization

**Problem**: Parameters must be on correct device

**Solution**: Use device abstraction in initialization

```python
class Dense(Layer):
    def __init__(self, input_dim, output_dim, device=None):
        self.device = device or get_default_device()
        xp = self.device.xp
        
        # Initialize on correct device
        self.weights = xp.random.randn(input_dim, output_dim) * 0.01
        self.bias = xp.zeros(output_dim)
```

**Why device-aware init:**
- Parameters created on correct device from start
- Avoids expensive device transfers later
- Consistent with device abstraction philosophy

### Pattern 3: Residual Connections

**Implementation**:
```python
class EncoderBlock(Layer):
    def forward(self, x):
        # Multi-head attention with residual
        attn_output = self.attention(x)
        x = self.norm1(x + attn_output)  # Residual connection
        
        # Feed-forward with residual
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)  # Residual connection
        
        return x
```

**Why residual connections:**
- Enable training very deep networks (12+ layers)
- Gradient highway prevents vanishing gradients
- Allows network to learn identity function if needed
- Essential for transformer architectures

### Pattern 4: Masking for Sequences

**Problem**: Variable-length sequences need masking

**Solution**: Explicit mask tensors

```python
def create_causal_mask(seq_len, device):
    """Lower triangular matrix for autoregressive attention"""
    xp = device.xp
    mask = xp.tril(xp.ones((seq_len, seq_len)))
    return mask

class DecoderBlock(Layer):
    def forward(self, x, mask=None):
        # Apply mask to attention scores
        scores = query @ key.T
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        weights = softmax(scores)
```

**Why explicit masking:**
- Clear semantics (what is masked and why)
- Flexible (different mask types for different tasks)
- Debuggable (can inspect masks)

## Optimization Strategies

### Memory Efficiency

**Current Optimizations**:

1. **In-place operations where safe**:
```python
# Safe: doesn't affect gradients
x += bias  # In-place addition

# Unsafe: would break gradient computation
x *= weights  # Creates new array
```

2. **Gradient accumulation support**:
```python
# Accumulate gradients over multiple batches
for mini_batch in accumulation_steps:
    loss = compute_loss(mini_batch)
    loss.backward()
    # Don't step optimizer yet

optimizer.step()  # Step after N accumulations
optimizer.zero_grad()
```

3. **Selective caching**:
```python
# Only cache what's needed for backward
if self.training:
    self.cache = {'values': needed_for_backward}
else:
    # Inference: no caching needed
    pass
```

**Future Optimizations** (2027+ with C++ core):
- Gradient checkpointing (recompute instead of cache)
- Memory pooling
- Kernel fusion
- Mixed precision (fp16/bf16)

### Computational Efficiency

**Current Strategies**:

1. **Batched operations**:
```python
# Batch matrix multiply instead of loop
output = x @ weights  # (batch, seq, hidden) @ (hidden, output)
# Instead of: [x[i] @ weights for i in range(batch)]
```

2. **Vectorization**:
```python
# Vectorized softmax
exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
softmax = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
```

**Future with C++**:
- Custom CUDA kernels for attention
- Fused operations (LayerNorm + residual)
- Flash Attention implementation
- Optimized matmul kernels

## Testing Strategy

### Test Levels

**1. Unit Tests**: Individual layer correctness
```python
def test_dense_forward():
    layer = Dense(10, 5)
    x = np.random.randn(2, 10)
    output = layer(x)
    assert output.shape == (2, 5)

def test_dense_backward():
    layer = Dense(10, 5)
    x = np.random.randn(2, 10)
    output = layer(x)
    grad = layer.backward(np.ones_like(output))
    assert grad.shape == x.shape
```

**2. Integration Tests**: Complete models
```python
def test_decoder_only_model():
    model = DecoderOnlyModel(vocab_size=1000, d_model=128)
    tokens = np.random.randint(0, 1000, (2, 10))
    logits = model(tokens)
    assert logits.shape == (2, 10, 1000)
```

**3. Gradient Tests**: Numerical gradient checking
```python
def test_layer_gradients():
    """Verify analytical gradients match numerical gradients"""
    layer = Dense(10, 5)
    x = np.random.randn(1, 10)
    
    # Analytical gradient
    output = layer(x)
    analytical_grad = layer.backward(np.ones_like(output))
    
    # Numerical gradient (finite differences)
    numerical_grad = compute_numerical_gradient(layer, x)
    
    np.testing.assert_allclose(analytical_grad, numerical_grad, rtol=1e-5)
```

**4. Device Tests**: CPU and GPU equivalence
```python
@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_attention_device(device):
    if device == "gpu" and not is_gpu_available():
        pytest.skip("GPU not available")
    
    layer = MultiHeadAttention(d_model=128, num_heads=8, device=device)
    # ... test logic
```

**5. Performance Tests**: Regression prevention
```python
def test_attention_performance():
    """Ensure attention doesn't regress in speed"""
    layer = MultiHeadAttention(d_model=512, num_heads=8)
    x = np.random.randn(32, 100, 512)  # Large batch
    
    start = time.time()
    output = layer(x)
    elapsed = time.time() - start
    
    assert elapsed < PERFORMANCE_THRESHOLD
```

## Future Architecture Evolution

### 2027: C++ Core Transition

**Plan**: Rewrite compute-intensive parts in C++ while maintaining Python API

**Architecture**:
```
yflow/
├── python/              # Python API layer (current)
│   ├── core/
│   ├── layers/
│   └── yformers/
│
├── cpp/                 # C++ core (new)
│   ├── tensor/          # Tensor operations
│   ├── autograd/        # Automatic differentiation
│   ├── kernels/         # CUDA kernels
│   └── bindings/        # Python bindings (pybind11)
│
└── tests/               # Tests for both
```

**Migration Strategy**:
1. **Phase 1**: Implement C++ tensor operations
2. **Phase 2**: Build autograd engine in C++
3. **Phase 3**: Port critical kernels (matmul, attention)
4. **Phase 4**: Create Python bindings
5. **Phase 5**: Migrate gradually, maintain API compatibility

**User Impact**: **None**
- Same Python API
- Automatically faster
- Can still modify Python code
- Gradual transition per module

### Automatic Differentiation Design

**Planned computational graph approach**:

```python
# Future API (same as current, but no manual backward)
class Layer:
    def forward(self, x):
        # Just define forward pass
        return some_operation(x)
    
    # No backward needed - computed automatically

# Engine builds graph during forward
x = Variable(data)
y = model(x)  # Graph built automatically
loss = criterion(y, target)
loss.backward()  # Automatic gradient computation
```

**Implementation concept**:
```cpp
// C++ computational graph node
struct Node {
    Tensor value;
    std::vector<Node*> inputs;
    std::function<Tensor(Tensor)> backward_fn;
};

// Automatic backward through graph
void backward(Node* node, Tensor grad) {
    for (auto input : node->inputs) {
        Tensor input_grad = node->backward_fn(grad);
        backward(input, input_grad);
    }
}
```

**Why deferred until 2027:**
- Current manual approach works well
- Can focus on architecture quality now
- C++ makes autograd more efficient
- Easier to implement correctly with C++ performance

## Governance Rationale

### Why Fixed Architectures?

**Problem**: Multiple implementations cause:
- User confusion ("which YFormers should I use?")
- Fragmented ecosystem
- Duplicated effort
- Incompatible models
- Maintenance burden

**Solution**: One canonical implementation per architecture

**Benefits**:
- Clear choice for users
- Concentrated development effort
- Better quality through focused improvement
- Ecosystem compatibility
- Easier maintenance

**Allowance**: New architectures for genuinely different use cases

**Example**:
- ✅ YFormers (transformers) + ConvNet (CNNs) = Good diversity
- ❌ YFormers + YFormers2 + FastYFormers = Bad fragmentation

### Why Y-Prefix Not Required for New Architectures?

**Decision**: Only fixed architectures have Y-prefix requirement

**Rationale**:
- "Y" prefix was historical (YFlow, YFormers, YSTM)
- New architectures should have descriptive names
- Forcing "Y" prefix limits creativity
- What matters: quality and innovation, not naming

**Examples**:
- `GraphNeuralNetwork` - Clear and descriptive
- `StateSpaceModel` - Industry-standard terminology
- `YGraphNet` - Unnecessarily branded

## Contributing to Architecture

### Proposing Architecture Changes

**For major changes, create RFC (Request for Comments)**:

1. **Open issue** with RFC template
2. **Describe problem** being solved
3. **Propose solution** with design doc
4. **Show alternatives** considered
5. **Get feedback** from community
6. **Iterate** based on discussion
7. **Implement** after approval

**Example RFC sections**:
```markdown
## Problem
Current attention mechanism doesn't support sparse patterns.

## Proposal
Add SparseAttention class within YFormers module.

## Design
- Inherit from base Attention
- Support configurable sparsity patterns
- Maintain same API as dense attention

## Alternatives Considered
1. Separate architecture: Rejected (belongs in YFormers)
2. Replace existing attention: Rejected (breaking change)
3. Optional parameter in existing: Rejected (too complex)

## Implementation Plan
1. Add SparseAttention class
2. Add sparsity pattern generators
3. Test against dense attention
4. Benchmark performance
5. Document usage
```

### Design Review Criteria

**Architecture proposals evaluated on**:

1. **Necessity**: Is this genuinely needed?
2. **Scope**: Does it fit in existing architecture or need new one?
3. **Quality**: Is the design sound?
4. **Performance**: Does it meet performance requirements?
5. **Maintainability**: Can we maintain this long-term?
6. **Documentation**: Is it well-documented?
7. **Tests**: Adequate test coverage?
8. **Independence**: No corporate ML library dependencies?

## FAQ for Contributors

**Q: Why no PyTorch for just one small function?**  
A: Slippery slope. Once we allow one dependency, where do we draw the line? Complete independence is a core principle.

**Q: The manual backward pass is tedious. Can I use autograd?**  
A: Not from PyTorch/JAX. You can help build YFlow's autograd engine (planned for 2027).

**Q: Why not make YFlow compatible with PyTorch?**  
A: That would defeat the purpose. We want independence, not interoperability with corporate frameworks.

**Q: Can I use cupy?**  
A: Yes! Through `device.xp`. That's part of our device abstraction.

**Q: Why version 0.3.0 when it's production-ready?**  
A: Semantic versioning. 1.0.0 will mean API stability guarantee. We're not there yet (C++ core transition ahead).

**Q: Will my Python code be wasted when C++ comes?**  
A: No. Python API stays the same. C++ is backend optimization. Your architecture contributions remain valuable.

**Q: Can I add a CNN if you're NLP-focused?**  
A: Yes! NLP-focus doesn't mean NLP-only. Quality CNN implementation would be welcomed.

---

**Questions?** Open an issue or ask in GitHub Discussions.

**Want to contribute?** This architecture doc should help you understand how to design features that fit YFlow's philosophy.
