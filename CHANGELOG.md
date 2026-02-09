# Changelog

All notable changes to YFlow will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- C++ core implementation (2027)
- CUDA kernel optimizations
- Model quantization (int8, int4)
- Distributed training support
- ONNX export functionality

## [0.3.0] - 2026-02-09

### Added
- **Production-validated YFormers**: All three transformer architectures (Encoder-Decoder, Encoder-Only, Decoder-Only) fully tested and validated on real workloads
- **Checkpoint Management System**: Complete training state save/restore with `CheckpointManager`
  - Save/restore model and optimizer state
  - Track best checkpoints by metric
  - Automatic checkpoint rotation (max_to_keep)
- **DataLoader (FlowDL)**: Efficient data loading with batching and shuffling
  - `TensorDataset` for wrapping numpy arrays
  - Configurable batch size and shuffling
  - Drop last batch option
  - Memory-efficient iteration
- **GPU Validation**: Complete testing and validation on GPU hardware
  - Confirmed device abstraction works on real GPUs
  - Performance benchmarks on GPU vs CPU
- **Advanced Text Generation**: Multiple sampling strategies for DecoderOnlyModel
  - Temperature-based sampling
  - Top-k sampling
  - Top-p (nucleus) sampling
  - Configurable max generation length

### Improved
- **Documentation**: Complete rewrite of README and CONTRIBUTING with current state
- **Architecture Governance**: Clarified rules and examples for contributions
- **Device Abstraction**: Validated and tested CPU/GPU switching
- **Error Messages**: More informative error messages across the framework

### Fixed
- GPU memory management in attention mechanisms
- Gradient flow in multi-layer transformers
- DataLoader edge cases with small datasets

## [0.2.0] - 2025-12-15

### Added
- **YFormers Module**: Complete transformer architecture implementation
  - `TransformerModel`: Full encoder-decoder transformer
  - `EncoderOnlyModel`: BERT-style encoder for classification
  - `DecoderOnlyModel`: GPT-style decoder for language modeling
  - Multi-head self-attention with proper masking
  - Positional encoding (fixed and learnable)
  - Layer normalization and feed-forward networks
- **Architecture Governance Model**: Strict rules to prevent fragmentation
  - Fixed architecture names (YFormers, YSTM, YQuence, BiYQuence)
  - No parallel implementations allowed
  - Clear contribution guidelines
- **LSTM Layer (YSTM)**: Long Short-Term Memory implementation
  - Proper gate mechanisms (input, forget, cell, output)
  - Gradient flow optimization
  - Sequence-to-sequence support
- **Advanced Optimizers**:
  - Adam optimizer with weight decay (AdamW)
  - Learning rate scheduling support
- **Loss Functions**:
  - Cross-entropy with ignore_index for padding
  - Label smoothing support

### Improved
- **Device Abstraction**: More robust CPU/GPU handling
- **Documentation**: Added YFormers documentation and examples
- **Testing**: Comprehensive test suite for transformers

## [0.1.0] - 2025-10-01

### Added
- **Core Framework**:
  - Device abstraction (CPU/GPU support with numpy/cupy)
  - Base Layer class with forward/backward methods
  - Model container class
  - Automatic differentiation system
- **Basic Layers**:
  - Dense (fully connected) layer
  - Activation functions (ReLU, Sigmoid, Tanh, GELU)
  - Dropout layer
  - Batch normalization
  - Layer normalization
- **Optimizers**:
  - Stochastic Gradient Descent (SGD) with momentum
  - Adam optimizer
  - RMSProp optimizer
- **Loss Functions**:
  - Mean Squared Error (MSE)
  - Cross-Entropy Loss
- **Initial Documentation**:
  - README with quick start examples
  - CONTRIBUTING guidelines
  - MIT License
  - Bug report and feature request templates

### Architecture Decisions
- Chose MIT License for maximum freedom
- Established zero-dependency policy (no corporate ML frameworks)
- Designed device abstraction for future GPU support
- Implemented clean separation between core, layers, optimizers, and losses

## Version History Summary

- **0.3.0** (Current): Production-ready with checkpointing, data loading, GPU validation
- **0.2.0**: YFormers transformers, LSTM, architecture governance
- **0.1.0**: Initial release with core framework and basic layers

## Migration Guides

### Migrating from 0.2.0 to 0.3.0

No breaking changes. New features are additive:

```python
# New checkpoint management
from yflow.checkpoint import CheckpointManager

ckpt_mgr = CheckpointManager('./checkpoints')
ckpt_mgr.save(model=model, optimizer=optimizer, epoch=epoch)

# New data loading
from yflow.data import DataLoader, TensorDataset

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Enhanced generation (existing code still works)
generated = model.generate(
    prompt,
    max_len=100,
    temperature=0.8,  # New parameter
    top_k=50,         # New parameter
    top_p=0.9         # New parameter
)
```

### Migrating from 0.1.0 to 0.2.0

YFormers addition is non-breaking. If you were using basic layers, all existing code continues to work:

```python
# Your 0.1.0 code still works
from yflow.layers.dense import Dense
from yflow.optimizers.adam import Adam

# New transformer support
from yflow.yformers.model import DecoderOnlyModel

model = DecoderOnlyModel(vocab_size=10000, d_model=512)
```

## Acknowledgments

### Contributors
Thank you to all contributors who have helped build YFlow:
- Core team and maintainers
- Community contributors
- Bug reporters and testers

### Special Thanks
- Everyone who believed in corporate-free AI infrastructure
- Contributors who helped validate GPU support
- Community members who provided feedback on architecture decisions

---

**Questions about a specific version?** Check the git tags or open an issue.

**Want to contribute?** See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
