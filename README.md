# YFlow: Independent Deep Learning Framework Built From Scratch

YFlow is a corporate-free deep learning framework built entirely from first principles with zero dependencies on existing ML libraries. Designed for NLP and sequence modeling tasks, it supports both CPU and GPU execution with a clean, intuitive API while maintaining complete independence and control.

## Philosophy

**Full Independence**: No PyTorch, TensorFlow, JAX, or any corporate-owned ML framework dependencies. Built from scratch to ensure complete architectural control and freedom from external constraints.

**Production-Ready NLP**: While educational in implementation transparency, YFlow is battle-tested and production-capable for natural language processing and transformer-based applications.

**Community-Governed**: Open development with strict architectural governance to prevent fragmentation while encouraging innovation.

## Features

### Core Capabilities
- **CPU and GPU Support**: Hardware acceleration with automatic fallback (GPU tested and validated)
- **Automatic Differentiation**: Built-in gradient computation for all operations
- **Device Abstraction**: Seamless CPU/GPU switching with unified API
- **Modular Architecture**: Clean separation of concerns across layers, optimizers, and losses

### Transformer Architecture (YFormers)
- **Three Model Types**: Encoder-Decoder (T5-style), Encoder-Only (BERT-style), Decoder-Only (GPT-style)
- **Production-Tested**: Full transformer implementation validated on real tasks
- **Advanced Generation**: Temperature, top-k, top-p sampling for text generation
- **Flexible Attention**: Multi-head self-attention, cross-attention, causal masking

### Training Infrastructure
- **Optimizers**: Adam, AdamW, SGD, RMSProp with learning rate scheduling
- **Loss Functions**: Cross-entropy (with ignore_index), MSE, and custom losses
- **Data Loading**: FlowDL DataLoader with batching, shuffling, and efficient iteration
- **Checkpointing**: Full training state save/restore with best model tracking
- **Mixed Precision**: Optional fp16/bf16 training support

### Sequence Modeling
- **LSTM (YSTM)**: Production-ready LSTM with proper gradient flow
- **RNN (YQuence)**: Standard recurrent architecture
- **BiRNN (BiYQuence)**: Bidirectional recurrent networks

## Installation

```bash
# Clone the repository
git clone https://github.com/krauscode920/YFlow.git
cd YFlow

# Install dependencies (numpy, tqdm, matplotlib only)
pip install -r requirements.txt

# Install YFlow
pip install -e .
```

## Quick Start

### Simple Classification Network

```python
from yflow.core.model import Model
from yflow.layers.dense import Dense
from yflow.layers.activations import ReLU, Sigmoid
from yflow.losses.mse import MSELoss
from yflow.optimizers.adam import Adam
import numpy as np

# Define network
class SimpleNN(Model):
    def __init__(self):
        super().__init__()
        self.fc1 = Dense(input_dim=10, output_dim=64)
        self.relu1 = ReLU()
        self.fc2 = Dense(input_dim=64, output_dim=32)
        self.relu2 = ReLU()
        self.fc3 = Dense(input_dim=32, output_dim=1)
        self.sigmoid = Sigmoid()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return self.sigmoid(x)

# Initialize
model = SimpleNN()
optimizer = Adam(learning_rate=0.001)

# Training loop
for epoch in range(100):
    y_pred = model(x_train)
    loss = MSELoss()(y_pred, y_train)
    
    loss.backward()
    optimizer.step(model.parameters())
    optimizer.zero_grad()
```

### Language Modeling with Transformers

```python
from yflow.yformers.model import DecoderOnlyModel
from yflow.losses.cross_entropy import CrossEntropyLoss
from yflow.optimizers.adam import Adam
from yflow.data import DataLoader, TensorDataset
from yflow.checkpoint import CheckpointManager

# Initialize GPT-style model
model = DecoderOnlyModel(
    vocab_size=50000,
    d_model=768,
    num_heads=12,
    d_ff=3072,
    num_layers=12,
    max_seq_len=1024,
    dropout=0.1
)

# Setup training
optimizer = Adam(learning_rate=3e-4, weight_decay=0.01)
loss_fn = CrossEntropyLoss(ignore_index=pad_token_id)
checkpoint_mgr = CheckpointManager(checkpoint_dir='./checkpoints')

# Create data loader
dataset = TensorDataset(input_ids, labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training loop with checkpointing
for epoch in range(num_epochs):
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        # Forward pass
        logits = model(inputs)
        loss = loss_fn(logits, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step(model.parameters())
        optimizer.zero_grad()
        
        # Checkpoint every 1000 steps
        if batch_idx % 1000 == 0:
            checkpoint_mgr.save(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                step=batch_idx,
                metrics={'loss': loss.item()}
            )

# Text generation
prompt_tokens = np.array([[1, 234, 5678, 91011]])  # Your tokenized prompt
generated = model.generate(
    prompt_tokens,
    max_len=200,
    temperature=0.8,
    top_k=50,
    top_p=0.9
)
```

### Using DataLoader

```python
from yflow.data import DataLoader, TensorDataset
import numpy as np

# Prepare data
X = np.random.randn(1000, 128)  # 1000 samples, 128 features
y = np.random.randint(0, 10, (1000,))  # 10 classes

# Create dataset and loader
dataset = TensorDataset(X, y)
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    drop_last=False
)

# Iterate
for batch_x, batch_y in loader:
    # batch_x: (32, 128), batch_y: (32,)
    predictions = model(batch_x)
    loss = loss_fn(predictions, batch_y)
```

## Project Structure

```
yflow/
├── core/                   # Core framework components
│   ├── device.py          # CPU/GPU abstraction layer
│   ├── layer.py           # Base layer class with forward/backward
│   ├── model.py           # Model container and training loop
│   └── context.py         # Computation graph context
│
├── layers/                 # Neural network layers
│   ├── dense.py           # Fully connected layer
│   ├── lstm.py            # LSTM implementation (YSTM)
│   ├── activations.py     # ReLU, GELU, Sigmoid, Tanh
│   ├── dropout.py         # Dropout regularization
│   └── normalization.py   # LayerNorm, BatchNorm
│
├── yformers/              # Transformer architecture
│   ├── attention.py       # Multi-head self-attention
│   ├── embeddings.py      # Token and positional embeddings
│   ├── encoder.py         # Encoder blocks and stack
│   ├── decoder.py         # Decoder blocks and stack
│   ├── model.py           # Complete transformer models
│   └── utils.py           # Masking and utilities
│
├── optimizers/            # Optimization algorithms
│   ├── adam.py            # Adam optimizer
│   ├── sgd.py             # SGD with momentum
│   └── rmsprop.py         # RMSProp optimizer
│
├── losses/                # Loss functions
│   ├── cross_entropy.py   # Cross-entropy with label smoothing
│   └── mse.py             # Mean squared error
│
├── data.py                # DataLoader and Dataset utilities
├── checkpoint.py          # Training state management
└── utils/                 # Utility functions
    ├── lr_scheduler.py    # Learning rate schedules
    └── metrics.py         # Evaluation metrics
```

## YFormers: Transformer Architecture

YFormers provides production-ready transformer implementations for all major architecture patterns.

### Architecture Variants

#### 1. Full Transformer (Encoder-Decoder)
For sequence-to-sequence tasks like translation and summarization.

```python
from yflow.yformers.model import TransformerModel

model = TransformerModel(
    src_vocab_size=30000,
    tgt_vocab_size=30000,
    d_model=512,
    num_heads=8,
    d_ff=2048,
    num_encoder_layers=6,
    num_decoder_layers=6,
    dropout=0.1,
    max_src_len=512,
    max_tgt_len=512
)

# Training
logits = model.forward(src_tokens, tgt_tokens)
loss = loss_fn(logits, target_tokens)

# Generation
output = model.generate(src_tokens, max_len=100)
```

#### 2. Encoder-Only (BERT-style)
For classification, feature extraction, and discriminative tasks.

```python
from yflow.yformers.model import EncoderOnlyModel

model = EncoderOnlyModel(
    vocab_size=30000,
    d_model=768,
    num_heads=12,
    d_ff=3072,
    num_layers=12,
    num_classes=2,  # For binary classification
    dropout=0.1
)

# Classification
logits = model.forward(tokens)  # (batch, num_classes)
```

#### 3. Decoder-Only (GPT-style)
For autoregressive language modeling and text generation.

```python
from yflow.yformers.model import DecoderOnlyModel

model = DecoderOnlyModel(
    vocab_size=50000,
    d_model=768,
    num_heads=12,
    d_ff=3072,
    num_layers=12,
    max_seq_len=2048,
    dropout=0.1
)

# Training
logits = model.forward(tokens)  # (batch, seq_len, vocab_size)

# Generation with sampling strategies
generated = model.generate(
    prompt_tokens,
    max_len=500,
    temperature=0.7,    # Controls randomness
    top_k=40,          # Top-k sampling
    top_p=0.95         # Nucleus sampling
)
```

### Key Features

- **Tested in Production**: YFormers has been validated on real language modeling tasks
- **Flexible Masking**: Padding masks, causal masks, cross-attention masks
- **Modern Implementations**: Pre-norm architecture, GELU activations
- **Efficient Generation**: Optimized text generation with multiple sampling strategies
- **Device Agnostic**: Seamless CPU/GPU execution

## Checkpoint Management

```python
from yflow.checkpoint import CheckpointManager

# Initialize checkpoint manager
ckpt_mgr = CheckpointManager(
    checkpoint_dir='./checkpoints',
    max_to_keep=3  # Keep only 3 most recent checkpoints
)

# Save checkpoint
ckpt_mgr.save(
    model=model,
    optimizer=optimizer,
    epoch=epoch,
    step=global_step,
    metrics={'loss': train_loss, 'accuracy': val_acc}
)

# Load latest checkpoint
checkpoint = ckpt_mgr.restore_latest()
if checkpoint:
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    start_epoch = checkpoint['epoch'] + 1
    
# Load best checkpoint (by metric)
best_checkpoint = ckpt_mgr.restore_best(metric='accuracy')
```

## GPU Support

YFlow has been tested and validated on GPU hardware. The device abstraction automatically handles CPU/GPU execution.

```python
from yflow.core.device import is_gpu_available, get_device

# Check GPU availability
if is_gpu_available():
    device = get_device('gpu')
    print("Training on GPU")
else:
    device = get_device('cpu')
    print("Training on CPU")

# Move model to device
model.to(device)

# All operations automatically use the correct device
output = model(input_data)  # Uses GPU if available
```

## Architecture Governance

YFlow maintains strict architectural governance to prevent fragmentation:

### Fixed Architecture Names

These architectures are permanent and cannot be duplicated:

- **YFormers**: All transformer variants (GPT, BERT, T5, etc.)
- **YSTM**: LSTM networks
- **YQuence**: Standard RNN
- **BiYQuence**: Bidirectional RNN

### Contribution Rules

**For Existing Architectures:**
- Work within the existing codebase (modify, don't duplicate)
- No parallel implementations (no "YFormers2" or "FastYFormers")
- Maintain backward compatibility
- Submit PRs to Contribute branch only

**For New Architectures:**
- Complete freedom to create novel architectures
- No naming restrictions (Y-prefix not required)
- Must demonstrate clear innovation
- Subject to review and approval

This ensures a unified, coherent framework while encouraging innovation.

## Contributing

We welcome contributions! All submissions must go to the **Contribute branch**.

### Priority Areas

**Immediate Needs:**
1. **Performance Optimization**: CUDA kernel optimization, memory efficiency
2. **Advanced Architectures**: Vision transformers, sparse attention within YFormers
3. **Training Utilities**: Advanced data augmentation, distributed training
4. **Documentation**: Tutorials, example notebooks, API documentation

**Core Requirements:**
- **NO external ML libraries** (PyTorch, TensorFlow, JAX, etc.) - automatic rejection
- Follow device abstraction patterns
- Include comprehensive tests
- Maintain code quality and documentation

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

### Contribution Process

1. Fork the repository
2. Create branch from **Contribute branch**
3. Make changes following guidelines
4. Submit PR to **Contribute branch ONLY**
5. Code review and approval
6. Merge to main upon approval

## Roadmap

### 2026 (Current Year)
- ✅ YFormers production-ready
- ✅ Checkpoint management
- ✅ DataLoader implementation
- ✅ GPU testing and validation
- 🔄 Advanced optimizers (Lion, Sophia)
- 🔄 Learning rate scheduling improvements
- 🔄 Model quantization for inference

### 2027
- C++ core implementation for performance
- CUDA kernel optimization
- Production deployment utilities
- Distributed training support
- Model serving infrastructure

### 2028
- Full production ecosystem
- Pre-trained model zoo
- Enterprise deployment tools

## Why YFlow?

**Independence**: No reliance on corporate-owned frameworks means no vendor lock-in, no sudden API changes, and complete control over your ML infrastructure.

**Transparency**: Every component is implemented from scratch, making the entire framework understandable and modifiable.

**NLP-First**: Specialized for natural language processing with battle-tested transformer implementations.

**Production-Ready**: Not just educational - YFormers and core components are validated on real workloads.

**Community-Governed**: Open development with clear architectural principles to prevent fragmentation.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

YFlow was created to demonstrate that high-quality, production-ready deep learning frameworks can exist outside corporate control. The project prioritizes independence, transparency, and community governance.

Special thanks to all contributors who believe in corporate-free, open-source AI infrastructure.

## Version

**Current Version**: 0.3.0

**What's New in 0.3.0:**
- ✅ Production-tested YFormers (all three architectures)
- ✅ Full checkpoint management system
- ✅ DataLoader with efficient batching
- ✅ GPU validation and testing complete
- ✅ Improved documentation and examples

---

**Questions?** Open an issue on GitHub.

**Want to contribute?** Read [CONTRIBUTING.md](CONTRIBUTING.md) and start with issues tagged `good-first-issue`.
