# YFlow: GPU-Compatible Deep Learning Library Built From Scratch

YFlow is a custom deep learning framework built entirely from scratch with no dependencies on existing ML libraries. It supports both CPU and GPU execution and provides a clean, intuitive API while maintaining flexibility for advanced deep learning research and applications.

## Core Principles

- **Zero External ML Dependencies**: Built completely from first principles without using TensorFlow, PyTorch, or other ML libraries
- **Unified Architecture**: Clean, consistent implementation with strict governance to prevent fragmentation
- **Educational Purpose**: Designed to understand deep learning fundamentals by implementing everything from scratch
- **Community-Driven**: Open development with structured contribution process

## Features

- **CPU and GPU Support**: Designed with hardware acceleration in mind (GPU support implementation included but currently untested)
- **Modular Architecture**: Well-organized structure separated into core functionality, layers, losses, optimizers, and utilities
- **Automatic Differentiation**: Built-in gradient computation
- **Customizable Layers**: Implement your own or use provided implementations
- **Optimizers**: Standard optimization algorithms including SGD, Adam, and RMSProp
- **Device Abstraction**: Clean separation between compute logic and hardware acceleration
- **Transformer Architecture**: Complete transformer implementation with YFormers module

## Installation

```bash
# Clone the repository
git clone https://github.com/krauscode920/YFlow.git

# Install dependencies
cd YFlow
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Quick Start Example

```python
from yflow.core.model import Model
from yflow.layers.dense import Dense
from yflow.layers.activations import ReLU, Sigmoid
from yflow.losses.mse import MSELoss
from yflow.optimizers.adam import Adam

# Define a simple neural network
class SimpleNN(Model):
    def __init__(self):
        super().__init__()
        self.fc1 = Dense(input_dim=10, output_dim=64)
        self.relu = ReLU()
        self.fc2 = Dense(input_dim=64, output_dim=32)
        self.fc3 = Dense(input_dim=32, output_dim=1)
        self.sigmoid = Sigmoid()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

# Create model, loss, and optimizer
model = SimpleNN()
loss_fn = MSELoss()
optimizer = Adam(learning_rate=0.001)

# Training loop example
def train(model, x_data, y_data, epochs=100):
    for epoch in range(epochs):
        # Forward pass
        y_pred = model(x_data)
        
        # Compute loss
        loss = loss_fn(y_pred, y_data)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step(model.parameters())
        
        # Reset gradients
        optimizer.zero_grad()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.value}")
```

## Architecture Naming Convention & Governance

YFlow follows a strict naming convention to maintain consistency and prevent architectural fragmentation:

### **Fixed Architecture Names (CANNOT BE CHANGED)**

These architectures have established names that are permanent and cannot be modified:

- **YFormers** - All Transformer architectures (GPT, BERT, T5, etc.)
- **YLSTM** - Long Short-Term Memory networks
- **YQuence** - Standard Recurrent Neural Networks (RNN)
- **BiYQuence** - Bidirectional Recurrent Neural Networks

### **Governance Principles**

#### **No Parallel Architectures**
- There can only be ONE implementation per architecture type
- Contributors cannot create alternative versions (e.g., "YFormers2" or "FastYFormers")
- All improvements must be made to the existing architecture

#### **Modification Process**
- To improve an existing architecture, contributors must:
  1. **Work within the existing codebase** (modify YFormers, not create alternatives)
  2. **Submit pull requests** to the **Contribute branch only**
  3. **Pass code review** and maintainer approval
  4. **Maintain backward compatibility** where possible

#### **New Architecture Freedom**
- Contributors are welcome to create entirely **new architectures**
- New architectures can be named freely (following general YFlow conventions)
- Novel architectures are subject to evaluation and approval
- Must demonstrate clear innovation over existing architectures

### **Examples**

#### ✅ **Allowed Contributions:**
```python
# Improving existing YFormers
class YFormers(TransformerModel):
    def __init__(self):
        # Add new attention mechanism to existing architecture
        self.improved_attention = NewAttentionVariant()

# Creating new architecture
class YConvolution(Model):
    """Novel convolutional architecture"""
    pass

class YHybrid(Model):
    """New hybrid CNN-Transformer architecture"""
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
```

## Project Structure

```
yflow/
├── core/               # Core functionality
│   ├── context.py      # Computation context
│   ├── device.py       # Device abstraction (CPU/GPU)
│   ├── layer.py        # Base layer class
│   ├── model.py        # Base model class
│   └── shape_handler.py # Tensor shape operations
├── layers/             # Layer implementations
│   ├── activations.py  # Activation functions
│   ├── dense.py        # Fully connected layer
│   ├── dropout.py      # Dropout regularization
│   └── normalization.py # Batch normalization
├── losses/             # Loss functions
│   ├── cross_entropy.py # Cross entropy loss
│   └── mse.py          # Mean squared error loss
├── optimizers/         # Optimization algorithms
│   ├── adam.py         # Adam optimizer
│   ├── rmsprop.py      # RMSProp optimizer
│   └── sgd.py          # Stochastic gradient descent
├── yformers/           # Transformer architecture module
│   ├── attention.py    # Self-attention and multi-head attention
│   ├── embeddings.py   # Token and positional embeddings
│   ├── encoder.py      # Encoder blocks and components
│   ├── decoder.py      # Decoder blocks and components
│   ├── model.py        # Complete transformer models
│   └── utils.py        # Transformer utilities and masks
└── utils/              # Utility functions
    ├── lr_scheduler.py # Learning rate schedulers
    └── seq_norm.py     # Sequence normalization utilities
```

## YFormers - Transformer Architecture

YFormers is a comprehensive transformer architecture implementation built on top of YFlow's device abstraction and layer system. It provides all the essential components needed to build and train transformer models with seamless CPU/GPU support.

### Three Model Architectures

#### 1. Full Transformer (Encoder-Decoder)
Complete encoder-decoder transformer following "Attention Is All You Need" architecture.

```python
from yflow.yformers.model import TransformerModel

model = TransformerModel(
    src_vocab_size=10000,
    tgt_vocab_size=10000,
    d_model=512,
    num_heads=8,
    d_ff=2048,
    num_encoder_layers=6,
    num_decoder_layers=6,
    dropout=0.1,
    max_src_len=5000,
    max_tgt_len=5000
)

# Forward pass
logits = model.forward(src_tokens, tgt_tokens)

# Text generation with advanced sampling
generated = model.generate(src_tokens, max_len=100, temperature=0.8)
```

#### 2. Encoder-Only Model (BERT-style)
Encoder-only transformer for classification and feature extraction.

```python
from yflow.yformers.model import EncoderOnlyModel

model = EncoderOnlyModel(
    vocab_size=30000,
    d_model=768,
    num_heads=12,
    d_ff=3072,
    num_layers=12,
    num_classes=2  # For classification
)

# Classification
logits = model.forward(tokens)
```

#### 3. Decoder-Only Model (GPT-style)
Decoder-only transformer for autoregressive language generation.

```python
from yflow.yformers.model import DecoderOnlyModel

model = DecoderOnlyModel(
    vocab_size=50000,
    d_model=768,
    num_heads=12,
    d_ff=3072,
    num_layers=12,
    max_seq_len=1024
)

# Text generation with advanced sampling strategies
generated = model.generate(
    prompt_tokens, 
    max_len=200, 
    temperature=0.7,
    top_k=50,
    top_p=0.9
)
```

### Key YFormers Features

- **Complete Transformer Components**: Self-attention, multi-head attention, encoder/decoder blocks
- **Advanced Generation**: Text generation with temperature, top-k, and top-p sampling
- **Flexible Embeddings**: Token embeddings with both fixed and learnable positional encodings
- **Modern Implementations**: Pre-norm and post-norm architectures, GELU activation
- **Masking Support**: Padding masks, causal masks, and cross-attention masks
- **Device Abstraction**: Seamless CPU/GPU support through YFlow's device system

### Core Components

```python
from yflow.yformers import (
    SelfAttention, MultiHeadAttention,
    TokenEmbedding, PositionalEncoding,
    EncoderBlock, DecoderBlock,
    EncoderStack, DecoderStack
)

# Multi-head attention layer
mha = MultiHeadAttention(embed_dim=512, num_heads=8, dropout=0.1)

# Complete encoder stack
encoder = EncoderStack(
    embed_dim=512,
    num_heads=8,
    ff_dim=2048,
    num_layers=6,
    dropout=0.1
)
```

## GPU Support

YFlow is designed with GPU acceleration in mind, though this functionality is currently untested on actual GPU hardware. The library includes device abstraction that automatically falls back to CPU execution when a GPU is not available.

```python
from yflow.yformers import is_gpu_available, get_device_info

# Check GPU availability
if is_gpu_available():
    print("GPU is available for transformer operations")

# Move model to GPU
model.to('gpu')
```

## Training with YFlow and YFormers

Both traditional neural networks and transformer models can be trained using YFlow's unified training infrastructure:

```python
# Training a transformer model
from yflow.yformers.model import DecoderOnlyModel
from yflow.losses.cross_entropy import CrossEntropyLoss
from yflow.optimizers.adam import Adam

# Initialize language model
model = DecoderOnlyModel(vocab_size=vocab_size, d_model=512)
loss_fn = CrossEntropyLoss()
optimizer = Adam(learning_rate=0.0001)

# Training loop
for epoch in range(epochs):
    logits = model.forward(input_tokens)
    loss = loss_fn(logits, target_tokens)
    loss.backward()
    optimizer.step(model.parameters())
    optimizer.zero_grad()
```

## Contributing

**IMPORTANT: All contributions must be submitted to the Contribute branch only.**

### **Branch Structure**
- **Main Branch**: https://github.com/krauscode920/YFlow/tree/main (Protected - No direct contributions)
- **Contribute Branch**: https://github.com/krauscode920/YFlow/tree/Contribute (All PRs go here)

### **Contribution Process**
1. **Fork the repository** from the main branch
2. **Create your feature branch** from the Contribute branch
3. **Make your changes** following our naming conventions
4. **Submit pull request** to the **Contribute branch ONLY**
5. **Code review** and approval by maintainers
6. **Merge to main** branch upon approval

### **Contribution Guidelines**

#### **For Existing Architectures (YFormers, YLSTM, etc.)**
- Work within the existing architecture codebase
- No parallel implementations allowed
- Must maintain backward compatibility
- Requires thorough testing and documentation

#### **For New Architectures**
- Propose architecture with clear innovation/use case
- Follow YFlow naming conventions
- Provide comprehensive tests and documentation
- Subject to evaluation and approval

### **Areas for Contribution**
- **GPU testing and optimization** for both core YFlow and YFormers
- **Additional transformer architectures** within YFormers
- **Extended layer implementations**
- **Documentation improvements** and example notebooks
- **Performance optimizations**
- **Bug fixes and testing**

#### **Specific Help Needed:**
- **GPU Validation**: Test YFormers and core layers on GPU hardware
- **Architecture Extensions**: Add Vision Transformers, sparse attention to YFormers
- **Performance Benchmarking**: Compare against PyTorch implementations
- **Documentation**: Tutorial notebooks and educational content

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for detailed contribution guidelines.

## Future Plans

- **GPU Testing and Optimization**: Comprehensive testing and optimization on GPU hardware for both core layers and YFormers
- **Extended Architecture Library**: YLSTM, YQuence, BiYQuence implementations
- **Training Utilities**: Data loaders, augmentation, and training loops
- **Advanced YFormers Features**: Vision transformers, sparse attention mechanisms within the unified YFormers architecture
- **Model Zoo**: Pre-trained transformer models and architectures

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

YFlow was created as an educational project to deeply understand deep learning frameworks and their implementation details. YFormers extends this educational mission to transformer architectures, providing a complete implementation of modern attention-based models. The project is not intended for production use but rather as a learning tool and research platform.

## Version

Current version: 0.2.0 (with YFormers integration and unified governance model)
