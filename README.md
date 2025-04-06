YFlow: GPU-Compatible Deep Learning Library Built From Scratch
YFlow is a custom deep learning framework built entirely from scratch with no dependencies on existing ML libraries. It supports both CPU and GPU execution and provides a clean, intuitive API while maintaining flexibility for advanced deep learning research and applications.
Core Principles

Zero External ML Dependencies: Built completely from first principles without using TensorFlow, PyTorch, or other ML libraries
'Y' Naming Convention: All modules and components follow a distinctive naming convention starting with 'Y'
Educational Purpose: Designed to understand deep learning fundamentals by implementing everything from scratch

Features

CPU and GPU Support: Designed with hardware acceleration in mind (GPU support implementation included but currently untested)
Modular Architecture: Well-organized structure separated into core functionality, layers, losses, optimizers, and utilities
Automatic Differentiation: Built-in gradient computation
Customizable Layers: Implement your own or use provided implementations
Optimizers: Standard optimization algorithms including SGD, Adam, and RMSProp
Device Abstraction: Clean separation between compute logic and hardware acceleration

Installation
bashCopy# Clone the repository
git clone https://github.com/krauscode920/YFlow.git

# Install dependencies
cd YFlow
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
Quick Start Example
pythonCopyfrom yflow.core.model import Model
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
└── utils/              # Utility functions
    ├── lr_scheduler.py # Learning rate schedulers
    └── seq_norm.py     # Sequence normalization utilities




GPU Support
YFlow is designed with GPU acceleration in mind, though this functionality is currently untested on actual GPU hardware. The library includes device abstraction that automatically falls back to CPU execution when a GPU is not available.
Future Plans

Transformer Architecture: Development of YFormers, a transformer-based module built on top of YFlow
GPU Testing and Optimization: Comprehensive testing and optimization on GPU hardware
Extended Layer Library: Additional layer types and activation functions
Training Utilities: Data loaders, augmentation, and training loops

Contributing
Contributions are welcome! Areas where we'd particularly appreciate help:

GPU testing and optimization
Transformer architecture development
Additional layer implementations
Documentation improvements
Example notebooks

Please see CONTRIBUTING.md for details on how to contribute.
License
This project is licensed under the MIT License - see the LICENSE file for details.
Acknowledgements
YFlow was created as an educational project to deeply understand deep learning frameworks and their implementation details. It is not intended for production use but rather as a learning tool and research platform.
