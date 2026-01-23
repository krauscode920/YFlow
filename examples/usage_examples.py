# usage_examples.py
"""
Usage examples for Model.save() and Model.load()

Shows how to use the new save/load functionality with different model types.
"""

import numpy as np
from yflow.core.model import Model
from yflow.layers.dense import Dense
from yflow.layers.lstm import YSTM
from yflow.layers.rnn import YQuence
from yflow.optimizers.adam import Adam
from yflow.losses.mse import MSELoss


print("=" * 70)
print("YFLOW MODEL SAVE/LOAD USAGE EXAMPLES")
print("=" * 70)
print()


# Example 1: Simple Save/Load
print("📝 Example 1: Basic Save/Load")
print("-" * 70)

# Create and train model
model = Model()
model.add(Dense(input_size=10, output_size=5))
model.add(Dense(input_size=5, output_size=2))
model.compile(loss=MSELoss(), optimizer=Adam())

# Train on some data
X = np.random.randn(100, 10)
y = np.random.randn(100, 2)
history = model.train(X, y, epochs=5, batch_size=32, verbose=0)

# Save the model
model.save('my_model.npz')

# Create a new model with same architecture
new_model = Model()
new_model.add(Dense(input_size=10, output_size=5))
new_model.add(Dense(input_size=5, output_size=2))
new_model.compile(loss=MSELoss(), optimizer=Adam())

# Load the saved weights
metadata = new_model.load('my_model.npz')

print("✓ Model saved and loaded successfully!")
print()


# Example 2: Save with Optimizer and Metadata
print("📝 Example 2: Save with Optimizer State and Metadata")
print("-" * 70)

model = Model()
model.add(Dense(input_size=10, output_size=3))
optimizer = Adam(learning_rate=0.001)
model.compile(loss=MSELoss(), optimizer=optimizer)

# Train for a bit
X = np.random.randn(50, 10)
y = np.random.randn(50, 3)
history = model.train(X, y, epochs=3, batch_size=16, verbose=0)

# Save with metadata
model.save(
    'trained_model.npz',
    optimizer=optimizer,
    epoch=3,
    loss=history['train_loss'][-1],
    metrics={'accuracy': 0.85, 'val_loss': 0.234}
)

# Load into new model
new_model = Model()
new_model.add(Dense(input_size=10, output_size=3))
new_optimizer = Adam(learning_rate=0.001)
new_model.compile(loss=MSELoss(), optimizer=new_optimizer)

metadata = new_model.load('trained_model.npz', optimizer=new_optimizer)

print(f"Loaded model:")
print(f"  Epoch: {metadata['epoch']}")
print(f"  Loss: {metadata['loss']:.4f}")
print(f"  Metrics: {metadata['metrics']}")
print()


# Example 3: Resume Training from Checkpoint
print("📝 Example 3: Resume Training from Checkpoint")
print("-" * 70)

# Initial training
model = Model()
model.add(Dense(input_size=5, output_size=3))
optimizer = Adam()
model.compile(loss=MSELoss(), optimizer=optimizer)

X = np.random.randn(100, 5)
y = np.random.randn(100, 3)

# Train for 5 epochs
history1 = model.train(X, y, epochs=5, batch_size=32, verbose=0)

# Save checkpoint
model.save('checkpoint.npz', optimizer=optimizer, epoch=5,
          loss=history1['train_loss'][-1])

# Later... create new model and resume training
resume_model = Model()
resume_model.add(Dense(input_size=5, output_size=3))
resume_optimizer = Adam()
resume_model.compile(loss=MSELoss(), optimizer=resume_optimizer)

# Load checkpoint
metadata = resume_model.load('checkpoint.npz', optimizer=resume_optimizer)
start_epoch = metadata['epoch']

print(f"Resuming from epoch {start_epoch}")

# Continue training for 5 more epochs
history2 = resume_model.train(X, y, epochs=5, batch_size=32, verbose=0)

print(f"✓ Training resumed successfully!")
print()


# Example 4: Save LSTM Model
print("📝 Example 4: Save/Load LSTM Model")
print("-" * 70)

# Create LSTM model
# Note: LSTM outputs hidden_size when return_sequences=False
lstm_model = Model()
lstm_model.add(YSTM(input_size=10, hidden_size=20, return_sequences=False))
lstm_model.add(Dense(input_size=20, output_size=3))  # Input must match hidden_size
lstm_model.compile(loss=MSELoss(), optimizer=Adam())

# Generate sequence data
X_seq = np.random.randn(50, 8, 10)  # (batch, seq_len, features)
y_seq = np.random.randn(50, 3)

# Train
history = lstm_model.train(X_seq, y_seq, epochs=3, batch_size=16, verbose=0)

# Save
lstm_model.save('lstm_model.npz', epoch=3, loss=history['train_loss'][-1])

# Load into new model
new_lstm = Model()
new_lstm.add(YSTM(input_size=10, hidden_size=20, return_sequences=False))
new_lstm.add(Dense(input_size=20, output_size=3))
new_lstm.compile(loss=MSELoss(), optimizer=Adam())

metadata = new_lstm.load('lstm_model.npz')

print(f"✓ LSTM model saved and loaded!")
print(f"  Final loss: {metadata['loss']:.4f}")
print()


# Example 5: Save RNN Model
print("📝 Example 5: Save/Load RNN Model")
print("-" * 70)

# Create RNN model
# Note: RNN uses output_size parameter for final output dimension
rnn_model = Model()
rnn_model.add(YQuence(input_size=8, hidden_size=16, output_size=10))
rnn_model.add(Dense(input_size=10, output_size=2))
rnn_model.compile(loss=MSELoss(), optimizer=Adam())

# Generate data
X_rnn = np.random.randn(40, 6, 8)  # (batch, seq_len, features)
y_rnn = np.random.randn(40, 2)

# Train
history = rnn_model.train(X_rnn, y_rnn, epochs=3, batch_size=16, verbose=0)

# Save
rnn_model.save('rnn_model.npz', epoch=3, loss=history['train_loss'][-1])

# Load
new_rnn = Model()
new_rnn.add(YQuence(input_size=8, hidden_size=16, output_size=10))
new_rnn.add(Dense(input_size=10, output_size=2))
new_rnn.compile(loss=MSELoss(), optimizer=Adam())

metadata = new_rnn.load('rnn_model.npz')

print(f"✓ RNN model saved and loaded!")
print(f"  Final loss: {metadata['loss']:.4f}")
print()


# Example 6: Inference after Loading
print("📝 Example 6: Inference After Loading")
print("-" * 70)

# Train a model
model = Model()
model.add(Dense(input_size=10, output_size=5))
model.add(Dense(input_size=5, output_size=2))
model.compile(loss=MSELoss(), optimizer=Adam())

X_train = np.random.randn(100, 10)
y_train = np.random.randn(100, 2)
model.train(X_train, y_train, epochs=5, batch_size=32, verbose=0)

# Save
model.save('inference_model.npz')

# Later... load for inference only (no optimizer needed)
inference_model = Model()
inference_model.add(Dense(input_size=10, output_size=5))
inference_model.add(Dense(input_size=5, output_size=2))
inference_model.compile(loss=MSELoss(), optimizer=Adam())

inference_model.load('inference_model.npz', load_optimizer=False)

# Make predictions
X_test = np.random.randn(10, 10)
predictions = inference_model.predict(X_test)

print(f"✓ Made predictions on {len(predictions)} samples")
print(f"  Prediction shape: {predictions.shape}")
print()


print("=" * 70)
print("🎉 ALL EXAMPLES COMPLETED!")
print("=" * 70)
print()
print("Summary of what you can do:")
print("  • model.save('file.npz') - Simple save")
print("  • model.save('file.npz', optimizer=opt, epoch=10) - Save with metadata")
print("  • model.load('file.npz') - Simple load")
print("  • model.load('file.npz', optimizer=opt) - Load with optimizer state")
print()
print("Works with:")
print("  ✓ Regular NNs (Dense layers)")
print("  ✓ RNNs (YQuence)")
print("  ✓ LSTMs (YSTM)")
print("  ✓ Transformers (TransformerModel, etc.)")