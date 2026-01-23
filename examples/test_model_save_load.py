# test_model_save_load.py
"""
Test suite for Model.save() and Model.load() functionality.

Tests that models can be saved and loaded correctly.
"""

import numpy as np
import os
from yflow.core.model import Model
from yflow.layers.dense import Dense
from yflow.layers.lstm import YSTM
from yflow.layers.rnn import YQuence
from yflow.optimizers.adam import Adam
from yflow.losses.mse import MSELoss


def cleanup_test_files():
    """Remove test model files"""
    test_files = [
        'test_nn_model.npz',
        'test_rnn_model.npz',
        'test_lstm_model.npz',
        'test_with_optimizer.npz'
    ]
    for f in test_files:
        if os.path.exists(f):
            os.remove(f)


def test_simple_nn_save_load():
    """Test save/load for simple feedforward network"""
    print("🧪 Test 1: Simple NN Save/Load")
    
    # Create model
    model = Model()
    model.add(Dense(input_size=10, output_size=5))
    model.add(Dense(input_size=5, output_size=2))
    
    # Compile
    model.compile(loss=MSELoss(), optimizer=Adam())
    
    # Get original parameters
    original_params = model.get_parameters()
    
    # Save model
    model.save('test_nn_model.npz', epoch=5, loss=2.5)
    
    # Create new model with same architecture
    model2 = Model()
    model2.add(Dense(input_size=10, output_size=5))
    model2.add(Dense(input_size=5, output_size=2))
    model2.compile(loss=MSELoss(), optimizer=Adam())
    
    # Load saved weights
    metadata = model2.load('test_nn_model.npz')
    
    print(f"  Loaded metadata: epoch={metadata.get('epoch')}, loss={metadata.get('loss')}")
    
    # Verify parameters match
    loaded_params = model2.get_parameters()
    
    for key in original_params:
        if key in loaded_params:
            assert np.allclose(original_params[key], loaded_params[key]), f"Mismatch in {key}"
    
    print("  ✅ Simple NN save/load works!\n")


def test_rnn_save_load():
    """Test save/load for RNN model"""
    print("🧪 Test 2: RNN Save/Load")
    
    # Create RNN model
    model = Model()
    model.add(YQuence(input_size=10, hidden_size=20, output_size=5))
    model.add(Dense(input_size=5, output_size=2))
    
    model.compile(loss=MSELoss(), optimizer=Adam())
    
    # Get original parameters
    original_params = model.get_parameters()
    
    # Save model
    model.save('test_rnn_model.npz', epoch=3, step=150)
    
    # Create new model
    model2 = Model()
    model2.add(YQuence(input_size=10, hidden_size=20, output_size=5))
    model2.add(Dense(input_size=5, output_size=2))
    model2.compile(loss=MSELoss(), optimizer=Adam())
    
    # Load
    metadata = model2.load('test_rnn_model.npz')
    
    print(f"  Loaded metadata: epoch={metadata.get('epoch')}, step={metadata.get('step')}")
    
    # Verify
    loaded_params = model2.get_parameters()
    
    for key in original_params:
        if key in loaded_params:
            assert np.allclose(original_params[key], loaded_params[key]), f"Mismatch in {key}"
    
    print("  ✅ RNN save/load works!\n")


def test_lstm_save_load():
    """Test save/load for LSTM model"""
    print("🧪 Test 3: LSTM Save/Load")
    
    # Create LSTM model
    model = Model()
    model.add(YSTM(input_size=10, hidden_size=20, output_size=15))
    model.add(Dense(input_size=15, output_size=3))
    
    model.compile(loss=MSELoss(), optimizer=Adam())
    
    # Get original parameters
    original_params = model.get_parameters()
    
    # Save model
    model.save('test_lstm_model.npz', epoch=7, loss=1.8)
    
    # Create new model
    model2 = Model()
    model2.add(YSTM(input_size=10, hidden_size=20, output_size=15))
    model2.add(Dense(input_size=15, output_size=3))
    model2.compile(loss=MSELoss(), optimizer=Adam())
    
    # Load
    metadata = model2.load('test_lstm_model.npz')
    
    print(f"  Loaded metadata: epoch={metadata.get('epoch')}, loss={metadata.get('loss')}")
    
    # Verify
    loaded_params = model2.get_parameters()
    
    for key in original_params:
        if key in loaded_params:
            assert np.allclose(original_params[key], loaded_params[key]), f"Mismatch in {key}"
    
    print("  ✅ LSTM save/load works!\n")


def test_save_with_optimizer():
    """Test save/load with optimizer state"""
    print("🧪 Test 4: Save/Load with Optimizer State")
    
    # Create model
    model = Model()
    model.add(Dense(input_size=5, output_size=3))
    
    # Create optimizer
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss=MSELoss(), optimizer=optimizer)
    
    # Simulate some training (to create optimizer state)
    X = np.random.randn(10, 5)
    y = np.random.randn(10, 3)
    
    # One training step to initialize optimizer state
    predictions = model._forward_pass(X, training=True)
    loss = model.loss.calculate(predictions, y)
    grad = model.loss.derivative(predictions, y)
    model._backward_pass(grad)
    model._update_params()
    
    # Save with optimizer
    model.save('test_with_optimizer.npz', optimizer=optimizer, epoch=1, loss=loss)
    
    # Create new model and optimizer
    model2 = Model()
    model2.add(Dense(input_size=5, output_size=3))
    optimizer2 = Adam(learning_rate=0.001)
    model2.compile(loss=MSELoss(), optimizer=optimizer2)
    
    # Load with optimizer
    metadata = model2.load('test_with_optimizer.npz', optimizer=optimizer2)
    
    print(f"  Loaded with optimizer state")
    print(f"  Epoch: {metadata.get('epoch')}, Loss: {metadata.get('loss'):.4f}")
    
    # Verify optimizer state was loaded
    assert optimizer2.t > 0, "Optimizer state not loaded correctly"
    
    print("  ✅ Save/load with optimizer works!\n")


def test_inference_after_load():
    """Test that loaded model can perform inference"""
    print("🧪 Test 5: Inference After Load")
    
    # Create and train a simple model
    model = Model()
    model.add(Dense(input_size=10, output_size=5))
    model.add(Dense(input_size=5, output_size=2))
    model.compile(loss=MSELoss(), optimizer=Adam())
    
    # Test input
    X_test = np.random.randn(5, 10)
    
    # Get predictions from original model
    original_predictions = model.predict(X_test)
    
    # Save model
    model.save('test_inference.npz')
    
    # Create new model and load
    model2 = Model()
    model2.add(Dense(input_size=10, output_size=5))
    model2.add(Dense(input_size=5, output_size=2))
    model2.compile(loss=MSELoss(), optimizer=Adam())
    model2.load('test_inference.npz')
    
    # Get predictions from loaded model
    loaded_predictions = model2.predict(X_test)
    
    # Verify predictions match
    assert np.allclose(original_predictions, loaded_predictions), "Predictions don't match!"
    
    print("  Original predictions shape:", original_predictions.shape)
    print("  Loaded predictions shape:", loaded_predictions.shape)
    print("  Predictions match: ✓")
    print("  ✅ Inference after load works!\n")
    
    # Cleanup
    if os.path.exists('test_inference.npz'):
        os.remove('test_inference.npz')


def run_all_tests():
    """Run all model save/load tests"""
    print("=" * 70)
    print("MODEL SAVE/LOAD TEST SUITE")
    print("=" * 70)
    print()
    
    try:
        # Clean up any existing test files
        cleanup_test_files()
        
        # Run tests
        test_simple_nn_save_load()
        test_rnn_save_load()
        test_lstm_save_load()
        test_save_with_optimizer()
        test_inference_after_load()
        
        print("=" * 70)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 70)
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        raise
    finally:
        # Cleanup test files
        cleanup_test_files()


if __name__ == "__main__":
    run_all_tests()
