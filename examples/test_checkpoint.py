# test_checkpoint.py
"""
Comprehensive test suite for YFlow checkpointing.

Tests save/load, checkpoint manager, and edge cases.
"""

import os
import shutil
import numpy as np
from yflow.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    list_checkpoints,
    get_latest_checkpoint,
    delete_old_checkpoints,
    get_checkpoint_info,
    CheckpointManager
)


# Mock Model and Optimizer for testing
class MockModel:
    """Mock model for testing"""
    
    def __init__(self):
        self.weights = {
            'layer1': np.random.randn(10, 5),
            'layer2': np.random.randn(5, 3),
            'bias1': np.random.randn(5),
            'bias2': np.random.randn(3)
        }
    
    def get_parameters(self):
        """Return model parameters"""
        return self.weights
    
    def set_parameters(self, params):
        """Set model parameters"""
        self.weights = params


class MockOptimizer:
    """Mock optimizer for testing"""
    
    def __init__(self):
        self.state = {
            'momentum': np.random.randn(10, 5),
            'velocity': np.random.randn(5, 3),
            'learning_rate': 0.001,
            'beta1': 0.9,
            'beta2': 0.999
        }
    
    def get_state(self):
        """Return optimizer state"""
        return self.state
    
    def set_state(self, state):
        """Set optimizer state"""
        self.state = state


def setup_test_dir():
    """Create test checkpoint directory"""
    test_dir = 'test_checkpoints'
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)
    return test_dir


def cleanup_test_dir():
    """Remove test checkpoint directory"""
    test_dir = 'test_checkpoints'
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)


def test_basic_save_load():
    """Test basic checkpoint save and load"""
    print("🧪 Test 1: Basic Save and Load")
    
    test_dir = setup_test_dir()
    
    # Create model and optimizer
    model = MockModel()
    optimizer = MockOptimizer()
    
    # Save original weights
    original_weights = {k: v.copy() for k, v in model.weights.items()}
    original_state = {k: v.copy() if isinstance(v, np.ndarray) else v 
                     for k, v in optimizer.state.items()}
    
    # Save checkpoint
    filepath = os.path.join(test_dir, 'test_checkpoint.npz')
    save_checkpoint(
        filepath=filepath,
        model=model,
        optimizer=optimizer,
        epoch=5,
        step=1000,
        loss=2.345
    )
    
    print(f"  Checkpoint saved: {os.path.exists(filepath)}")
    
    # Modify model and optimizer (simulate continued training)
    model.weights['layer1'] += 1.0
    optimizer.state['momentum'] += 0.5
    
    # Load checkpoint
    checkpoint = load_checkpoint(filepath)
    
    # Restore
    model.set_parameters(checkpoint['model_parameters'])
    optimizer.set_state(checkpoint['optimizer_state'])
    
    # Verify restoration
    for key in original_weights:
        assert np.allclose(model.weights[key], original_weights[key])
    
    for key in original_state:
        if isinstance(original_state[key], np.ndarray):
            assert np.allclose(optimizer.state[key], original_state[key])
        else:
            assert optimizer.state[key] == original_state[key]
    
    # Verify metadata
    assert checkpoint['metadata']['epoch'] == 5
    assert checkpoint['metadata']['step'] == 1000
    assert checkpoint['metadata']['loss'] == 2.345
    
    print("  ✅ Save and load works!\n")
    
    cleanup_test_dir()


def test_metadata_only():
    """Test saving and loading metadata only"""
    print("🧪 Test 2: Metadata Only")
    
    test_dir = setup_test_dir()
    
    model = MockModel()
    
    # Save checkpoint with just metadata
    filepath = os.path.join(test_dir, 'metadata_checkpoint.npz')
    save_checkpoint(
        filepath=filepath,
        model=model,
        epoch=10,
        step=5000,
        loss=1.234,
        train_acc=0.95,
        val_acc=0.92
    )
    
    # Get metadata without loading full checkpoint
    info = get_checkpoint_info(filepath)
    
    print(f"  Epoch: {info['epoch']}")
    print(f"  Step: {info['step']}")
    print(f"  Loss: {info['loss']}")
    print(f"  Train Acc: {info['train_acc']}")
    print(f"  Val Acc: {info['val_acc']}")
    
    assert info['epoch'] == 10
    assert info['step'] == 5000
    assert info['loss'] == 1.234
    assert info['train_acc'] == 0.95
    
    print("  ✅ Metadata loading works!\n")
    
    cleanup_test_dir()


def test_list_and_latest():
    """Test listing checkpoints and getting latest"""
    print("🧪 Test 3: List and Get Latest Checkpoint")
    
    test_dir = setup_test_dir()
    model = MockModel()
    
    # Save multiple checkpoints
    import time
    for i in range(5):
        filepath = os.path.join(test_dir, f'checkpoint_{i}.npz')
        save_checkpoint(filepath=filepath, model=model, epoch=i)
        time.sleep(0.01)  # Ensure different timestamps
    
    # List checkpoints
    checkpoints = list_checkpoints(test_dir)
    
    print(f"  Number of checkpoints: {len(checkpoints)}")
    print(f"  Checkpoints: {[os.path.basename(c) for c in checkpoints]}")
    
    assert len(checkpoints) == 5
    
    # Get latest checkpoint
    latest = get_latest_checkpoint(test_dir)
    
    print(f"  Latest checkpoint: {os.path.basename(latest)}")
    
    assert 'checkpoint_4' in latest  # Last one saved
    
    print("  ✅ List and latest works!\n")
    
    cleanup_test_dir()


def test_delete_old_checkpoints():
    """Test automatic deletion of old checkpoints"""
    print("🧪 Test 4: Delete Old Checkpoints")
    
    test_dir = setup_test_dir()
    model = MockModel()
    
    # Save 10 checkpoints
    import time
    for i in range(10):
        filepath = os.path.join(test_dir, f'checkpoint_{i}.npz')
        save_checkpoint(filepath=filepath, model=model, epoch=i)
        time.sleep(0.01)
    
    print(f"  Checkpoints before cleanup: {len(list_checkpoints(test_dir))}")
    
    # Keep only last 3
    delete_old_checkpoints(test_dir, keep_last_n=3)
    
    remaining = list_checkpoints(test_dir)
    print(f"  Checkpoints after cleanup: {len(remaining)}")
    print(f"  Remaining: {[os.path.basename(c) for c in remaining]}")
    
    assert len(remaining) == 3
    
    print("  ✅ Old checkpoint deletion works!\n")
    
    cleanup_test_dir()


def test_checkpoint_manager_basic():
    """Test CheckpointManager basic functionality"""
    print("🧪 Test 5: CheckpointManager Basic Usage")
    
    test_dir = setup_test_dir()
    model = MockModel()
    optimizer = MockOptimizer()
    
    # Create checkpoint manager
    manager = CheckpointManager(
        checkpoint_dir=test_dir,
        save_every_n_steps=100,
        keep_last_n=3,
        track_best=False
    )
    
    # Simulate training loop
    for step in range(350):
        loss = 5.0 - step * 0.01  # Decreasing loss
        
        saved = manager.step(
            model=model,
            optimizer=optimizer,
            step=step,
            loss=loss
        )
        
        if saved:
            print(f"    Checkpoint saved at step {step}")
    
    # Check number of checkpoints
    checkpoints = list_checkpoints(test_dir)
    print(f"  Total checkpoints: {len(checkpoints)}")
    
    # Should have 3 checkpoints (kept last 3 of 4 saves at steps 100, 200, 300)
    assert len(checkpoints) == 3
    
    print("  ✅ CheckpointManager basic usage works!\n")
    
    cleanup_test_dir()


def test_checkpoint_manager_best_tracking():
    """Test CheckpointManager best model tracking"""
    print("🧪 Test 6: CheckpointManager Best Model Tracking")
    
    test_dir = setup_test_dir()
    model = MockModel()
    optimizer = MockOptimizer()
    
    # Create checkpoint manager with best tracking
    manager = CheckpointManager(
        checkpoint_dir=test_dir,
        save_every_n_steps=1000,  # Won't trigger in this test
        keep_last_n=5,
        track_best=True,
        best_metric='loss',
        best_mode='min'
    )
    
    # Simulate training with varying loss
    losses = [5.0, 4.5, 4.8, 3.9, 4.2, 3.5, 4.0]  # Best is 3.5 at step 5
    
    for step, loss in enumerate(losses):
        manager.step(
            model=model,
            optimizer=optimizer,
            step=step,
            loss=loss
        )
    
    # Check if best model was saved
    best_checkpoint = os.path.join(test_dir, 'best_model.npz')
    assert os.path.exists(best_checkpoint)
    
    # Verify best model metadata
    info = get_checkpoint_info(best_checkpoint)
    print(f"  Best model loss: {info['loss']}")
    print(f"  Best model step: {info['step']}")
    
    assert info['loss'] == 3.5
    assert info['step'] == 5
    
    print("  ✅ Best model tracking works!\n")
    
    cleanup_test_dir()


def test_checkpoint_manager_restore():
    """Test CheckpointManager restore functionality"""
    print("🧪 Test 7: CheckpointManager Restore")
    
    test_dir = setup_test_dir()
    model = MockModel()
    optimizer = MockOptimizer()
    
    # Save original weights
    original_weights = {k: v.copy() for k, v in model.weights.items()}
    
    # Create manager and save checkpoint
    manager = CheckpointManager(checkpoint_dir=test_dir)
    
    save_checkpoint(
        filepath=os.path.join(test_dir, 'checkpoint_step_100.npz'),
        model=model,
        optimizer=optimizer,
        step=100,
        epoch=5,
        loss=2.5
    )
    
    # Modify model
    model.weights['layer1'] += 10.0
    
    # Restore latest checkpoint
    metadata = manager.restore_latest(model, optimizer)
    
    print(f"  Restored from step: {metadata['step']}")
    print(f"  Restored from epoch: {metadata['epoch']}")
    print(f"  Restored loss: {metadata['loss']}")
    
    # Verify restoration
    for key in original_weights:
        assert np.allclose(model.weights[key], original_weights[key])
    
    assert metadata['step'] == 100
    assert metadata['epoch'] == 5
    
    print("  ✅ Checkpoint restoration works!\n")
    
    cleanup_test_dir()


def test_epoch_checkpoint():
    """Test epoch-based checkpointing"""
    print("🧪 Test 8: Epoch Checkpointing")
    
    test_dir = setup_test_dir()
    model = MockModel()
    optimizer = MockOptimizer()
    
    manager = CheckpointManager(
        checkpoint_dir=test_dir,
        keep_last_n=2
    )
    
    # Save checkpoints for multiple epochs
    for epoch in range(5):
        manager.save_epoch_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            loss=5.0 - epoch * 0.5
        )
    
    # List epoch checkpoints
    checkpoints = list_checkpoints(test_dir, pattern='checkpoint_epoch_*.npz')
    
    print(f"  Epoch checkpoints: {len(checkpoints)}")
    print(f"  Files: {[os.path.basename(c) for c in checkpoints]}")
    
    # Should keep only last 2
    assert len(checkpoints) == 2
    
    print("  ✅ Epoch checkpointing works!\n")
    
    cleanup_test_dir()


def test_custom_metrics():
    """Test saving custom metrics"""
    print("🧪 Test 9: Custom Metrics")
    
    test_dir = setup_test_dir()
    model = MockModel()
    
    # Save checkpoint with custom metrics
    filepath = os.path.join(test_dir, 'metrics_checkpoint.npz')
    
    custom_metrics = {
        'train_accuracy': 0.95,
        'val_accuracy': 0.92,
        'train_perplexity': 15.3,
        'val_perplexity': 18.7,
        'learning_rate': 0.0001
    }
    
    save_checkpoint(
        filepath=filepath,
        model=model,
        epoch=10,
        step=5000,
        loss=2.1,
        metrics=custom_metrics,
        custom_field='test_value'  # Additional kwarg
    )
    
    # Load and verify
    checkpoint = load_checkpoint(filepath)
    metadata = checkpoint['metadata']
    
    print(f"  Saved metrics:")
    print(f"    Train Acc: {metadata['metrics']['train_accuracy']}")
    print(f"    Val Acc: {metadata['metrics']['val_accuracy']}")
    print(f"    Custom field: {metadata['custom_field']}")
    
    assert metadata['metrics']['train_accuracy'] == 0.95
    assert metadata['metrics']['val_perplexity'] == 18.7
    assert metadata['custom_field'] == 'test_value'
    
    print("  ✅ Custom metrics work!\n")
    
    cleanup_test_dir()


def run_all_tests():
    """Run all checkpoint tests"""
    print("=" * 70)
    print("YFLOW CHECKPOINT TEST SUITE")
    print("=" * 70)
    print()
    
    try:
        test_basic_save_load()
        test_metadata_only()
        test_list_and_latest()
        test_delete_old_checkpoints()
        test_checkpoint_manager_basic()
        test_checkpoint_manager_best_tracking()
        test_checkpoint_manager_restore()
        test_epoch_checkpoint()
        test_custom_metrics()
        
        print("=" * 70)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 70)
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()
