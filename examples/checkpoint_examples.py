# checkpoint_examples.py
"""
Comprehensive examples showing how to use YFlow checkpointing.
"""

import numpy as np
from yflow.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    CheckpointManager
)


# Mock classes for examples
class SimpleModel:
    """Simple model for examples"""

    def __init__(self):
        self.weights = {'layer1': np.random.randn(10, 5)}

    def get_parameters(self):
        return self.weights

    def set_parameters(self, params):
        self.weights = params


class SimpleOptimizer:
    """Simple optimizer for examples"""

    def __init__(self):
        self.state = {'momentum': np.random.randn(10, 5), 'lr': 0.001}

    def get_state(self):
        return self.state

    def set_state(self, state):
        self.state = state


# ============================================================================
# Example 1: Basic Manual Checkpointing
# ============================================================================

def example_basic_checkpoint():
    """Basic checkpoint save and load"""
    print("Example 1: Basic Manual Checkpointing")
    print("-" * 50)

    model = SimpleModel()
    optimizer = SimpleOptimizer()

    # Training loop
    for epoch in range(10):
        # ... training code ...
        loss = 5.0 - epoch * 0.3  # Simulate decreasing loss

        # Save checkpoint at end of each epoch
        if epoch % 2 == 0:  # Save every 2 epochs
            save_checkpoint(
                filepath=f'checkpoints/manual_epoch_{epoch}.npz',
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=loss
            )
            print(f"  Saved checkpoint at epoch {epoch}")

    # Later: Load the checkpoint
    print("\n  Loading checkpoint...")
    checkpoint = load_checkpoint('checkpoints/manual_epoch_8.npz')

    model.set_parameters(checkpoint['model_parameters'])
    optimizer.set_state(checkpoint['optimizer_state'])

    start_epoch = checkpoint['metadata']['epoch']
    print(f"  Resumed from epoch {start_epoch}")
    print()


# ============================================================================
# Example 2: CheckpointManager for Automatic Saving
# ============================================================================

def example_checkpoint_manager():
    """Using CheckpointManager for automatic checkpointing"""
    print("Example 2: CheckpointManager (Automatic)")
    print("-" * 50)

    model = SimpleModel()
    optimizer = SimpleOptimizer()

    # Create checkpoint manager
    manager = CheckpointManager(
        checkpoint_dir='checkpoints/',
        save_every_n_steps=1000,  # Save every 1000 steps
        keep_last_n=5,  # Keep only 5 most recent
        track_best=True  # Track best model
    )

    print("  Training with automatic checkpointing...")

    # Training loop
    step = 0
    for epoch in range(3):
        for batch in range(100):  # 100 batches per epoch
            # ... training code ...
            loss = 5.0 - step * 0.001  # Simulate decreasing loss

            # Automatically saves if needed
            manager.step(
                model=model,
                optimizer=optimizer,
                step=step,
                epoch=epoch,
                loss=loss
            )

            step += 1

        # Save at end of epoch
        manager.save_epoch_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            step=step,
            loss=loss
        )

    print("  Training complete!")
    print()


# ============================================================================
# Example 3: Resuming Training from Checkpoint
# ============================================================================

def example_resume_training():
    """Resume training from checkpoint"""
    print("Example 3: Resume Training from Checkpoint")
    print("-" * 50)

    model = SimpleModel()
    optimizer = SimpleOptimizer()

    manager = CheckpointManager('checkpoints/')

    # Try to restore latest checkpoint
    metadata = manager.restore_latest(model, optimizer)

    if metadata:
        start_epoch = metadata.get('epoch', 0) + 1
        start_step = metadata.get('step', 0)
        print(f"  Resuming from epoch {start_epoch}, step {start_step}")
    else:
        start_epoch = 0
        start_step = 0
        print("  Starting training from scratch")

    # Continue training
    step = start_step
    for epoch in range(start_epoch, 10):
        for batch in range(100):
            # ... training code ...
            loss = 5.0 - step * 0.001

            manager.step(
                model=model,
                optimizer=optimizer,
                step=step,
                epoch=epoch,
                loss=loss
            )

            step += 1

    print("  Training complete!")
    print()


# ============================================================================
# Example 4: Best Model Tracking
# ============================================================================

def example_best_model_tracking():
    """Track and save best model"""
    print("Example 4: Best Model Tracking")
    print("-" * 50)

    model = SimpleModel()
    optimizer = SimpleOptimizer()

    # Manager with best model tracking
    manager = CheckpointManager(
        checkpoint_dir='checkpoints/',
        save_every_n_steps=10000,  # Don't save regular checkpoints for this example
        track_best=True,
        best_metric='val_loss',
        best_mode='min'
    )

    # Training with validation
    for epoch in range(10):
        # Training
        train_loss = 3.0 - epoch * 0.2

        # Validation
        val_loss = 3.5 - epoch * 0.15 + np.random.uniform(-0.1, 0.1)

        print(f"  Epoch {epoch}: train_loss={train_loss:.3f}, val_loss={val_loss:.3f}")

        # Check if this is the best model
        manager.step(
            model=model,
            optimizer=optimizer,
            step=epoch * 1000,
            epoch=epoch,
            loss=train_loss,
            metrics={'val_loss': val_loss}
        )

    # Load best model at the end
    print("\n  Loading best model...")
    best_metadata = manager.restore_best(model, optimizer)

    if best_metadata:
        print(f"  Best model from epoch {best_metadata['epoch']}")
        print(f"  Best val_loss: {best_metadata['metrics']['val_loss']:.3f}")

    print()


# ============================================================================
# Example 5: Language Model Training with Checkpointing
# ============================================================================

def example_language_model_training():
    """Realistic language model training scenario"""
    print("Example 5: Language Model Training")
    print("-" * 50)

    # Simulate language model and optimizer
    model = SimpleModel()
    optimizer = SimpleOptimizer()

    # Setup checkpoint manager
    manager = CheckpointManager(
        checkpoint_dir='checkpoints/lm_training/',
        save_every_n_steps=5000,
        keep_last_n=3,
        track_best=True,
        best_metric='val_perplexity',
        best_mode='min'
    )

    # Try to resume from checkpoint
    metadata = manager.restore_latest(model, optimizer)
    start_step = metadata.get('step', 0) if metadata else 0

    print(f"  Starting/resuming from step {start_step}")

    # Training loop
    for step in range(start_step, 20000):
        # Simulate training
        train_loss = 5.0 - step * 0.0002

        # Periodic validation
        if step % 1000 == 0:
            val_loss = train_loss + 0.3
            val_perplexity = np.exp(val_loss)

            print(f"    Step {step}: "
                  f"train_loss={train_loss:.3f}, "
                  f"val_loss={val_loss:.3f}, "
                  f"perplexity={val_perplexity:.2f}")

            # Save with metrics
            manager.step(
                model=model,
                optimizer=optimizer,
                step=step,
                loss=train_loss,
                metrics={
                    'val_loss': val_loss,
                    'val_perplexity': val_perplexity
                }
            )
        else:
            # Regular step without validation
            manager.step(
                model=model,
                optimizer=optimizer,
                step=step,
                loss=train_loss
            )

    print("  Training complete!")
    print()


# ============================================================================
# Example 6: Checkpoint with Custom Metadata
# ============================================================================

def example_custom_metadata():
    """Save checkpoints with custom metadata"""
    print("Example 6: Custom Metadata")
    print("-" * 50)

    model = SimpleModel()
    optimizer = SimpleOptimizer()

    # Save checkpoint with lots of metadata
    save_checkpoint(
        filepath='checkpoints/custom_metadata.npz',
        model=model,
        optimizer=optimizer,
        epoch=10,
        step=5000,
        loss=2.345,
        metrics={
            'train_accuracy': 0.95,
            'val_accuracy': 0.92,
            'train_perplexity': 15.3,
            'val_perplexity': 18.7
        },
        # Custom fields
        learning_rate=0.0001,
        batch_size=32,
        model_architecture='transformer',
        num_parameters=125000000,
        dataset='wikitext-103',
        notes='Best model so far!'
    )

    print("  Checkpoint saved with custom metadata")

    # Load and inspect
    from yflow.checkpoint import get_checkpoint_info

    info = get_checkpoint_info('checkpoints/custom_metadata.npz')

    print("\n  Checkpoint metadata:")
    for key, value in info.items():
        print(f"    {key}: {value}")

    print()


# ============================================================================
# Run All Examples
# ============================================================================

def run_all_examples():
    """Run all checkpointing examples"""
    print("=" * 70)
    print("YFLOW CHECKPOINT USAGE EXAMPLES")
    print("=" * 70)
    print()

    example_basic_checkpoint()
    example_checkpoint_manager()
    example_resume_training()
    example_best_model_tracking()
    example_language_model_training()
    example_custom_metadata()

    print("=" * 70)
    print("All examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_examples()