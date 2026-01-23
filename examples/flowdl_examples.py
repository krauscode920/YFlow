# FlowDL Usage Examples

"""
Comprehensive examples showing how to use FlowDL in different scenarios.
"""

import numpy as np
from yflow.data import Dataset, TensorDataset, FlowDL, InfiniteFlowDL


# ============================================================================
# Example 1: Basic Usage
# ============================================================================

def example_basic():
    """Basic FlowDL usage"""
    print("Example 1: Basic Usage")
    print("-" * 50)
    
    # Create simple dataset
    data = np.random.randn(100, 10)  # 100 samples, 10 features
    labels = np.random.randint(0, 5, size=100)  # 5 classes
    
    dataset = TensorDataset(data, labels)
    dataloader = FlowDL(dataset, batch_size=32, shuffle=True)
    
    # Training loop
    for epoch in range(3):
        print(f"\nEpoch {epoch + 1}")
        for i, (batch_data, batch_labels) in enumerate(dataloader):
            # Your training code here
            # loss = model(batch_data, batch_labels)
            # ...
            
            if i == 0:  # Just show first batch
                print(f"  Batch {i}: data shape = {batch_data.shape}, "
                      f"labels shape = {batch_labels.shape}")
    
    print()


# ============================================================================
# Example 2: Custom Dataset
# ============================================================================

def example_custom_dataset():
    """Create a custom dataset class"""
    print("Example 2: Custom Dataset")
    print("-" * 50)
    
    class ImageDataset(Dataset):
        """Custom dataset for loading images"""
        
        def __init__(self, image_paths, labels):
            self.image_paths = image_paths
            self.labels = labels
        
        def __len__(self):
            return len(self.image_paths)
        
        def __getitem__(self, idx):
            # In real scenario, you'd load the image from disk here
            # For demo, we'll create fake images
            image = np.random.randn(224, 224, 3)  # Fake image
            label = self.labels[idx]
            
            # Could apply transformations here
            # image = self.transform(image)
            
            return image, label
    
    # Create dataset
    image_paths = [f"image_{i}.jpg" for i in range(50)]
    labels = np.random.randint(0, 10, size=50)
    
    dataset = ImageDataset(image_paths, labels)
    dataloader = FlowDL(dataset, batch_size=8, shuffle=True)
    
    # Use it
    for batch_images, batch_labels in dataloader:
        print(f"  Batch: images shape = {batch_images.shape}, "
              f"labels shape = {batch_labels.shape}")
        break  # Just show first batch
    
    print()


# ============================================================================
# Example 3: Language Model Training
# ============================================================================

def example_language_model():
    """Language model training with FlowDL"""
    print("Example 3: Language Model Training")
    print("-" * 50)
    
    # Simulate tokenized text data
    vocab_size = 50257  # GPT-2 vocab size
    sequence_length = 128
    num_sequences = 1000
    
    # Create dataset (input_ids and target_ids)
    input_ids = np.random.randint(0, vocab_size, size=(num_sequences, sequence_length))
    target_ids = np.random.randint(0, vocab_size, size=(num_sequences, sequence_length))
    
    dataset = TensorDataset(input_ids, target_ids)
    dataloader = FlowDL(
        dataset,
        batch_size=32,
        shuffle=True,
        drop_last=True  # Drop incomplete batches
    )
    
    print(f"  Total sequences: {num_sequences}")
    print(f"  Sequence length: {sequence_length}")
    print(f"  Batch size: 32")
    print(f"  Number of batches per epoch: {len(dataloader)}")
    
    # Training loop
    for epoch in range(2):
        print(f"\n  Epoch {epoch + 1}")
        for i, (batch_inputs, batch_targets) in enumerate(dataloader):
            # Forward pass
            # logits = model(batch_inputs)
            # loss = cross_entropy_loss(logits, batch_targets)
            # loss.backward()
            # optimizer.step()
            
            if i % 10 == 0:
                print(f"    Step {i}: batch shape = {batch_inputs.shape}")
    
    print()


# ============================================================================
# Example 4: Custom Collate Function
# ============================================================================

def example_custom_collate():
    """Using custom collate function for variable-length sequences"""
    print("Example 4: Custom Collate Function")
    print("-" * 50)
    
    # Simulate variable-length sequences
    sequences = [
        np.random.randn(10, 5),   # Length 10
        np.random.randn(15, 5),   # Length 15
        np.random.randn(8, 5),    # Length 8
        np.random.randn(12, 5),   # Length 12
        np.random.randn(20, 5),   # Length 20
        np.random.randn(11, 5),   # Length 11
    ]
    labels = np.array([0, 1, 0, 1, 0, 1])
    
    class SequenceDataset(Dataset):
        def __init__(self, sequences, labels):
            self.sequences = sequences
            self.labels = labels
        
        def __len__(self):
            return len(self.sequences)
        
        def __getitem__(self, idx):
            return self.sequences[idx], self.labels[idx]
    
    # Custom collate function to pad sequences
    def pad_collate(batch):
        sequences, labels = zip(*batch)
        
        # Find max length in this batch
        max_len = max(seq.shape[0] for seq in sequences)
        
        # Pad sequences
        padded_sequences = []
        for seq in sequences:
            pad_len = max_len - seq.shape[0]
            if pad_len > 0:
                padding = np.zeros((pad_len, seq.shape[1]))
                padded = np.vstack([seq, padding])
            else:
                padded = seq
            padded_sequences.append(padded)
        
        return np.stack(padded_sequences), np.array(labels)
    
    dataset = SequenceDataset(sequences, labels)
    dataloader = FlowDL(dataset, batch_size=3, collate_fn=pad_collate)
    
    for batch_sequences, batch_labels in dataloader:
        print(f"  Batch sequences shape: {batch_sequences.shape}")
        print(f"  Batch labels shape: {batch_labels.shape}")
    
    print()


# ============================================================================
# Example 5: Infinite DataLoader for Step-Based Training
# ============================================================================

def example_infinite():
    """Using InfiniteFlowDL for step-based training"""
    print("Example 5: Infinite DataLoader")
    print("-" * 50)
    
    # Create dataset
    data = np.random.randn(100, 20)
    labels = np.random.randint(0, 3, size=100)
    dataset = TensorDataset(data, labels)
    
    # Create infinite dataloader
    dataloader = InfiniteFlowDL(dataset, batch_size=16, shuffle=True)
    
    # Train for fixed number of steps (not epochs)
    max_steps = 50
    print(f"  Training for {max_steps} steps...")
    
    for step, (batch_data, batch_labels) in enumerate(dataloader):
        if step >= max_steps:
            break
        
        # Your training code here
        # loss = model(batch_data, batch_labels)
        # ...
        
        if step % 10 == 0:
            print(f"    Step {step}: batch shape = {batch_data.shape}")
    
    print()


# ============================================================================
# Example 6: Multiple Dataloaders (Train/Val/Test Split)
# ============================================================================

def example_train_val_test():
    """Using multiple dataloaders for train/val/test"""
    print("Example 6: Train/Val/Test Split")
    print("-" * 50)
    
    # Create full dataset
    full_data = np.random.randn(1000, 15)
    full_labels = np.random.randint(0, 4, size=1000)
    
    # Split into train/val/test (80/10/10)
    train_data, train_labels = full_data[:800], full_labels[:800]
    val_data, val_labels = full_data[800:900], full_labels[800:900]
    test_data, test_labels = full_data[900:], full_labels[900:]
    
    # Create datasets
    train_dataset = TensorDataset(train_data, train_labels)
    val_dataset = TensorDataset(val_data, val_labels)
    test_dataset = TensorDataset(test_data, test_labels)
    
    # Create dataloaders
    train_loader = FlowDL(train_dataset, batch_size=32, shuffle=True, drop_last=True)
    val_loader = FlowDL(val_dataset, batch_size=32, shuffle=False)
    test_loader = FlowDL(test_dataset, batch_size=32, shuffle=False)
    
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Training loop
    for epoch in range(2):
        # Training
        print(f"\n  Epoch {epoch + 1} - Training")
        for batch_data, batch_labels in train_loader:
            # Train
            pass
        
        # Validation
        print(f"  Epoch {epoch + 1} - Validation")
        for batch_data, batch_labels in val_loader:
            # Validate
            pass
    
    # Final test
    print(f"\n  Final Testing")
    for batch_data, batch_labels in test_loader:
        # Test
        pass
    
    print()


# ============================================================================
# Run All Examples
# ============================================================================

def run_all_examples():
    """Run all usage examples"""
    print("=" * 70)
    print("FLOWDL USAGE EXAMPLES")
    print("=" * 70)
    print()
    
    example_basic()
    example_custom_dataset()
    example_language_model()
    example_custom_collate()
    example_infinite()
    example_train_val_test()
    
    print("=" * 70)
    print("All examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_examples()
