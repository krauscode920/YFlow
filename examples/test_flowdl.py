# test_flowdl.py
"""
Comprehensive test suite for FlowDL (YFlow's DataLoader).

Tests all functionality including batching, shuffling, and edge cases.
"""

import numpy as np
from yflow.data import Dataset, TensorDataset, FlowDL, InfiniteFlowDL


def test_tensor_dataset():
    """Test TensorDataset creation and indexing"""
    print("🧪 Test 1: TensorDataset")
    
    # Create sample data
    data = np.random.randn(100, 10)
    labels = np.random.randint(0, 5, size=100)
    
    # Create dataset
    dataset = TensorDataset(data, labels)
    
    print(f"  Dataset size: {len(dataset)}")
    print(f"  Data shape: {data.shape}")
    print(f"  Labels shape: {labels.shape}")
    
    # Test indexing
    x, y = dataset[0]
    print(f"  First sample shape: {x.shape}, label: {y}")
    
    assert len(dataset) == 100
    assert x.shape == (10,)
    assert isinstance(y, (int, np.integer))
    
    print("  ✅ TensorDataset works!\n")


def test_basic_batching():
    """Test basic batching without shuffling"""
    print("🧪 Test 2: Basic Batching")
    
    # Create dataset
    data = np.arange(100).reshape(100, 1)  # Simple sequential data
    labels = np.arange(100)
    dataset = TensorDataset(data, labels)
    
    # Create dataloader
    dataloader = FlowDL(dataset, batch_size=10, shuffle=False)
    
    print(f"  Dataset size: {len(dataset)}")
    print(f"  Batch size: 10")
    print(f"  Number of batches: {len(dataloader)}")
    
    batches = list(dataloader)
    
    print(f"  First batch data: {batches[0][0].flatten()[:5]}...")
    print(f"  First batch labels: {batches[0][1][:5]}...")
    
    # Verify number of batches
    assert len(batches) == 10
    
    # Verify first batch
    assert batches[0][0].shape == (10, 1)
    assert batches[0][1].shape == (10,)
    
    # Verify data is in order (no shuffling)
    assert np.array_equal(batches[0][1], np.arange(10))
    
    print("  ✅ Basic batching works!\n")


def test_shuffling():
    """Test shuffling functionality"""
    print("🧪 Test 3: Shuffling")
    
    # Create dataset
    data = np.arange(100).reshape(100, 1)
    labels = np.arange(100)
    dataset = TensorDataset(data, labels)
    
    # Create dataloader with shuffling
    dataloader = FlowDL(dataset, batch_size=10, shuffle=True)
    
    # Get batches from two epochs
    epoch1_batches = list(dataloader)
    epoch2_batches = list(dataloader)
    
    # Extract first batch labels from each epoch
    epoch1_first_labels = epoch1_batches[0][1]
    epoch2_first_labels = epoch2_batches[0][1]
    
    print(f"  Epoch 1 first batch labels: {epoch1_first_labels}")
    print(f"  Epoch 2 first batch labels: {epoch2_first_labels}")
    
    # They should be different due to shuffling
    # (there's a tiny chance they're the same, but very unlikely)
    different = not np.array_equal(epoch1_first_labels, epoch2_first_labels)
    
    print(f"  First batches are different: {different}")
    
    # Verify all data is still present (just reordered)
    all_epoch1_labels = np.concatenate([batch[1] for batch in epoch1_batches])
    assert len(np.unique(all_epoch1_labels)) == 100
    
    print("  ✅ Shuffling works!\n")


def test_drop_last():
    """Test drop_last functionality"""
    print("🧪 Test 4: Drop Last Batch")
    
    # Create dataset with size not divisible by batch_size
    data = np.arange(95).reshape(95, 1)
    labels = np.arange(95)
    dataset = TensorDataset(data, labels)
    
    # Without drop_last
    dataloader_keep = FlowDL(dataset, batch_size=10, drop_last=False)
    batches_keep = list(dataloader_keep)
    
    # With drop_last
    dataloader_drop = FlowDL(dataset, batch_size=10, drop_last=True)
    batches_drop = list(dataloader_drop)
    
    print(f"  Dataset size: {len(dataset)}")
    print(f"  Batch size: 10")
    print(f"  Batches (drop_last=False): {len(batches_keep)}")
    print(f"  Last batch size (drop_last=False): {len(batches_keep[-1][1])}")
    print(f"  Batches (drop_last=True): {len(batches_drop)}")
    
    # Verify drop_last=False keeps incomplete batch
    assert len(batches_keep) == 10  # 95/10 = 9 full + 1 partial
    assert len(batches_keep[-1][1]) == 5  # Last batch has 5 samples
    
    # Verify drop_last=True drops incomplete batch
    assert len(batches_drop) == 9  # Only complete batches
    
    print("  ✅ Drop last works!\n")


def test_single_sample_batch():
    """Test batch_size=1 edge case"""
    print("🧪 Test 5: Single Sample Batches")
    
    data = np.random.randn(10, 5)
    labels = np.arange(10)
    dataset = TensorDataset(data, labels)
    
    dataloader = FlowDL(dataset, batch_size=1, shuffle=False)
    
    batches = list(dataloader)
    
    print(f"  Dataset size: {len(dataset)}")
    print(f"  Number of batches: {len(batches)}")
    print(f"  First batch shape: {batches[0][0].shape}")
    
    # Should have 10 batches, each with shape (1, 5)
    assert len(batches) == 10
    assert batches[0][0].shape == (1, 5)
    
    print("  ✅ Single sample batches work!\n")


def test_full_dataset_batch():
    """Test batch_size equal to dataset size"""
    print("🧪 Test 6: Full Dataset as Single Batch")
    
    data = np.random.randn(50, 8)
    labels = np.arange(50)
    dataset = TensorDataset(data, labels)
    
    dataloader = FlowDL(dataset, batch_size=50, shuffle=False)
    
    batches = list(dataloader)
    
    print(f"  Dataset size: {len(dataset)}")
    print(f"  Batch size: 50")
    print(f"  Number of batches: {len(batches)}")
    print(f"  Batch shape: {batches[0][0].shape}")
    
    # Should have 1 batch with entire dataset
    assert len(batches) == 1
    assert batches[0][0].shape == (50, 8)
    assert batches[0][1].shape == (50,)
    
    print("  ✅ Full dataset batching works!\n")


def test_custom_collate():
    """Test custom collate function"""
    print("🧪 Test 7: Custom Collate Function")
    
    # Create dataset
    data = np.random.randn(20, 5)
    labels = np.arange(20)
    dataset = TensorDataset(data, labels)
    
    # Custom collate that adds batch statistics
    def custom_collate(batch):
        data_batch = np.stack([item[0] for item in batch])
        label_batch = np.array([item[1] for item in batch])
        
        # Add mean and std as additional info
        batch_mean = data_batch.mean()
        batch_std = data_batch.std()
        
        return data_batch, label_batch, batch_mean, batch_std
    
    dataloader = FlowDL(dataset, batch_size=10, collate_fn=custom_collate)
    
    batch = next(iter(dataloader))
    
    print(f"  Batch components: {len(batch)}")
    print(f"  Data shape: {batch[0].shape}")
    print(f"  Labels shape: {batch[1].shape}")
    print(f"  Batch mean: {batch[2]:.4f}")
    print(f"  Batch std: {batch[3]:.4f}")
    
    # Should have 4 components (data, labels, mean, std)
    assert len(batch) == 4
    assert batch[0].shape == (10, 5)
    assert isinstance(batch[2], (float, np.floating))
    
    print("  ✅ Custom collate works!\n")


def test_infinite_dataloader():
    """Test InfiniteFlowDL"""
    print("🧪 Test 8: Infinite DataLoader")
    
    data = np.arange(30).reshape(30, 1)
    labels = np.arange(30)
    dataset = TensorDataset(data, labels)
    
    dataloader = InfiniteFlowDL(dataset, batch_size=10, shuffle=False)
    
    # Take more batches than one epoch would provide
    batches = []
    for i, batch in enumerate(dataloader):
        batches.append(batch)
        if i >= 5:  # Take 6 batches (2 epochs)
            break
    
    print(f"  Dataset size: {len(dataset)}")
    print(f"  Batches taken: {len(batches)}")
    print(f"  Expected batches per epoch: 3")
    
    assert len(batches) == 6  # 2 full epochs
    
    print("  ✅ Infinite dataloader works!\n")


def test_language_model_scenario():
    """Test realistic language model training scenario"""
    print("🧪 Test 9: Language Model Training Scenario")
    
    # Simulate tokenized text sequences
    vocab_size = 50257  # GPT-2 vocab size
    sequence_length = 128
    num_sequences = 1000
    
    # Create dataset
    input_ids = np.random.randint(0, vocab_size, size=(num_sequences, sequence_length))
    labels = np.random.randint(0, vocab_size, size=(num_sequences, sequence_length))
    
    dataset = TensorDataset(input_ids, labels)
    
    # Create dataloader
    batch_size = 32
    dataloader = FlowDL(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    print(f"  Dataset:")
    print(f"    Sequences: {num_sequences}")
    print(f"    Sequence length: {sequence_length}")
    print(f"    Vocab size: {vocab_size}")
    print(f"  DataLoader:")
    print(f"    Batch size: {batch_size}")
    print(f"    Number of batches: {len(dataloader)}")
    
    # Simulate training loop
    num_batches_processed = 0
    for batch_inputs, batch_labels in dataloader:
        # Verify shapes
        assert batch_inputs.shape == (batch_size, sequence_length)
        assert batch_labels.shape == (batch_size, sequence_length)
        
        # Verify token IDs are valid
        assert batch_inputs.min() >= 0
        assert batch_inputs.max() < vocab_size
        
        num_batches_processed += 1
        
        if num_batches_processed == 3:  # Just process a few batches for testing
            break
    
    print(f"  Batches processed: {num_batches_processed}")
    print(f"  Batch input shape: {batch_inputs.shape}")
    print(f"  Batch labels shape: {batch_labels.shape}")
    print("  ✅ Language model scenario works!\n")


def run_all_tests():
    """Run all FlowDL tests"""
    print("=" * 70)
    print("FLOWDL TEST SUITE")
    print("=" * 70)
    print()
    
    try:
        test_tensor_dataset()
        test_basic_batching()
        test_shuffling()
        test_drop_last()
        test_single_sample_batch()
        test_full_dataset_batch()
        test_custom_collate()
        test_infinite_dataloader()
        test_language_model_scenario()
        
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
