# YFormers Component Testing Suite
# Step 1: Component-level testing with tiny inputs

import numpy as np
import sys
import traceback

# Test configuration
BATCH_SIZE = 1
SEQ_LEN = 4
EMBED_DIM = 8
NUM_HEADS = 2
FF_DIM = 16
VOCAB_SIZE = 10


def print_separator(title):
    """Print a nice separator for test sections"""
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print('=' * 60)


def print_tensor_info(name, tensor, expected_shape=None):
    """Print detailed tensor information"""
    if tensor is None:
        print(f"{name}: None")
        return

    print(f"{name}:")
    print(f"  Shape: {tensor.shape}")
    print(f"  Dtype: {tensor.dtype}")
    print(f"  Device: {'GPU' if hasattr(tensor, 'device') else 'CPU'}")
    if expected_shape:
        shape_match = tensor.shape == expected_shape
        print(f"  Expected: {expected_shape} - {'✓' if shape_match else '✗'}")

    # Show some values for small tensors
    if tensor.size <= 32:
        print(f"  Values: {tensor.flatten()[:8]}...")
    print()


def test_device_setup():
    """Test device setup and basic operations"""
    print_separator("DEVICE SETUP TEST")

    try:
        from yflow.core.device import Device

        # Test CPU device
        cpu_device = Device('cpu')
        print(f"CPU Device: {cpu_device}")

        # Test basic array operations
        test_array = cpu_device.xp.array([1, 2, 3, 4])
        print_tensor_info("CPU test array", test_array)

        # Test GPU device (if available)
        try:
            gpu_device = Device('gpu')
            print(f"GPU Device: {gpu_device}")

            # Test GPU operations
            gpu_array = gpu_device.to_device(test_array)
            print_tensor_info("GPU test array", gpu_array)

        except Exception as e:
            print(f"GPU not available or failed: {e}")

        return True

    except Exception as e:
        print(f"Device setup failed: {e}")
        traceback.print_exc()
        return False


def test_token_embedding():
    """Test TokenEmbedding layer"""
    print_separator("TOKEN EMBEDDING TEST")

    try:
        from yflow.yformers.embeddings import TokenEmbedding

        # Create embedding layer
        embedding = TokenEmbedding(
            vocab_size=VOCAB_SIZE,
            embed_dim=EMBED_DIM,
            padding_idx=0,
            scale_by_dim=True
        )

        print(f"TokenEmbedding created:")
        print(f"  Vocab size: {embedding.vocab_size}")
        print(f"  Embed dim: {embedding.embed_dim}")
        print(f"  Scale by dim: {embedding.scale_by_dim}")

        # Test with simple input
        tokens = np.array([[1, 2, 3, 0]])  # batch=1, seq=4, with padding
        print_tensor_info("Input tokens", tokens, (BATCH_SIZE, SEQ_LEN))

        # Forward pass
        embedded = embedding.forward(tokens)
        print_tensor_info("Embedded output", embedded, (BATCH_SIZE, SEQ_LEN, EMBED_DIM))

        # Test backward pass
        grad = np.ones_like(embedded)
        input_grad = embedding.backward(grad)
        print(f"Backward pass completed - input grad: {input_grad}")

        # Check trainable parameters
        params = embedding.get_trainable_params()
        print(f"Trainable parameters: {list(params.keys())}")
        for name, param in params.items():
            print_tensor_info(f"  {name}", param)

        return True

    except Exception as e:
        print(f"TokenEmbedding test failed: {e}")
        traceback.print_exc()
        return False


def test_positional_encoding():
    """Test PositionalEncoding layer"""
    print_separator("POSITIONAL ENCODING TEST")

    try:
        from yflow.yformers.embeddings import PositionalEncoding

        # Create positional encoding
        pos_encoding = PositionalEncoding(
            embed_dim=EMBED_DIM,
            max_seq_len=10,
            dropout=0.0  # No dropout for testing
        )

        print(f"PositionalEncoding created:")
        print(f"  Embed dim: {pos_encoding.embed_dim}")
        print(f"  Max seq len: {pos_encoding.max_seq_len}")

        # Create dummy embeddings
        embeddings = np.random.randn(BATCH_SIZE, SEQ_LEN, EMBED_DIM)
        print_tensor_info("Input embeddings", embeddings, (BATCH_SIZE, SEQ_LEN, EMBED_DIM))

        # Forward pass
        pos_encoded = pos_encoding.forward(embeddings)
        print_tensor_info("Position encoded", pos_encoded, (BATCH_SIZE, SEQ_LEN, EMBED_DIM))

        # Test backward pass
        grad = np.ones_like(pos_encoded)
        input_grad = pos_encoding.backward(grad)
        print_tensor_info("Input gradient", input_grad, (BATCH_SIZE, SEQ_LEN, EMBED_DIM))

        return True

    except Exception as e:
        print(f"PositionalEncoding test failed: {e}")
        traceback.print_exc()
        return False


def test_self_attention():
    """Test SelfAttention layer"""
    print_separator("SELF ATTENTION TEST")

    try:
        from yflow.yformers.attention import SelfAttention

        # Create self-attention layer
        self_attn = SelfAttention(
            embed_dim=EMBED_DIM,
            dropout=0.0  # No dropout for testing
        )

        print(f"SelfAttention created:")
        print(f"  Embed dim: {self_attn.embed_dim}")
        print(f"  Scale: {self_attn.scale}")

        # Create dummy input
        x = np.random.randn(BATCH_SIZE, SEQ_LEN, EMBED_DIM) * 0.1  # Small values
        print_tensor_info("Input", x, (BATCH_SIZE, SEQ_LEN, EMBED_DIM))

        # Forward pass
        attn_output = self_attn.forward(x)
        print_tensor_info("Attention output", attn_output, (BATCH_SIZE, SEQ_LEN, EMBED_DIM))

        # Test backward pass
        grad = np.ones_like(attn_output)
        input_grad = self_attn.backward(grad)
        print_tensor_info("Input gradient", input_grad, (BATCH_SIZE, SEQ_LEN, EMBED_DIM))

        # Check trainable parameters
        params = self_attn.get_trainable_params()
        print(f"Trainable parameters: {list(params.keys())}")

        return True

    except Exception as e:
        print(f"SelfAttention test failed: {e}")
        traceback.print_exc()
        return False


def test_multi_head_attention():
    """Test MultiHeadAttention layer"""
    print_separator("MULTI-HEAD ATTENTION TEST")

    try:
        from yflow.yformers.attention import MultiHeadAttention

        # Create multi-head attention
        mha = MultiHeadAttention(
            embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS,
            dropout=0.0
        )

        print(f"MultiHeadAttention created:")
        print(f"  Embed dim: {mha.embed_dim}")
        print(f"  Num heads: {mha.num_heads}")
        print(f"  Head dim: {mha.head_dim}")

        # Create dummy input
        q = np.random.randn(BATCH_SIZE, SEQ_LEN, EMBED_DIM) * 0.1
        print_tensor_info("Query input", q, (BATCH_SIZE, SEQ_LEN, EMBED_DIM))

        # Test self-attention (k=v=q)
        print("\nTesting self-attention mode:")
        attn_output = mha.forward(q)
        print_tensor_info("Self-attention output", attn_output, (BATCH_SIZE, SEQ_LEN, EMBED_DIM))

        # Test cross-attention
        print("\nTesting cross-attention mode:")
        k = np.random.randn(BATCH_SIZE, SEQ_LEN, EMBED_DIM) * 0.1
        v = np.random.randn(BATCH_SIZE, SEQ_LEN, EMBED_DIM) * 0.1
        cross_output = mha.forward(q, k, v)
        print_tensor_info("Cross-attention output", cross_output, (BATCH_SIZE, SEQ_LEN, EMBED_DIM))

        # Test backward pass
        grad = np.ones_like(attn_output)
        input_grad = mha.backward(grad)

        # Handle tuple return for cross-attention or single tensor for self-attention
        if isinstance(input_grad, tuple):
            print(f"Cross-attention gradients returned (tuple of {len(input_grad)} tensors)")
            for i, grad_tensor in enumerate(input_grad):
                print_tensor_info(f"Gradient {i}", grad_tensor, (BATCH_SIZE, SEQ_LEN, EMBED_DIM))
        else:
            print_tensor_info("Input gradient", input_grad, (BATCH_SIZE, SEQ_LEN, EMBED_DIM))

        return True

    except Exception as e:
        print(f"MultiHeadAttention test failed: {e}")
        traceback.print_exc()
        return False


def test_feed_forward():
    """Test FeedForward layer from encoder module"""
    print_separator("FEED FORWARD TEST")

    try:
        from yflow.yformers.encoder import FeedForward

        # Create feed-forward network
        ff = FeedForward(
            embed_dim=EMBED_DIM,
            ff_dim=FF_DIM,
            dropout=0.0,
            activation='relu'
        )

        print(f"FeedForward created:")
        print(f"  Embed dim: {ff.embed_dim}")
        print(f"  FF dim: {ff.ff_dim}")

        # Create dummy input
        x = np.random.randn(BATCH_SIZE, SEQ_LEN, EMBED_DIM) * 0.1
        print_tensor_info("Input", x, (BATCH_SIZE, SEQ_LEN, EMBED_DIM))

        # Forward pass
        ff_output = ff.forward(x)
        print_tensor_info("FF output", ff_output, (BATCH_SIZE, SEQ_LEN, EMBED_DIM))

        # Test backward pass
        grad = np.ones_like(ff_output)
        input_grad = ff.backward(grad)
        print_tensor_info("Input gradient", input_grad, (BATCH_SIZE, SEQ_LEN, EMBED_DIM))

        return True

    except Exception as e:
        print(f"FeedForward test failed: {e}")
        traceback.print_exc()
        return False


def test_encoder_block():
    """Test EncoderBlock"""
    print_separator("ENCODER BLOCK TEST")

    try:
        from yflow.yformers.encoder import EncoderBlock

        # Create encoder block
        encoder_block = EncoderBlock(
            embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS,
            ff_dim=FF_DIM,
            dropout=0.0,
            pre_norm=True,
            activation='relu'
        )

        print(f"EncoderBlock created:")
        print(f"  Embed dim: {encoder_block.embed_dim}")
        print(f"  Num heads: {encoder_block.num_heads}")
        print(f"  FF dim: {encoder_block.ff_dim}")
        print(f"  Pre-norm: {encoder_block.pre_norm}")

        # Create dummy input
        x = np.random.randn(BATCH_SIZE, SEQ_LEN, EMBED_DIM) * 0.1
        print_tensor_info("Input", x, (BATCH_SIZE, SEQ_LEN, EMBED_DIM))

        # Forward pass
        block_output = encoder_block.forward(x)
        print_tensor_info("Block output", block_output, (BATCH_SIZE, SEQ_LEN, EMBED_DIM))

        # Test backward pass
        grad = np.ones_like(block_output)
        input_grad = encoder_block.backward(grad)

        # Handle potential shape issues
        if hasattr(input_grad, 'shape'):
            if input_grad.ndim > 3:
                # If there's an extra dimension, take the first element
                input_grad = input_grad[0] if input_grad.shape[0] == 1 else input_grad
            print_tensor_info("Input gradient", input_grad, (BATCH_SIZE, SEQ_LEN, EMBED_DIM))
        else:
            print(f"Input gradient: {type(input_grad)} - {input_grad}")

        return True

    except Exception as e:
        print(f"EncoderBlock test failed: {e}")
        traceback.print_exc()
        return False


def test_decoder_block():
    """Test DecoderBlock"""
    print_separator("DECODER BLOCK TEST")

    try:
        from yflow.yformers.decoder import DecoderBlock

        # Create decoder block
        decoder_block = DecoderBlock(
            embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS,
            ff_dim=FF_DIM,
            dropout=0.0,
            pre_norm=True,
            activation='relu'
        )

        print(f"DecoderBlock created:")
        print(f"  Embed dim: {decoder_block.embed_dim}")
        print(f"  Num heads: {decoder_block.num_heads}")
        print(f"  FF dim: {decoder_block.ff_dim}")

        # Create dummy inputs
        x = np.random.randn(BATCH_SIZE, SEQ_LEN, EMBED_DIM) * 0.1
        encoder_output = np.random.randn(BATCH_SIZE, SEQ_LEN, EMBED_DIM) * 0.1

        print_tensor_info("Decoder input", x, (BATCH_SIZE, SEQ_LEN, EMBED_DIM))
        print_tensor_info("Encoder output", encoder_output, (BATCH_SIZE, SEQ_LEN, EMBED_DIM))

        # Forward pass
        block_output = decoder_block.forward(x, encoder_output)
        print_tensor_info("Block output", block_output, (BATCH_SIZE, SEQ_LEN, EMBED_DIM))

        # Test backward pass
        grad = np.ones_like(block_output)
        try:
            result = decoder_block.backward(grad)

            # Handle different return formats
            if isinstance(result, tuple):
                if len(result) == 2:
                    input_grad, enc_grad = result
                    print_tensor_info("Input gradient", input_grad, (BATCH_SIZE, SEQ_LEN, EMBED_DIM))
                    print_tensor_info("Encoder gradient", enc_grad, (BATCH_SIZE, SEQ_LEN, EMBED_DIM))
                else:
                    print(f"Backward returned {len(result)} values: {[type(r) for r in result]}")
                    # Take first two if more than 2
                    input_grad, enc_grad = result[0], result[1]
                    print_tensor_info("Input gradient", input_grad, (BATCH_SIZE, SEQ_LEN, EMBED_DIM))
                    print_tensor_info("Encoder gradient", enc_grad, (BATCH_SIZE, SEQ_LEN, EMBED_DIM))
            else:
                print_tensor_info("Single gradient", result, (BATCH_SIZE, SEQ_LEN, EMBED_DIM))
        except Exception as e:
            print(f"Backward pass failed: {e}")
            return False

        return True

    except Exception as e:
        print(f"DecoderBlock test failed: {e}")
        traceback.print_exc()
        return False


def test_simple_integration():
    """Test that components work together in a simple sequence"""
    print_separator("SIMPLE INTEGRATION TEST")

    try:
        from yflow.yformers.embeddings import TokenEmbedding, PositionalEncoding
        from yflow.yformers.attention import MultiHeadAttention
        from yflow.yformers.encoder import FeedForward

        # Create a simple pipeline
        embedding = TokenEmbedding(vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM)
        pos_encoding = PositionalEncoding(embed_dim=EMBED_DIM, dropout=0.0)
        attention = MultiHeadAttention(embed_dim=EMBED_DIM, num_heads=NUM_HEADS, dropout=0.0)
        feed_forward = FeedForward(embed_dim=EMBED_DIM, ff_dim=FF_DIM, dropout=0.0)

        # Test tokens
        tokens = np.array([[1, 2, 3, 0]])
        print_tensor_info("Input tokens", tokens)

        # Pipeline: tokens -> embeddings -> positional -> attention -> feedforward
        embedded = embedding.forward(tokens)
        print_tensor_info("After embedding", embedded)

        pos_encoded = pos_encoding.forward(embedded)
        print_tensor_info("After positional encoding", pos_encoded)

        attended = attention.forward(pos_encoded)
        print_tensor_info("After attention", attended)

        final_output = feed_forward.forward(attended)
        print_tensor_info("Final output", final_output)

        print("✓ Integration test completed successfully!")
        return True

    except Exception as e:
        print(f"Integration test failed: {e}")
        traceback.print_exc()
        return False


def run_all_component_tests():
    """Run all component tests in sequence"""
    print("Starting YFormers Component Testing Suite")
    print(f"Test configuration: batch={BATCH_SIZE}, seq={SEQ_LEN}, embed={EMBED_DIM}")

    tests = [
        ("Device Setup", test_device_setup),
        ("Token Embedding", test_token_embedding),
        ("Positional Encoding", test_positional_encoding),
        ("Self Attention", test_self_attention),
        ("Multi-Head Attention", test_multi_head_attention),
        ("Feed Forward", test_feed_forward),
        ("Encoder Block", test_encoder_block),
        ("Decoder Block", test_decoder_block),
        ("Simple Integration", test_simple_integration),  # New integration test
    ]

    results = {}

    for test_name, test_func in tests:
        print(f"\n\nRunning {test_name} test...")
        try:
            success = test_func()
            results[test_name] = "✓ PASSED" if success else "✗ FAILED"
        except Exception as e:
            print(f"Test {test_name} crashed: {e}")
            results[test_name] = "💥 CRASHED"

    # Print summary
    print_separator("TEST RESULTS SUMMARY")
    for test_name, result in results.items():
        print(f"{test_name:25}: {result}")

    # Overall results
    passed = sum(1 for r in results.values() if "PASSED" in r)
    total = len(results)
    print(f"\nOverall: {passed}/{total} tests passed")

    return results


if __name__ == "__main__":
    # Add the yflow directory to Python path
    # Adjust this path as needed based on your setup
    import os

    yflow_path = os.path.join(os.path.dirname(__file__), '..')
    if yflow_path not in sys.path:
        sys.path.insert(0, yflow_path)

    results = run_all_component_tests()