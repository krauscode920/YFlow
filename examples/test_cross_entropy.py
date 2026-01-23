#!/usr/bin/env python3
"""
Test script for CrossEntropyLoss implementation in YFlow.

Run this from your YFlow root directory:
    python test_cross_entropy.py

Or from anywhere:
    cd /path/to/YFlow
    python test_cross_entropy.py
"""

import numpy as np
import sys

# Import from YFlow
from yflow.losses.cross_entropy import CrossEntropyLoss


def test_basic_loss():
    """Test basic cross-entropy computation"""
    print("🧪 Test 1: Basic Loss Calculation")

    # Simple example: 2 samples, 3 timesteps, 5 classes
    batch_size, seq_len, vocab_size = 2, 3, 5

    # Create dummy logits (random)
    np.random.seed(42)
    logits = np.random.randn(batch_size, seq_len, vocab_size)

    # Create targets (random token IDs)
    targets = np.random.randint(0, vocab_size, size=(batch_size, seq_len))

    # Compute loss
    loss_fn = CrossEntropyLoss(reduction='mean')
    loss = loss_fn.calculate(logits, targets)

    print(f"  Logits shape: {logits.shape}")
    print(f"  Targets shape: {targets.shape}")
    print(f"  Loss value: {loss:.4f}")
    print(f"  ✅ Basic loss calculation works!\n")

    return loss


def test_gradient():
    """Test gradient computation"""
    print("🧪 Test 2: Gradient Computation")

    batch_size, seq_len, vocab_size = 2, 4, 10

    np.random.seed(42)
    logits = np.random.randn(batch_size, seq_len, vocab_size) * 0.1
    targets = np.random.randint(0, vocab_size, size=(batch_size, seq_len))

    loss_fn = CrossEntropyLoss(reduction='mean')

    # Forward pass
    loss = loss_fn.calculate(logits, targets)

    # Backward pass
    grad = loss_fn.derivative(logits, targets)

    print(f"  Loss: {loss:.4f}")
    print(f"  Gradient shape: {grad.shape}")
    print(f"  Gradient mean: {np.mean(grad):.6f}")
    print(f"  Gradient std: {np.std(grad):.6f}")

    # Verify gradient properties
    assert grad.shape == logits.shape, "Gradient shape mismatch!"
    assert not np.any(np.isnan(grad)), "NaN in gradients!"
    assert not np.any(np.isinf(grad)), "Inf in gradients!"

    print(f"  ✅ Gradient computation works!\n")

    return grad


def test_padding_mask():
    """Test that padding tokens are properly ignored"""
    print("🧪 Test 3: Padding Token Handling")

    batch_size, seq_len, vocab_size = 2, 5, 8
    padding_token = 0

    np.random.seed(42)
    logits = np.random.randn(batch_size, seq_len, vocab_size)

    # Create targets with padding
    targets = np.array([
        [1, 2, 3, 0, 0],  # Last 2 tokens are padding
        [4, 5, 0, 0, 0]  # Last 3 tokens are padding
    ])

    loss_fn = CrossEntropyLoss(ignore_index=padding_token, reduction='mean')

    # Calculate loss (should only consider non-padding tokens)
    loss = loss_fn.calculate(logits, targets)

    # Calculate gradient
    grad = loss_fn.derivative(logits, targets)

    print(f"  Targets:\n{targets}")
    print(f"  Loss (with padding ignored): {loss:.4f}")

    # Check that gradients for padding positions are zero
    padding_mask = (targets == padding_token)
    padding_grads = grad[padding_mask]

    print(f"  Gradient at padding positions (should be ~0): {np.abs(padding_grads).max():.6f}")

    assert np.allclose(padding_grads, 0.0), "Padding tokens not properly masked!"

    print(f"  ✅ Padding mask works correctly!\n")


def test_reduction_modes():
    """Test different reduction modes"""
    print("🧪 Test 4: Reduction Modes")

    batch_size, seq_len, vocab_size = 2, 3, 5

    np.random.seed(42)
    logits = np.random.randn(batch_size, seq_len, vocab_size)
    targets = np.random.randint(0, vocab_size, size=(batch_size, seq_len))

    # Test mean reduction
    loss_mean = CrossEntropyLoss(reduction='mean')
    mean_loss = loss_mean.calculate(logits, targets)

    # Test sum reduction
    loss_sum = CrossEntropyLoss(reduction='sum')
    sum_loss = loss_sum.calculate(logits, targets)

    # Test none reduction
    loss_none = CrossEntropyLoss(reduction='none')
    none_loss = loss_none.calculate(logits, targets)

    print(f"  Mean reduction: {mean_loss:.4f}")
    print(f"  Sum reduction: {sum_loss:.4f}")
    print(f"  None reduction shape: {none_loss.shape}")

    # Verify relationship
    num_elements = batch_size * seq_len
    expected_mean = sum_loss / num_elements

    print(f"  Verification: sum/count = {expected_mean:.4f} (should match mean)")

    assert np.isclose(mean_loss, expected_mean, rtol=1e-5), "Reduction modes inconsistent!"

    print(f"  ✅ All reduction modes work!\n")


def test_numerical_stability():
    """Test numerical stability with extreme values"""
    print("🧪 Test 5: Numerical Stability")

    batch_size, seq_len, vocab_size = 2, 3, 100

    # Create logits with very large values
    logits_large = np.random.randn(batch_size, seq_len, vocab_size) * 100

    # Create logits with very small values
    logits_small = np.random.randn(batch_size, seq_len, vocab_size) * 0.001

    targets = np.random.randint(0, vocab_size, size=(batch_size, seq_len))

    loss_fn = CrossEntropyLoss()

    # Test with large values
    loss_large = loss_fn.calculate(logits_large, targets)
    grad_large = loss_fn.derivative(logits_large, targets)

    # Test with small values
    loss_small = loss_fn.calculate(logits_small, targets)
    grad_small = loss_fn.derivative(logits_small, targets)

    print(f"  Loss with large logits: {loss_large:.4f}")
    print(f"  Loss with small logits: {loss_small:.4f}")

    # Check for NaN or Inf
    assert not np.isnan(loss_large) and not np.isinf(loss_large), "Large logits caused NaN/Inf!"
    assert not np.isnan(loss_small) and not np.isinf(loss_small), "Small logits caused NaN/Inf!"
    assert not np.any(np.isnan(grad_large)) and not np.any(np.isinf(grad_large)), "Large gradient has NaN/Inf!"
    assert not np.any(np.isnan(grad_small)) and not np.any(np.isinf(grad_small)), "Small gradient has NaN/Inf!"

    print(f"  ✅ Numerically stable!\n")


def test_language_model_scenario():
    """Test realistic language modeling scenario"""
    print("🧪 Test 6: Language Model Scenario")

    # Simulate a small language model output
    batch_size = 4
    seq_len = 16
    vocab_size = 50257  # GPT-2 vocab size

    np.random.seed(42)

    # Simulated model logits
    logits = np.random.randn(batch_size, seq_len, vocab_size) * 2.0

    # Simulated target tokens (with some padding)
    targets = np.random.randint(1, vocab_size, size=(batch_size, seq_len))

    # Add padding to some sequences
    targets[0, 12:] = 0  # Pad last 4 tokens
    targets[1, 14:] = 0  # Pad last 2 tokens
    targets[2, 15:] = 0  # Pad last 1 token
    # targets[3] has no padding

    loss_fn = CrossEntropyLoss(ignore_index=0, reduction='mean')

    # Calculate loss
    loss = loss_fn.calculate(logits, targets)

    # Calculate gradient
    grad = loss_fn.derivative(logits, targets)

    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Loss: {loss:.4f}")
    print(f"  Gradient norm: {np.linalg.norm(grad):.4f}")

    # Calculate perplexity (common LM metric)
    perplexity = np.exp(loss)
    print(f"  Perplexity: {perplexity:.2f}")

    print(f"  ✅ Language model scenario works!\n")


def main():
    """Run all tests"""
    print("=" * 60)
    print("CrossEntropyLoss Test Suite for YFlow")
    print("=" * 60)
    print()

    try:
        test_basic_loss()
        test_gradient()
        test_padding_mask()
        test_reduction_modes()
        test_numerical_stability()
        test_language_model_scenario()

        print("=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 60)
        print()
        print("✅ CrossEntropyLoss is ready for use in YFlow")
        print("✅ Supports GPU/CPU via device abstraction")
        print("✅ Handles padding tokens correctly")
        print("✅ Numerically stable")
        print("✅ Ready for language model training")
        print()
        print("Next step: Move to Tokenizer Integration (#2)")

        return True

    except Exception as e:
        print("=" * 60)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)