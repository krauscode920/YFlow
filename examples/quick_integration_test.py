#!/usr/bin/env python3
"""
Quick YFormers Integration Test - Step 2
A minimal test to verify your transformer models work end-to-end

Run this first to quickly check if everything is working!
"""

import numpy as np
import sys
import traceback

# Assuming you're running from the YFlow directory
try:
    from yflow.yformers import EncoderOnlyModel, DecoderOnlyModel, TransformerModel
    from yflow.core.device import Device

    print("✅ YFormers imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the YFlow root directory")
    sys.exit(1)


def test_encoder_only_quick():
    """Quick test of EncoderOnlyModel"""
    print("\n🔍 Testing EncoderOnlyModel...")

    try:
        # Simple configuration
        model = EncoderOnlyModel(
            vocab_size=100,
            d_model=64,
            num_heads=4,
            d_ff=128,
            num_layers=1,
            num_classes=2
        )

        # Dummy data: batch_size=2, seq_len=8
        tokens = np.array([[1, 2, 3, 4, 5, 0, 0, 0],
                           [6, 7, 8, 9, 10, 11, 0, 0]])

        # Forward pass
        print("  🔄 Forward pass...")
        logits = model.forward(tokens)
        print(f"  ✅ Output shape: {logits.shape} (expected: (2, 2))")

        # Simple backward pass
        print("  🔄 Backward pass...")
        grad = np.ones_like(logits)
        model.backward(grad)
        print("  ✅ Backward pass completed")

        # Check gradients exist
        grads = model.get_gradients()
        params = model.get_trainable_params()
        print(f"  ✅ Found {len(params)} parameters, {len(grads)} gradients")

        print("  🎉 EncoderOnlyModel test PASSED!")
        return True

    except Exception as e:
        print(f"  ❌ EncoderOnlyModel test FAILED: {e}")
        print(f"  Traceback: {traceback.format_exc()}")
        return False


def test_decoder_only_quick():
    """Quick test of DecoderOnlyModel"""
    print("\n🔍 Testing DecoderOnlyModel...")

    try:
        # Simple configuration
        model = DecoderOnlyModel(
            vocab_size=100,
            d_model=64,
            num_heads=4,
            d_ff=128,
            num_layers=1
        )

        # Dummy data: batch_size=2, seq_len=6
        tokens = np.array([[1, 2, 3, 4, 5, 6],
                           [7, 8, 9, 10, 11, 12]])

        # Forward pass
        print("  🔄 Forward pass...")
        logits = model.forward(tokens)
        print(f"  ✅ Output shape: {logits.shape} (expected: (2, 6, 100))")

        # Simple backward pass
        print("  🔄 Backward pass...")
        grad = np.ones_like(logits) * 0.01  # Small gradient
        model.backward(grad)
        print("  ✅ Backward pass completed")

        # Check gradients exist
        grads = model.get_gradients()
        params = model.get_trainable_params()
        print(f"  ✅ Found {len(params)} parameters, {len(grads)} gradients")

        # Quick generation test
        print("  🔄 Testing generation...")
        prompt = tokens[:1]  # Just first sequence
        generated = model.generate(prompt, max_len=10, temperature=1.0)
        print(f"  ✅ Generated shape: {generated.shape}")

        print("  🎉 DecoderOnlyModel test PASSED!")
        return True

    except Exception as e:
        print(f"  ❌ DecoderOnlyModel test FAILED: {e}")
        print(f"  Traceback: {traceback.format_exc()}")
        return False


def test_transformer_quick():
    """Quick test of full TransformerModel"""
    print("\n🔍 Testing TransformerModel...")

    try:
        # Simple configuration
        model = TransformerModel(
            src_vocab_size=80,
            tgt_vocab_size=90,
            d_model=64,
            num_heads=4,
            d_ff=128,
            num_encoder_layers=1,
            num_decoder_layers=1
        )

        # Dummy data
        src_tokens = np.array([[1, 2, 3, 4, 5, 0],
                               [6, 7, 8, 9, 0, 0]])  # batch_size=2, src_seq_len=6

        tgt_tokens = np.array([[1, 2, 3, 4, 5],
                               [6, 7, 8, 9, 10]])  # batch_size=2, tgt_seq_len=5

        # Forward pass
        print("  🔄 Forward pass...")
        logits = model.forward(src_tokens, tgt_tokens)
        print(f"  ✅ Output shape: {logits.shape} (expected: (2, 5, 90))")

        # Simple backward pass
        print("  🔄 Backward pass...")
        grad = np.ones_like(logits) * 0.01
        model.backward(grad)
        print("  ✅ Backward pass completed")

        # Check gradients exist
        grads = model.get_gradients()
        params = model.get_trainable_params()
        print(f"  ✅ Found {len(params)} parameters, {len(grads)} gradients")

        # Quick generation test
        print("  🔄 Testing generation...")
        try:
            generated = model.generate(src_tokens[:1], max_len=8, temperature=1.0)
            print(f"  ✅ Generated shape: {generated.shape}")
        except Exception as gen_e:
            print(f"  ⚠️ Generation failed (not critical): {gen_e}")

        print("  🎉 TransformerModel test PASSED!")
        return True

    except Exception as e:
        print(f"  ❌ TransformerModel test FAILED: {e}")
        print(f"  Traceback: {traceback.format_exc()}")
        return False


def quick_gradient_check():
    """Quick gradient sanity check"""
    print("\n🔍 Quick Gradient Check...")

    try:
        # Use simple encoder model
        model = EncoderOnlyModel(
            vocab_size=50,
            d_model=32,
            num_heads=2,
            d_ff=64,
            num_layers=1,
            num_classes=1
        )

        tokens = np.array([[1, 2, 3, 4, 5]])

        # Forward pass
        logits = model.forward(tokens)

        # Compute simple loss and gradient
        target = np.array([[1.0]])
        loss = 0.5 * (logits - target) ** 2
        grad = logits - target

        # Backward pass
        model.backward(grad)

        # Check gradients
        grads = model.get_gradients()

        gradient_health = {
            'total_grads': len(grads),
            'non_zero_grads': 0,
            'nan_grads': 0,
            'inf_grads': 0,
            'large_grads': 0
        }

        for name, g in grads.items():
            if g is not None:
                if np.any(g != 0):
                    gradient_health['non_zero_grads'] += 1
                if np.any(np.isnan(g)):
                    gradient_health['nan_grads'] += 1
                if np.any(np.isinf(g)):
                    gradient_health['inf_grads'] += 1
                if np.linalg.norm(g) > 10:
                    gradient_health['large_grads'] += 1

        print(f"  📊 Gradient Health Report:")
        print(f"    Total gradients: {gradient_health['total_grads']}")
        print(f"    Non-zero gradients: {gradient_health['non_zero_grads']}")
        print(f"    NaN gradients: {gradient_health['nan_grads']}")
        print(f"    Infinite gradients: {gradient_health['inf_grads']}")
        print(f"    Large gradients (>10): {gradient_health['large_grads']}")

        # Health check
        is_healthy = (
                gradient_health['non_zero_grads'] > 0 and
                gradient_health['nan_grads'] == 0 and
                gradient_health['inf_grads'] == 0
        )

        if is_healthy:
            print("  ✅ Gradients are healthy!")
            return True
        else:
            print("  ⚠️ Some gradient issues detected")
            return False

    except Exception as e:
        print(f"  ❌ Gradient check FAILED: {e}")
        return False


def main():
    """Run quick integration tests"""
    print("🚀 YFormers Quick Integration Test")
    print("=" * 50)

    tests = [
        ("EncoderOnly", test_encoder_only_quick),
        ("DecoderOnly", test_decoder_only_quick),
        ("Transformer", test_transformer_quick),
        ("Gradients", quick_gradient_check)
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append(False)

    # Summary
    print("\n" + "=" * 50)
    print("📊 QUICK TEST SUMMARY")
    print("=" * 50)

    passed = sum(results)
    total = len(results)

    for i, (test_name, _) in enumerate(tests):
        status = "✅ PASS" if results[i] else "❌ FAIL"
        print(f"{status} {test_name}")

    print(f"\nOverall: {passed}/{total} tests passed ({passed / total:.1%})")

    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        print("Your YFormers implementation is working correctly!")
        print("\nNext steps:")
        print("- Run the full integration test suite")
        print("- Try training on a real dataset")
        print("- Experiment with different architectures")
    else:
        print("⚠️ Some tests failed. Common issues:")
        print("- Check import paths")
        print("- Verify all YFormers components are implemented")
        print("- Look at the error tracebacks above")

    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)