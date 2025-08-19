#!/usr/bin/env python3
"""
Transformer Debug Helper - Step 2 Troubleshooting

This script helps debug common issues with transformer integration:
1. Shape mismatches
2. Gradient flow problems
3. Method missing errors
4. Device compatibility issues
"""

import numpy as np
import sys
import traceback
from typing import Any, Dict, List


def debug_imports():
    """Debug import issues"""
    print("🔍 Debugging Imports...")

    import_tests = [
        ("yflow.core.device", "Device"),
        ("yflow.core.model", "Model"),
        ("yflow.layers.dense", "Dense"),
        ("yflow.layers.normalization", "LayerNorm"),
        ("yflow.layers.activations", "GELU", "ReLU"),
        ("yflow.yformers.embeddings", "TokenEmbedding", "PositionalEmbedding"),
        ("yflow.yformers.attention", "MultiHeadAttention", "SelfAttention"),
        ("yflow.yformers.encoder", "EncoderBlock", "EncoderStack", "FeedForward"),
        ("yflow.yformers.decoder", "DecoderBlock", "DecoderStack"),
        ("yflow.yformers.model", "EncoderOnlyModel", "DecoderOnlyModel", "TransformerModel"),
        ("yflow.yformers.utils", "create_padding_mask", "create_look_ahead_mask")
    ]

    failed_imports = []

    for import_info in import_tests:
        module_name = import_info[0]
        class_names = import_info[1:]

        try:
            module = __import__(module_name, fromlist=class_names)
            missing_classes = []

            for class_name in class_names:
                if not hasattr(module, class_name):
                    missing_classes.append(class_name)

            if missing_classes:
                failed_imports.append(f"{module_name}: missing {missing_classes}")
            else:
                print(f"  ✅ {module_name}: {class_names}")

        except ImportError as e:
            failed_imports.append(f"{module_name}: {e}")

    if failed_imports:
        print("\n❌ Import Issues Found:")
        for issue in failed_imports:
            print(f"  - {issue}")
        return False
    else:
        print("✅ All imports successful!")
        return True


def debug_basic_components():
    """Debug individual components before full integration"""
    print("\n🔍 Debugging Basic Components...")

    try:
        from yflow.yformers.embeddings import TokenEmbedding
        from yflow.yformers.attention import MultiHeadAttention
        from yflow.yformers.encoder import FeedForward
        from yflow.layers.normalization import LayerNorm
    except ImportError as e:
        print(f"❌ Can't import components: {e}")
        return False

    # Test TokenEmbedding
    try:
        print("  🔧 Testing TokenEmbedding...")
        embed = TokenEmbedding(vocab_size=100, embed_dim=64)
        tokens = np.array([[1, 2, 3, 4]])
        embedded = embed.forward(tokens)
        print(f"    ✅ Embedding: {tokens.shape} -> {embedded.shape}")
    except Exception as e:
        print(f"    ❌ TokenEmbedding failed: {e}")
        return False

    # Test MultiHeadAttention
    try:
        print("  🔧 Testing MultiHeadAttention...")
        mha = MultiHeadAttention(embed_dim=64, num_heads=4)
        x = np.random.randn(2, 8, 64)  # batch=2, seq=8, dim=64
        output = mha.forward(x)
        print(f"    ✅ Attention: {x.shape} -> {output.shape}")
    except Exception as e:
        print(f"    ❌ MultiHeadAttention failed: {e}")
        print(f"    Details: {traceback.format_exc()}")
        return False

    # Test FeedForward
    try:
        print("  🔧 Testing FeedForward...")
        ff = FeedForward(embed_dim=64, ff_dim=128)
        x = np.random.randn(2, 8, 64)
        output = ff.forward(x)
        print(f"    ✅ FeedForward: {x.shape} -> {output.shape}")
    except Exception as e:
        print(f"    ❌ FeedForward failed: {e}")
        print(f"    Details: {traceback.format_exc()}")
        return False

    # Test LayerNorm
    try:
        print("  🔧 Testing LayerNorm...")
        ln = LayerNorm(64)
        x = np.random.randn(2, 8, 64)
        output = ln.forward(x)
        print(f"    ✅ LayerNorm: {x.shape} -> {output.shape}")
    except Exception as e:
        print(f"    ❌ LayerNorm failed: {e}")
        return False

    print("✅ Basic components working!")
    return True


def debug_model_creation():
    """Debug model creation issues"""
    print("\n🔍 Debugging Model Creation...")

    try:
        from yflow.yformers.model import EncoderOnlyModel, DecoderOnlyModel, TransformerModel
    except ImportError as e:
        print(f"❌ Can't import models: {e}")
        return False

    models_to_test = [
        ("EncoderOnlyModel", EncoderOnlyModel, {
            'vocab_size': 100,
            'd_model': 64,
            'num_heads': 4,
            'd_ff': 128,
            'num_layers': 1
        }),
        ("DecoderOnlyModel", DecoderOnlyModel, {
            'vocab_size': 100,
            'd_model': 64,
            'num_heads': 4,
            'd_ff': 128,
            'num_layers': 1
        }),
        ("TransformerModel", TransformerModel, {
            'src_vocab_size': 100,
            'tgt_vocab_size': 100,
            'd_model': 64,
            'num_heads': 4,
            'd_ff': 128,
            'num_encoder_layers': 1,
            'num_decoder_layers': 1
        })
    ]

    for model_name, model_class, config in models_to_test:
        try:
            print(f"  🔧 Creating {model_name}...")
            model = model_class(**config)
            print(f"    ✅ {model_name} created successfully")

            # Test basic methods exist
            methods_to_check = ['forward', 'backward', 'get_trainable_params', 'get_gradients', 'update_params']
            for method in methods_to_check:
                if not hasattr(model, method):
                    print(f"    ⚠️ Missing method: {method}")

        except Exception as e:
            print(f"    ❌ {model_name} creation failed: {e}")
            print(f"    Details: {traceback.format_exc()}")
            return False

    print("✅ All models can be created!")
    return True


def debug_forward_pass():
    """Debug forward pass issues"""
    print("\n🔍 Debugging Forward Pass...")

    try:
        from yflow.yformers.model import EncoderOnlyModel

        model = EncoderOnlyModel(
            vocab_size=50,
            d_model=32,
            num_heads=2,
            d_ff=64,
            num_layers=1,
            num_classes=2
        )

        # Test with different input shapes
        test_inputs = [
            (np.array([[1, 2, 3, 4]]), "single sequence"),
            (np.array([[1, 2, 3, 4], [5, 6, 7, 8]]), "batch of sequences"),
            (np.array([[1, 2, 3, 0], [5, 6, 0, 0]]), "sequences with padding")
        ]

        for tokens, description in test_inputs:
            try:
                print(f"  🔧 Testing {description}: {tokens.shape}")
                output = model.forward(tokens)
                print(f"    ✅ Output shape: {output.shape}")

                # Check output properties
                if np.any(np.isnan(output)):
                    print(f"    ⚠️ Output contains NaN values")
                if np.any(np.isinf(output)):
                    print(f"    ⚠️ Output contains infinite values")

            except Exception as e:
                print(f"    ❌ Failed on {description}: {e}")
                print(f"    Details: {traceback.format_exc()}")
                return False

    except Exception as e:
        print(f"❌ Forward pass debugging failed: {e}")
        return False

    print("✅ Forward pass working!")
    return True


def debug_backward_pass():
    """Debug backward pass and gradient issues"""
    print("\n🔍 Debugging Backward Pass...")

    try:
        from yflow.yformers.model import EncoderOnlyModel

        model = EncoderOnlyModel(
            vocab_size=30,
            d_model=32,
            num_heads=2,
            d_ff=64,
            num_layers=1,
            num_classes=1
        )

        tokens = np.array([[1, 2, 3, 4, 5]])

        # Forward pass
        print("  🔧 Forward pass...")
        output = model.forward(tokens)
        print(f"    ✅ Forward output shape: {output.shape}")

        # Simple gradient
        print("  🔧 Backward pass...")
        grad = np.ones_like(output)
        model.backward(grad)
        print("    ✅ Backward pass completed")

        # Check gradients
        print("  🔧 Checking gradients...")
        params = model.get_trainable_params()
        grads = model.get_gradients()

        print(f"    📊 Parameters: {len(params)}")
        print(f"    📊 Gradients: {len(grads)}")

        # Analyze gradient health
        gradient_issues = []
        for name, grad in grads.items():
            if grad is None:
                gradient_issues.append(f"{name}: gradient is None")
            elif np.any(np.isnan(grad)):
                gradient_issues.append(f"{name}: contains NaN")
            elif np.any(np.isinf(grad)):
                gradient_issues.append(f"{name}: contains inf")
            elif np.all(grad == 0):
                gradient_issues.append(f"{name}: all zeros")

        if gradient_issues:
            print("    ⚠️ Gradient issues found:")
            for issue in gradient_issues[:5]:  # Show first 5
                print(f"      - {issue}")
        else:
            print("    ✅ All gradients look healthy!")

    except Exception as e:
        print(f"❌ Backward pass debugging failed: {e}")
        print(f"Details: {traceback.format_exc()}")
        return False

    print("✅ Backward pass working!")
    return True


def debug_device_compatibility():
    """Debug device (CPU/GPU) compatibility"""
    print("\n🔍 Debugging Device Compatibility...")

    try:
        from yflow.core.device import Device, is_gpu_available
        from yflow.yformers.model import EncoderOnlyModel

        # Test CPU
        print("  🔧 Testing CPU device...")
        cpu_device = Device('cpu')
        print(f"    ✅ CPU device created: {cpu_device}")

        model = EncoderOnlyModel(
            vocab_size=20,
            d_model=16,
            num_heads=2,
            d_ff=32,
            num_layers=1
        ).to('cpu')

        tokens = np.array([[1, 2, 3]])
        output = model.forward(tokens)
        print(f"    ✅ CPU forward pass: {output.shape}")

        # Test GPU if available
        if is_gpu_available():
            print("  🔧 Testing GPU device...")
            try:
                gpu_model = EncoderOnlyModel(
                    vocab_size=20,
                    d_model=16,
                    num_heads=2,
                    d_ff=32,
                    num_layers=1
                ).to('gpu')

                gpu_output = gpu_model.forward(tokens)
                print(f"    ✅ GPU forward pass: {gpu_output.shape}")
            except Exception as e:
                print(f"    ⚠️ GPU test failed: {e}")
        else:
            print("    ℹ️ GPU not available, skipping GPU tests")

    except Exception as e:
        print(f"❌ Device compatibility debugging failed: {e}")
        return False

    print("✅ Device compatibility working!")
    return True


def debug_shape_issues():
    """Debug common shape-related issues"""
    print("\n🔍 Debugging Shape Issues...")

    try:
        from yflow.yformers.model import DecoderOnlyModel

        model = DecoderOnlyModel(
            vocab_size=50,
            d_model=32,
            num_heads=4,
            d_ff=64,
            num_layers=1
        )

        # Test different sequence lengths
        test_cases = [
            ((1, 4), "single short sequence"),
            ((2, 8), "batch of sequences"),
            ((1, 16), "longer sequence"),
            ((3, 6), "different batch size")
        ]

        for (batch_size, seq_len), description in test_cases:
            try:
                print(f"  🔧 Testing {description}: ({batch_size}, {seq_len})")
                tokens = np.random.randint(1, 50, size=(batch_size, seq_len))
                output = model.forward(tokens)
                expected_shape = (batch_size, seq_len, 50)  # vocab_size

                if output.shape == expected_shape:
                    print(f"    ✅ Shape correct: {output.shape}")
                else:
                    print(f"    ⚠️ Shape mismatch: expected {expected_shape}, got {output.shape}")

            except Exception as e:
                print(f"    ❌ Failed on {description}: {e}")
                return False

    except Exception as e:
        print(f"❌ Shape debugging failed: {e}")
        return False

    print("✅ Shape handling working!")
    return True


def main():
    """Run comprehensive debugging"""
    print("🔧 YFormers Integration Debugging Suite")
    print("=" * 60)

    debug_functions = [
        ("Import Check", debug_imports),
        ("Basic Components", debug_basic_components),
        ("Model Creation", debug_model_creation),
        ("Forward Pass", debug_forward_pass),
        ("Backward Pass", debug_backward_pass),
        ("Device Compatibility", debug_device_compatibility),
        ("Shape Handling", debug_shape_issues)
    ]

    results = []
    for debug_name, debug_func in debug_functions:
        try:
            print(f"\n{'=' * 20} {debug_name} {'=' * 20}")
            result = debug_func()
            results.append((debug_name, result))
        except Exception as e:
            print(f"❌ {debug_name} crashed: {e}")
            results.append((debug_name, False))

    # Summary
    print(f"\n{'=' * 60}")
    print("🔍 DEBUG SUMMARY")
    print("=" * 60)

    for debug_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {debug_name}")

    passed = sum(result for _, result in results)
    total = len(results)

    print(f"\nOverall: {passed}/{total} debug tests passed")

    if passed == total:
        print("🎉 All debugging tests passed!")
        print("Your YFormers setup looks good. Try running the integration tests.")
    else:
        print("⚠️ Some issues found. Address the failures above before integration testing.")

    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)