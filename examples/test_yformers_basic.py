# test_yformers_basic.py
"""
Basic functionality test for YFormers transformer architecture.
Tests that models can be created and produce correct output shapes.
"""

import numpy as np
import sys
import os

# Add yflow to path if needed
# sys.path.append('path/to/yflow')

from yflow.yformers.model import TransformerModel, EncoderOnlyModel, DecoderOnlyModel
from yflow.yformers.attention import SelfAttention, MultiHeadAttention
from yflow.yformers.embeddings import TokenEmbedding, PositionalEmbedding


def test_basic_components():
    """Test individual YFormers components"""
    print("🧪 Testing YFormers Components...")

    try:
        # Test TokenEmbedding
        print("  - Testing TokenEmbedding...")
        token_embed = TokenEmbedding(vocab_size=100, embed_dim=64)
        tokens = np.array([[1, 2, 3, 4, 0]])  # batch_size=1, seq_len=5
        embedded = token_embed.forward(tokens)
        assert embedded.shape == (1, 5, 64), f"Expected (1, 5, 64), got {embedded.shape}"

        # Test SelfAttention
        print("  - Testing SelfAttention...")
        self_attn = SelfAttention(embed_dim=64)
        attn_out = self_attn.forward(embedded)
        assert attn_out.shape == (1, 5, 64), f"Expected (1, 5, 64), got {attn_out.shape}"

        # Test MultiHeadAttention
        print("  - Testing MultiHeadAttention...")
        mha = MultiHeadAttention(embed_dim=64, num_heads=8)
        mha_out = mha.forward(embedded)
        assert mha_out.shape == (1, 5, 64), f"Expected (1, 5, 64), got {mha_out.shape}"

        print("  ✅ All components work!")
        return True

    except Exception as e:
        print(f"  ❌ Component test failed: {e}")
        return False


def test_transformer_model():
    """Test complete Transformer model (encoder-decoder)"""
    print("🧪 Testing TransformerModel...")

    try:
        # Create small transformer
        model = TransformerModel(
            src_vocab_size=100,
            tgt_vocab_size=100,
            d_model=64,
            num_heads=4,
            d_ff=128,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dropout=0.1,
            max_src_len=50,
            max_tgt_len=50
        )

        # Create dummy data
        src = np.array([[1, 2, 3, 4, 0]])  # batch_size=1, src_seq_len=5
        tgt = np.array([[1, 2, 3, 0, 0]])  # batch_size=1, tgt_seq_len=5

        # Forward pass
        logits = model.forward(src, tgt)
        expected_shape = (1, 5, 100)  # (batch_size, tgt_seq_len, tgt_vocab_size)
        assert logits.shape == expected_shape, f"Expected {expected_shape}, got {logits.shape}"

        # Test generation
        generated = model.generate(src, max_len=10)
        assert generated.shape[0] == 1, "Generated batch size should be 1"
        assert generated.shape[1] <= 10, "Generated sequence should not exceed max_len"

        print("  ✅ TransformerModel works!")
        return True

    except Exception as e:
        print(f"  ❌ TransformerModel test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_encoder_only_model():
    """Test EncoderOnly model (BERT-style)"""
    print("🧪 Testing EncoderOnlyModel...")

    try:
        # Test without classification head
        model = EncoderOnlyModel(
            vocab_size=100,
            d_model=64,
            num_heads=4,
            d_ff=128,
            num_layers=2,
            max_seq_len=50,
            num_classes=0  # No classification head
        )

        tokens = np.array([[1, 2, 3, 4, 0]])  # batch_size=1, seq_len=5
        encoded = model.forward(tokens)
        expected_shape = (1, 5, 64)  # (batch_size, seq_len, d_model)
        assert encoded.shape == expected_shape, f"Expected {expected_shape}, got {encoded.shape}"

        # Test with classification head
        model_cls = EncoderOnlyModel(
            vocab_size=100,
            d_model=64,
            num_heads=4,
            d_ff=128,
            num_layers=2,
            max_seq_len=50,
            num_classes=10  # 10 classes
        )

        logits = model_cls.forward(tokens)
        expected_shape = (1, 10)  # (batch_size, num_classes)
        assert logits.shape == expected_shape, f"Expected {expected_shape}, got {logits.shape}"

        print("  ✅ EncoderOnlyModel works!")
        return True

    except Exception as e:
        print(f"  ❌ EncoderOnlyModel test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_decoder_only_model():
    """Test DecoderOnly model (GPT-style)"""
    print("🧪 Testing DecoderOnlyModel...")

    try:
        model = DecoderOnlyModel(
            vocab_size=100,
            d_model=64,
            num_heads=4,
            d_ff=128,
            num_layers=2,
            max_seq_len=50
        )

        tokens = np.array([[1, 2, 3, 4, 0]])  # batch_size=1, seq_len=5
        logits = model.forward(tokens)
        expected_shape = (1, 5, 100)  # (batch_size, seq_len, vocab_size)
        assert logits.shape == expected_shape, f"Expected {expected_shape}, got {logits.shape}"

        # Test generation
        prompt = np.array([[1, 2]])  # batch_size=1, prompt_len=2
        generated = model.generate(prompt, max_len=10)
        assert generated.shape[0] == 1, "Generated batch size should be 1"
        assert generated.shape[1] <= 10, "Generated sequence should not exceed max_len"
        assert generated.shape[1] >= 2, "Generated sequence should include prompt"

        print("  ✅ DecoderOnlyModel works!")
        return True

    except Exception as e:
        print(f"  ❌ DecoderOnlyModel test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_different_shapes():
    """Test models with different input shapes"""
    print("🧪 Testing Different Input Shapes...")

    try:
        model = EncoderOnlyModel(
            vocab_size=50,
            d_model=32,
            num_heads=4,
            d_ff=64,
            num_layers=1,
            max_seq_len=20,
            num_classes=0
        )

        # Test different batch sizes and sequence lengths
        test_cases = [
            (1, 5),  # batch_size=1, seq_len=5
            (2, 3),  # batch_size=2, seq_len=3
            (4, 10),  # batch_size=4, seq_len=10
            (1, 1),  # batch_size=1, seq_len=1
        ]

        for batch_size, seq_len in test_cases:
            tokens = np.random.randint(1, 49, size=(batch_size, seq_len))
            output = model.forward(tokens)
            expected_shape = (batch_size, seq_len, 32)
            assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
            print(f"  ✅ Shape test passed for batch_size={batch_size}, seq_len={seq_len}")

        print("  ✅ All shape tests passed!")
        return True

    except Exception as e:
        print(f"  ❌ Shape test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all basic functionality tests"""
    print("🚀 Starting YFormers Basic Functionality Tests")
    print("=" * 50)

    tests = [
        test_basic_components,
        test_transformer_model,
        test_encoder_only_model,
        test_decoder_only_model,
        test_different_shapes
    ]

    passed = 0
    total = len(tests)

    for test_func in tests:
        if test_func():
            passed += 1
        print()

    print("=" * 50)
    print(f"📊 Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests PASSED! YFormers is working correctly.")
        return True
    else:
        print("⚠️  Some tests FAILED. Check the errors above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)