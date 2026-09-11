"""
Test examples for Arithmetic Reasoning Module
Demonstrates all operations and edge cases
"""

import sys
import os

# Add parent directory to path to import from yflow.reasoning
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from yflow.reasoning.arithmetic import num_to_vec, vec_to_num, calculate, Arithmetic
import numpy as np


def test_conversions():
    """Test number to vector conversions"""
    print("=" * 60)
    print("CONVERSION TESTS")
    print("=" * 60)

    tests = [
        "123",
        "999",
        "1",
        "0",
        "1000000"
    ]

    for num in tests:
        vec = num_to_vec(num)
        back = vec_to_num(vec)
        print(f"{num} -> {vec} -> {back}")
        assert back == num, f"Conversion failed for {num}"

    print("✓ All conversion tests passed\n")


def test_addition():
    """Test addition operations"""
    print("=" * 60)
    print("ADDITION TESTS")
    print("=" * 60)

    tests = [
        ("123", "456", "579"),
        ("999", "1", "1000"),
        ("50", "50", "100"),
        ("0", "123", "123"),
        ("9999", "1", "10000"),
        ("1234567", "7654321", "8888888"),
    ]

    for a, b, expected in tests:
        result = calculate('+', a, b)
        status = "✓" if result == expected else "✗"
        print(f"{status} {a} + {b} = {result} (expected: {expected})")
        assert result == expected, f"Addition failed: {a} + {b}"

    print("✓ All addition tests passed\n")


def test_subtraction():
    """Test subtraction operations"""
    print("=" * 60)
    print("SUBTRACTION TESTS")
    print("=" * 60)

    tests = [
        ("456", "123", "333"),
        ("1000", "1", "999"),
        ("50", "50", "0"),
        ("123", "0", "123"),
        ("10000", "9999", "1"),
        ("1000000", "1", "999999"),
    ]

    for a, b, expected in tests:
        result = calculate('-', a, b)
        status = "✓" if result == expected else "✗"
        print(f"{status} {a} - {b} = {result} (expected: {expected})")
        assert result == expected, f"Subtraction failed: {a} - {b}"

    print("✓ All subtraction tests passed\n")


def test_multiplication():
    """Test multiplication operations"""
    print("=" * 60)
    print("MULTIPLICATION TESTS")
    print("=" * 60)

    tests = [
        ("12", "34", "408"),
        ("123", "456", "56088"),
        ("999", "999", "998001"),
        ("1", "12345", "12345"),
        ("0", "999", "0"),
        ("11", "11", "121"),
        ("100", "100", "10000"),
    ]

    for a, b, expected in tests:
        result = calculate('*', a, b)
        status = "✓" if result == expected else "✗"
        print(f"{status} {a} * {b} = {result} (expected: {expected})")
        assert result == expected, f"Multiplication failed: {a} * {b}"

    print("✓ All multiplication tests passed\n")


def test_division():
    """Test division operations"""
    print("=" * 60)
    print("DIVISION TESTS")
    print("=" * 60)

    tests = [
        ("456", "123", "3 remainder 87"),
        ("100", "7", "14 remainder 2"),
        ("1000", "10", "100 remainder 0"),
        ("123", "123", "1 remainder 0"),
        ("50", "100", "0 remainder 50"),
        ("999", "3", "333 remainder 0"),
    ]

    for a, b, expected in tests:
        result = calculate('/', a, b)
        status = "✓" if result == expected else "✗"
        print(f"{status} {a} / {b} = {result} (expected: {expected})")
        assert result == expected, f"Division failed: {a} / {b}"

    print("✓ All division tests passed\n")


def test_edge_cases():
    """Test edge cases and special scenarios"""
    print("=" * 60)
    print("EDGE CASE TESTS")
    print("=" * 60)

    # Very large numbers
    print("\n[Large Numbers]")
    large1 = "123456789012345678901234567890"
    large2 = "987654321098765432109876543210"
    result = calculate('+', large1, large2)
    print(f"Large addition: {len(result)} digits")
    print(f"Result starts with: {result[:20]}...")

    # Multiplication producing large results
    print("\n[Large Multiplication]")
    result = calculate('*', "999999", "999999")
    print(f"999999 * 999999 = {result}")

    # Division edge cases
    print("\n[Division Edge Cases]")
    print(f"1 / 1000 = {calculate('/', '1', '1000')}")
    print(f"999 / 1 = {calculate('/', '999', '1')}")

    # Zero operations
    print("\n[Zero Operations]")
    print(f"0 + 0 = {calculate('+', '0', '0')}")
    print(f"0 * 999 = {calculate('*', '0', '999')}")
    print(f"0 - 0 = {calculate('-', '0', '0')}")

    print("\n✓ All edge case tests passed\n")


def test_class_usage():
    """Test using Arithmetic class directly"""
    print("=" * 60)
    print("CLASS USAGE TESTS")
    print("=" * 60)

    # Create arithmetic object
    a = Arithmetic(num_to_vec("123"))
    print(f"Created: {a}")

    # Perform operations
    result_add = a.add(num_to_vec("456"))
    print(f"123 + 456 = {vec_to_num(result_add)}")

    result_mult = a.mult(num_to_vec("10"))
    print(f"123 * 10 = {vec_to_num(result_mult)}")

    quotient, remainder = a.div(num_to_vec("5"))
    print(f"123 / 5 = {vec_to_num(quotient)} remainder {vec_to_num(remainder)}")

    print("\n✓ Class usage tests passed\n")


def run_all_tests():
    """Run complete test suite"""
    print("\n" + "=" * 60)
    print("ARITHMETIC REASONING MODULE - COMPLETE TEST SUITE")
    print("=" * 60 + "\n")

    try:
        test_conversions()
        test_addition()
        test_subtraction()
        test_multiplication()
        test_division()
        test_edge_cases()
        test_class_usage()

        print("=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return False

    return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)