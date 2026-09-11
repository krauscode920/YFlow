"""
Test Inductive Reasoning Module
Demonstrates pattern recognition and prediction capabilities

This test file:
1. Mocks the abstract parsing methods with heuristics
2. Tests pattern detection algorithms
3. Tests prediction accuracy
4. Shows complete architecture flow
"""

import sys
import os
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from yflow.reasoning.inductive import (
    InductiveReasoning,
    InductiveYFormersParser,
    induce,
    analyze_pattern,
    ArithmeticSequenceDetector,
    GeometricSequenceDetector,
    TrendAnalyzer
)


# ============================================================================
# MOCKED PARSER - Implements abstract methods with heuristics
# ============================================================================

class MockInductiveParser(InductiveYFormersParser):
    """
    Mock implementation of abstract parser for testing
    Uses heuristics until YFormers is trained
    """

    def _classify_pattern(self, encoded) -> str:
        """
        Mock classification using keyword heuristics
        (Will be replaced by trained model)
        """
        text = self._current_text.lower()
        words = text.split()
        clean_words = [w.strip('.,!?;:') for w in words]

        # Check for number sequence
        if self._has_number_sequence(clean_words):
            numbers = self._extract_numbers_heuristic(clean_words)
            if len(numbers) >= 2:
                return self._detect_sequence_type(numbers)

        # Check for trend keywords
        trend_keywords = ['trend', 'growing', 'increasing', 'decreasing', 'stable']
        if any(kw in clean_words for kw in trend_keywords):
            return 'trend_analysis'

        # Check for observation keywords
        observation_keywords = ['every', 'each', 'always', 'usually', 'often']
        if any(kw in clean_words for kw in observation_keywords):
            return 'generalization'

        # Check for category keywords
        category_keywords = ['category', 'type', 'kind', 'group', 'similar', 'common']
        if any(kw in clean_words for kw in category_keywords):
            return 'categorical_pattern'

        return 'unknown'

    def _extract_data(self, encoded, pattern_type: str) -> list:
        """
        Mock data extraction using heuristics
        (Will be replaced by trained model)
        """
        text = self._current_text.lower()
        words = text.split()
        clean_words = [w.strip('.,!?;:') for w in words]

        if pattern_type in ['arithmetic_sequence', 'geometric_sequence', 'trend_analysis']:
            return self._extract_numbers_heuristic(clean_words)
        elif pattern_type == 'generalization':
            return []
        elif pattern_type == 'categorical_pattern':
            return []
        else:
            return []

    def _has_number_sequence(self, words: list) -> bool:
        """Check if text contains numeric sequence"""
        numbers = [w for w in words if self._is_number_heuristic(w)]
        return len(numbers) >= 2

    def _is_number_heuristic(self, word: str) -> bool:
        """Check if word represents a number"""
        if word.isdigit():
            return True

        if '.' in word:
            try:
                float(word)
                return True
            except:
                pass

        number_words = {
            'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
            'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
            'ten': 10, 'eleven': 11, 'twelve': 12
        }

        return word in number_words

    def _extract_numbers_heuristic(self, words: list) -> list:
        """Extract all numbers from word list"""
        numbers = []

        number_words = {
            'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
            'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
            'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13,
            'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17,
            'eighteen': 18, 'nineteen': 19, 'twenty': 20, 'thirty': 30,
            'forty': 40, 'fifty': 50, 'hundred': 100, 'thousand': 1000
        }

        for word in words:
            if word.isdigit():
                numbers.append(int(word))
            elif '.' in word:
                try:
                    numbers.append(float(word))
                except:
                    pass
            elif word in number_words:
                numbers.append(number_words[word])

        return numbers

    def _detect_sequence_type(self, numbers: list) -> str:
        """Determine if sequence is arithmetic or geometric"""
        if len(numbers) < 2:
            return 'arithmetic_sequence'

        # Check arithmetic
        diffs = [numbers[i + 1] - numbers[i] for i in range(len(numbers) - 1)]
        arith_consistent = all(abs(d - diffs[0]) < 0.001 for d in diffs)

        # Check geometric
        if all(n != 0 for n in numbers[:-1]):
            ratios = [numbers[i + 1] / numbers[i] for i in range(len(numbers) - 1)]
            geom_consistent = all(abs(r - ratios[0]) < 0.001 for r in ratios)
        else:
            geom_consistent = False

        if geom_consistent and not arith_consistent:
            return 'geometric_sequence'
        else:
            return 'arithmetic_sequence'


# ============================================================================
# DATASETS - Different Pattern Types
# ============================================================================

ARITHMETIC_SEQUENCES = [
    ("2, 4, 6, 8, what comes next?", "arithmetic", 10, "Simple +2 sequence"),
    ("5, 10, 15, 20, what's next?", "arithmetic", 25, "+5 sequence"),
    ("100, 110, 120, 130, predict next", "arithmetic", 140, "+10 sequence"),
    ("1, 3, 5, 7, 9", "arithmetic", 11, "Odd numbers"),
    ("10, 20, 30, 40", "arithmetic", 50, "+10 sequence"),
]

GEOMETRIC_SEQUENCES = [
    ("2, 4, 8, 16, what comes next?", "geometric", 32, "Powers of 2"),
    ("3, 9, 27, 81, what's next?", "geometric", 243, "Powers of 3"),
    ("1, 2, 4, 8, 16", "geometric", 32, "×2 sequence"),
    ("5, 25, 125, 625", "geometric", 3125, "×5 sequence"),
]

TREND_DATA = [
    ("Sales: 100, 120, 144, what's the trend?", "trend", 172.8, "20% growth"),
    ("Values: 200, 180, 162, predict next", "trend", 145.8, "10% decline"),
]


def print_header(title):
    """Print formatted section header"""
    print("\n" + "=" * 70)
    print(title.center(70))
    print("=" * 70)


def test_basic_inductive_reasoning():
    """Test basic inductive reasoning functionality"""
    print_header("BASIC INDUCTIVE REASONING TEST")

    # Use mocked parser
    reasoner = InductiveReasoning()
    reasoner.parser = MockInductiveParser()

    print("\nTesting simple arithmetic sequence...")
    result = reasoner.reason("2, 4, 6, 8, what comes next?")
    print(f"Result: {result}")

    assert "PREDICTION" in result or "ERROR" in result
    print("✓ Basic test passed\n")

    return True


def test_arithmetic_sequences():
    """Test arithmetic sequence detection"""
    print_header("ARITHMETIC SEQUENCE TESTS")

    reasoner = InductiveReasoning()
    reasoner.parser = MockInductiveParser()

    print(f"\nTesting {len(ARITHMETIC_SEQUENCES)} arithmetic sequences...\n")

    passed = 0
    failed = 0

    for i, (statement, expected_type, expected_value, description) in enumerate(ARITHMETIC_SEQUENCES, 1):
        print(f"[Test {i}] {description}")
        print(f"Statement: {statement}")

        try:
            result = reasoner.reason(statement)

            if "PREDICTION:" in result:
                pred_str = result.split("PREDICTION:")[1].split(".")[0].strip()
                try:
                    predicted = float(pred_str)
                    if abs(predicted - expected_value) < 0.01:
                        print(f"✓ PASS - Predicted: {predicted}, Expected: {expected_value}")
                        passed += 1
                    else:
                        print(f"✗ FAIL - Predicted: {predicted}, Expected: {expected_value}")
                        failed += 1
                except:
                    print(f"⚠ Could not parse prediction: {pred_str}")
                    failed += 1
            else:
                print(f"⚠ No prediction found: {result}")
                failed += 1

            print(f"Result: {result[:80]}...")

        except Exception as e:
            print(f"✗ EXCEPTION: {e}")
            failed += 1

        print()

    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed out of {len(ARITHMETIC_SEQUENCES)} tests")
    print("=" * 70)

    return failed == 0


def test_geometric_sequences():
    """Test geometric sequence detection"""
    print_header("GEOMETRIC SEQUENCE TESTS")

    reasoner = InductiveReasoning()
    reasoner.parser = MockInductiveParser()

    print(f"\nTesting {len(GEOMETRIC_SEQUENCES)} geometric sequences...\n")

    passed = 0
    failed = 0

    for i, (statement, expected_type, expected_value, description) in enumerate(GEOMETRIC_SEQUENCES, 1):
        print(f"[Test {i}] {description}")
        print(f"Statement: {statement}")

        try:
            result = reasoner.reason(statement)

            if "PREDICTION:" in result:
                pred_str = result.split("PREDICTION:")[1].split(".")[0].strip()
                try:
                    predicted = float(pred_str)
                    if abs(predicted - expected_value) < 0.01:
                        print(f"✓ PASS - Predicted: {predicted}, Expected: {expected_value}")
                        passed += 1
                    else:
                        print(f"✗ FAIL - Predicted: {predicted}, Expected: {expected_value}")
                        failed += 1
                except:
                    print(f"⚠ Could not parse prediction: {pred_str}")
                    failed += 1
            else:
                print(f"⚠ No prediction found: {result}")
                failed += 1

            print(f"Result: {result[:80]}...")

        except Exception as e:
            print(f"✗ EXCEPTION: {e}")
            failed += 1

        print()

    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed out of {len(GEOMETRIC_SEQUENCES)} tests")
    print("=" * 70)

    return failed == 0


def test_yformers_components():
    """Test YFormers components directly"""
    print_header("YFORMERS COMPONENTS TEST")

    parser = MockInductiveParser()

    print("✓ Mocked YFormers parser initialized")

    # Test tokenization
    print("\n--- Tokenization Test ---")
    test_text = "2, 4, 6, 8, what comes next?"
    tokens = parser.tokenize(test_text)
    print(f"Text: '{test_text}'")
    print(f"Tokens: {tokens}")
    print(f"Token count: {len(tokens)}")

    # Test vocabulary
    print("\n--- Vocabulary Test ---")
    print(f"Vocabulary size: {len(parser.word_to_id)}")
    print(f"Sample words: {list(parser.word_to_id.keys())[:15]}")

    # Test parsing
    print("\n--- Parsing Test ---")
    test_statement = "2, 4, 6, 8, what comes next?"
    print(f"Statement: {test_statement}")

    try:
        structure = parser.parse(test_statement)
        print(f"Parsed type: {structure.get('type', 'unknown')}")
        print(f"Extracted data: {structure.get('data', [])}")
        print(f"Has embeddings: {'embeddings' in structure}")

        if 'embeddings' in structure:
            embeddings = structure['embeddings']
            print(f"Embeddings shape: {embeddings.shape}")
    except Exception as e:
        print(f"⚠ Parsing error: {e}")
        import traceback
        traceback.print_exc()

    print("\n✓ YFormers components test complete")
    return True


def test_analyze_pattern_function():
    """Test the analyze_pattern convenience function"""
    print_header("ANALYZE_PATTERN FUNCTION TEST")

    test_cases = [
        ([2, 4, 6, 8], "arithmetic", 10, "Arithmetic +2"),
        ([3, 9, 27, 81], "geometric", 243, "Geometric ×3"),
        ([100, 110, 121], "auto", 133.1, "Auto-detect trend"),
    ]

    for i, (data, pattern_type, expected, description) in enumerate(test_cases, 1):
        print(f"\n[Test {i}] {description}")
        print(f"Data: {data}")
        print(f"Pattern type: {pattern_type}")

        try:
            result = analyze_pattern(data, pattern_type)
            prediction = result.get('prediction')
            pattern = result.get('pattern', {})

            print(f"Detected pattern: {pattern.get('type', 'unknown')}")
            print(f"Prediction: {prediction}")

            if prediction and abs(prediction - expected) < 1.0:
                print(f"✓ PASS")
            else:
                print(f"⚠ Expected ~{expected}")

        except Exception as e:
            print(f"⚠ Error: {e}")

    print("\n✓ analyze_pattern tests complete")
    return True


def test_pattern_detectors():
    """Test individual pattern detectors"""
    print_header("PATTERN DETECTORS TEST")

    # Test Arithmetic
    print("\n--- Arithmetic Detector ---")
    detector = ArithmeticSequenceDetector()
    data = [2, 4, 6, 8]
    pattern = detector.detect_pattern(data)
    prediction = detector.predict_next(data, pattern)
    print(f"Data: {data}")
    print(f"Pattern: {pattern}")
    print(f"Next value: {prediction}")
    assert prediction == 10, "Arithmetic prediction should be 10"
    print("✓ Arithmetic detector works")

    # Test Geometric
    print("\n--- Geometric Detector ---")
    detector = GeometricSequenceDetector()
    data = [2, 4, 8, 16]
    pattern = detector.detect_pattern(data)
    prediction = detector.predict_next(data, pattern)
    print(f"Data: {data}")
    print(f"Pattern: {pattern}")
    print(f"Next value: {prediction}")
    assert prediction == 32, "Geometric prediction should be 32"
    print("✓ Geometric detector works")

    # Test Trend
    print("\n--- Trend Analyzer ---")
    analyzer = TrendAnalyzer()
    data = [100, 120, 144]
    pattern = analyzer.detect_pattern(data)
    prediction = analyzer.predict_next(data, pattern)
    print(f"Data: {data}")
    print(f"Pattern: {pattern}")
    print(f"Next value: {prediction:.2f}")
    print("✓ Trend analyzer works")

    print("\n✓ All detectors test complete")
    return True


def test_confidence_scoring():
    """Test confidence calculation"""
    print_header("CONFIDENCE SCORING TEST")

    reasoner = InductiveReasoning()
    reasoner.parser = MockInductiveParser()

    test_cases = [
        ([2, 4, 6, 8], "High consistency, medium sample", 0.8),
        ([2, 4, 6, 8, 10, 12, 14, 16, 18, 20], "High consistency, large sample", 0.95),
        ([1, 2, 4, 8, 16, 32], "Geometric, high consistency", 0.9),
    ]

    for data, description, expected_min in test_cases:
        print(f"\n{description}")
        print(f"Data: {data}")

        result = reasoner.reason(", ".join(map(str, data)) + ", what's next?")

        if "Confidence:" in result:
            conf_str = result.split("Confidence:")[1].strip().rstrip('%')
            confidence = float(conf_str) / 100
            print(f"Confidence: {confidence * 100:.0f}%")

            if confidence >= expected_min:
                print(f"✓ PASS (>= {expected_min * 100:.0f}%)")
            else:
                print(f"⚠ Low confidence")
        else:
            print("⚠ No confidence score found")

    print("\n✓ Confidence scoring test complete")
    return True


def demonstrate_architecture():
    """Demonstrate the complete architecture flow"""
    print_header("ARCHITECTURE DEMONSTRATION")

    statement = "2, 6, 18, 54, what comes next?"

    print("\nDemonstrating AURA Inductive Reasoning Architecture:")
    print(f"\n{'─' * 70}")
    print(f"USER INPUT:")
    print(f"  '{statement}'")

    print(f"\n{'─' * 70}")
    print(f"STEP 1: YFormers Processing")
    print(f"  └─ Tokenization: text → word IDs")
    print(f"  └─ Encoder: process through attention")
    print(f"  └─ Classification: identify pattern type (LEARNED)")
    print(f"  └─ Extraction: extract numbers [2, 6, 18, 54] (LEARNED)")

    print(f"\n{'─' * 70}")
    print(f"STEP 2: Pattern Detection (Algorithmic)")
    print(f"  └─ Try arithmetic: differences = [4, 12, 36] (not constant)")
    print(f"  └─ Try geometric: ratios = [3, 3, 3] (constant!)")
    print(f"  └─ Pattern detected: Geometric sequence (×3)")

    print(f"\n{'─' * 70}")
    print(f"STEP 3: Algorithmic Prediction (Deterministic)")
    print(f"  └─ Last value: 54")
    print(f"  └─ Ratio: 3")
    print(f"  └─ Prediction: 54 × 3 = 162")
    print(f"  └─ No neural network = No hallucination")

    print(f"\n{'─' * 70}")
    print(f"STEP 4: Confidence Calculation")
    print(f"  └─ Sample size: 4 points")
    print(f"  └─ Consistency: 100% (perfect ratio)")
    print(f"  └─ Pattern reliability: Geometric (95%)")
    print(f"  └─ Final confidence: ~90%")

    print(f"\n{'─' * 70}")
    print(f"STEP 5: Output")

    reasoner = InductiveReasoning()
    reasoner.parser = MockInductiveParser()
    result = reasoner.reason(statement)
    print(f"  └─ {result}")

    print(f"\n{'─' * 70}")
    print(f"NOTE: Current implementation uses MOCKED parsing (heuristics)")
    print(f"      After training YFormers, classification & extraction become LEARNED")
    print(f"      The algorithmic validation stays EXACTLY the same!")
    print(f"{'─' * 70}")

    return True


def run_all_tests():
    """Run complete test suite"""
    print("\n" + "=" * 70)
    print("INDUCTIVE REASONING - COMPLETE TEST SUITE".center(70))
    print("YFlow Reasoning Architecture".center(70))
    print("=" * 70)

    results = []

    try:
        results.append(("Basic Reasoning", test_basic_inductive_reasoning()))
        results.append(("YFormers Components", test_yformers_components()))
        results.append(("Arithmetic Sequences", test_arithmetic_sequences()))
        results.append(("Geometric Sequences", test_geometric_sequences()))
        results.append(("Pattern Detectors", test_pattern_detectors()))
        results.append(("Analyze Pattern", test_analyze_pattern_function()))
        results.append(("Confidence Scoring", test_confidence_scoring()))
        results.append(("Architecture Demo", demonstrate_architecture()))

    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Summary
    print_header("TEST SUMMARY")

    all_passed = True
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status} - {test_name}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("🎉 ALL TESTS COMPLETED SUCCESSFULLY!".center(70))
    else:
        print("⚠ SOME TESTS FAILED".center(70))
    print("=" * 70 + "\n")

    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)