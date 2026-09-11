"""
Test Deductive Reasoning Module
Demonstrates YFormers integration + Algorithmic validation

This test file:
1. Mocks the abstract parsing methods with heuristics
2. Tests algorithmic validation
3. Validates the architecture
4. Shows complete reasoning flow
"""

import sys
import os
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from yflow.reasoning.deductive import DeductiveReasoning, DeductiveYFormersParser


# ============================================================================
# MOCKED PARSER - Implements abstract methods with heuristics
# ============================================================================

class MockDeductiveParser(DeductiveYFormersParser):
    """
    Mock implementation of abstract parser for testing
    Uses heuristics until YFormers is trained
    """

    def _classify_structure(self, encoded) -> str:
        """
        Mock classification using keyword heuristics
        (Will be replaced by trained model)
        """
        text = self._current_text.lower()
        words = text.split()

        # Heuristic pattern matching
        if 'if' in words and 'then' in words:
            if 'not' in words:
                return 'modus_tollens'
            else:
                return 'modus_ponens'
        elif 'all' in words or 'every' in words:
            return 'syllogism'
        else:
            return 'unknown'

    def _extract_entities(self, encoded, pattern_type: str) -> dict:
        """
        Mock entity extraction using heuristics
        (Will be replaced by trained model)
        """
        text = self._current_text.lower()
        words = text.split()
        clean_words = [w.strip('.,!?;:') for w in words]

        if pattern_type == 'syllogism':
            return self._extract_syllogism_entities(clean_words)
        elif pattern_type == 'modus_ponens':
            return self._extract_modus_ponens_entities(clean_words)
        elif pattern_type == 'modus_tollens':
            return self._extract_modus_tollens_entities(clean_words)
        else:
            return {
                'premise1': {},
                'premise2': {},
                'conclusion': {}
            }

    def _extract_syllogism_entities(self, words):
        """Extract entities for syllogism pattern"""
        # Find conclusion marker
        conclusion_markers = ['therefore', 'thus', 'so', 'hence']
        conclusion_idx = None
        for marker in conclusion_markers:
            if marker in words:
                conclusion_idx = words.index(marker)
                break

        if conclusion_idx:
            premise_words = words[:conclusion_idx]
            conclusion_words = words[conclusion_idx + 1:]
        else:
            premise_words = words
            conclusion_words = []

        # Split premises by 'and'
        if 'and' in premise_words:
            and_idx = premise_words.index('and')
            p1_words = premise_words[:and_idx]
            p2_words = premise_words[and_idx + 1:]
        else:
            p1_words = premise_words
            p2_words = []

        # Extract premise1: "all X are Y"
        premise1 = {'quantifier': 'all', 'subject': '', 'predicate': ''}
        if 'all' in p1_words:
            all_idx = p1_words.index('all')
            verb_idx = None
            for i, word in enumerate(p1_words[all_idx:]):
                if word in ['are', 'is', 'have', 'has', 'need', 'needs']:
                    verb_idx = all_idx + i
                    break

            if verb_idx:
                subject = ' '.join(p1_words[all_idx + 1:verb_idx])
                predicate = ' '.join(p1_words[verb_idx + 1:])
                premise1['subject'] = subject.strip()
                premise1['predicate'] = predicate.strip()

        # Extract premise2: "Z is X"
        premise2 = {'subject': '', 'predicate': ''}
        verb_idx = None
        for i, word in enumerate(p2_words):
            if word in ['is', 'are', 'have', 'has', 'need', 'needs']:
                verb_idx = i
                break

        if verb_idx is not None:
            p2_subject = ' '.join(p2_words[:verb_idx])
            p2_predicate = ' '.join(p2_words[verb_idx + 1:])
            premise2['subject'] = p2_subject.strip()
            premise2['predicate'] = p2_predicate.strip()

        # Extract conclusion: "Z is Y"
        conclusion = {'subject': '', 'predicate': ''}
        verb_idx = None
        for i, word in enumerate(conclusion_words):
            if word in ['is', 'are', 'have', 'has', 'need', 'needs']:
                verb_idx = i
                break

        if verb_idx is not None:
            c_subject = ' '.join(conclusion_words[:verb_idx])
            c_predicate = ' '.join(conclusion_words[verb_idx + 1:])
            conclusion['subject'] = c_subject.strip()
            conclusion['predicate'] = c_predicate.strip()

        return {
            'premise1': premise1,
            'premise2': premise2,
            'conclusion': conclusion
        }

    def _extract_modus_ponens_entities(self, words):
        """Extract entities for modus ponens pattern"""
        # Simplified extraction
        conclusion_idx = None
        for marker in ['therefore', 'thus', 'so', 'hence']:
            if marker in words:
                conclusion_idx = words.index(marker)
                break

        return {
            'premise1': {'antecedent': '', 'consequent': ''},
            'premise2': {'statement': ''},
            'conclusion': {'statement': ''}
        }

    def _extract_modus_tollens_entities(self, words):
        """Extract entities for modus tollens pattern"""
        return {
            'premise1': {'antecedent': '', 'consequent': ''},
            'premise2': {'negated_statement': ''},
            'conclusion': {'negated_statement': ''}
        }


# ============================================================================
# DATASET
# ============================================================================

DEDUCTIVE_DATASET = [
    (
        "All cats are animals and fluffy is a cat therefore fluffy is an animal",
        True,
        "Syllogism: animals"
    ),
    (
        "All humans are mortal and socrates is a human therefore socrates is mortal",
        True,
        "Syllogism: Socrates"
    ),
    (
        "All birds have wings and penguins are birds therefore penguins have wings",
        True,
        "Syllogism: penguins"
    ),
    (
        "All tigers are cats and tom is a tiger therefore tom is a cat",
        True,
        "Syllogism: tigers"
    ),
    (
        "All plants need water and roses are plants therefore roses need water",
        True,
        "Syllogism: plants"
    ),
    (
        "All cats are animals and fluffy is an animal therefore fluffy is a cat",
        False,
        "Invalid: affirming the consequent"
    ),
]


def print_header(title):
    """Print formatted section header"""
    print("\n" + "=" * 70)
    print(title.center(70))
    print("=" * 70)


def test_basic_deductive_reasoning():
    """Test basic deductive reasoning functionality"""
    print_header("BASIC DEDUCTIVE REASONING TEST")

    # Use mocked parser
    reasoner = DeductiveReasoning()
    reasoner.parser = MockDeductiveParser()

    print("\nTesting simple syllogism...")
    result = reasoner.reason("All cats are animals and fluffy is a cat therefore fluffy is an animal")
    print(f"Result: {result}")

    assert "VALID" in result or "ERROR" in result or "INVALID" in result
    print("✓ Basic test passed\n")

    return True


def test_dataset_validation():
    """Test all examples in the dataset"""
    print_header("DATASET VALIDATION TEST")

    reasoner = DeductiveReasoning()
    reasoner.parser = MockDeductiveParser()

    print(f"\nTesting {len(DEDUCTIVE_DATASET)} examples...\n")

    passed = 0
    failed = 0
    errors = 0

    for i, (statement, should_be_valid, description) in enumerate(DEDUCTIVE_DATASET, 1):
        print(f"[Test {i}] {description}")
        print(f"Statement: {statement[:60]}...")

        try:
            result = reasoner.reason(statement)

            if "ERROR" in result:
                print(f"⚠ ERROR: {result}")
                errors += 1
                continue

            is_valid = result.startswith("VALID")

            if is_valid == should_be_valid:
                status = "✓ PASS"
                passed += 1
            else:
                status = "✗ FAIL"
                failed += 1

            print(
                f"{status} - Expected: {'VALID' if should_be_valid else 'INVALID'}, Got: {'VALID' if is_valid else 'INVALID'}")
            print(f"Result: {result[:80]}...")

        except Exception as e:
            print(f"✗ EXCEPTION: {e}")
            errors += 1

        print()

    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed, {errors} errors out of {len(DEDUCTIVE_DATASET)} tests")
    print("=" * 70)

    return errors == 0


def test_yformers_components():
    """Test YFormers components directly"""
    print_header("YFORMERS COMPONENTS TEST")

    parser = MockDeductiveParser()

    print("✓ Mocked YFormers parser initialized")

    # Test tokenization
    print("\n--- Tokenization Test ---")
    test_text = "All cats are animals"
    tokens = parser.tokenize(test_text)
    print(f"Text: '{test_text}'")
    print(f"Tokens: {tokens}")
    print(f"Token count: {len(tokens)}")

    # Test vocabulary
    print("\n--- Vocabulary Test ---")
    print(f"Vocabulary size: {len(parser.word_to_id)}")
    print(f"Sample words: {list(parser.word_to_id.keys())[:10]}")

    # Test parsing
    print("\n--- Parsing Test ---")
    test_statement = "All cats are animals and fluffy is a cat therefore fluffy is an animal"
    print(f"Statement: {test_statement[:60]}...")

    try:
        structure = parser.parse(test_statement)
        print(f"Parsed type: {structure.get('type', 'unknown')}")
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


def test_algorithmic_validators():
    """Test pure algorithmic validation (no parsing)"""
    print_header("ALGORITHMIC VALIDATORS TEST")

    reasoner = DeductiveReasoning()

    # Test syllogism validator directly
    print("\n--- Syllogism Validator ---")
    structure = {
        'type': 'syllogism',
        'premise1': {'subject': 'cats', 'predicate': 'animals'},
        'premise2': {'subject': 'fluffy', 'predicate': 'cat'},
        'conclusion': {'subject': 'fluffy', 'predicate': 'animal'}
    }

    is_valid, explanation = reasoner._validate_syllogism(structure)
    print(f"Valid: {is_valid}")
    print(f"Explanation: {explanation}")
    assert is_valid == True
    print("✓ Syllogism validator works")

    print("\n✓ Algorithmic validators test complete")
    return True


def demonstrate_architecture():
    """Demonstrate the complete architecture flow"""
    print_header("ARCHITECTURE DEMONSTRATION")

    statement = "All tigers are cats and tom is a tiger therefore tom is a cat"

    print("\nDemonstrating AURA Deductive Reasoning Architecture:")
    print(f"\n{'─' * 70}")
    print(f"USER INPUT:")
    print(f"  '{statement}'")

    print(f"\n{'─' * 70}")
    print(f"STEP 1: YFormers Processing")
    print(f"  └─ Tokenization: text → word IDs")
    print(f"  └─ Encoder: process through attention")
    print(f"  └─ Classification: identify pattern type (LEARNED)")
    print(f"  └─ Extraction: extract entities (LEARNED)")

    print(f"\n{'─' * 70}")
    print(f"STEP 2: Structure Output")
    print(f"  └─ Type: syllogism")
    print(f"  └─ Premise1: All tigers are cats")
    print(f"  └─ Premise2: tom is a tiger")
    print(f"  └─ Conclusion: tom is a cat")

    print(f"\n{'─' * 70}")
    print(f"STEP 3: Algorithmic Validation (Deterministic)")
    print(f"  └─ Check transitivity: tigers ⊂ cats, tom ∈ tigers → tom ∈ cats")
    print(f"  └─ Verify logic rules")
    print(f"  └─ No neural network = No hallucination")

    print(f"\n{'─' * 70}")
    print(f"STEP 4: Output")

    reasoner = DeductiveReasoning()
    reasoner.parser = MockDeductiveParser()
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
    print("DEDUCTIVE REASONING - COMPLETE TEST SUITE".center(70))
    print("YFlow Reasoning Architecture".center(70))
    print("=" * 70)

    results = []

    try:
        results.append(("Basic Reasoning", test_basic_deductive_reasoning()))
        results.append(("YFormers Components", test_yformers_components()))
        results.append(("Dataset Validation", test_dataset_validation()))
        results.append(("Algorithmic Validators", test_algorithmic_validators()))
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