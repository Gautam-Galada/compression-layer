"""
Simple Unit Tests for Sanitization and Dataset Manager
Run with: python -m pytest test_all.py -v
"""

import shutil
import tempfile
from pathlib import Path

import pytest

# Import the functions we're testing (handle missing modules gracefully)
try:
    from scripts.data_sanitization import (
        compute_compression_ratio,
        extract_verbose_compressed,
        is_code_sample,
        rule_a_ratio_check,
        rule_b_orphaned_symbols,
        rule_c_negation_preservation,
        rule_d_semantic_symbol_usage_nl,
    )

    HAS_SANITIZE = True
except ImportError:
    HAS_SANITIZE = False
    # Create dummy functions for tests to import

    def is_code_sample(x):
        return False

    def extract_verbose_compressed(x):
        return ("", "")

    def compute_compression_ratio(x, y):
        return 0.0

    def rule_a_ratio_check(x, y):
        return (True, "")

    def rule_b_orphaned_symbols(x):
        return (True, "")

    def rule_c_negation_preservation(x, y):
        return (True, "")

    def rule_d_semantic_symbol_usage_nl(x, y):
        return (True, "")


try:
    from scripts.dataset_manager import (
        CONFIG,
        count_samples,
        load_state,
        save_state,
    )

    HAS_DATASET_MANAGER = True
except ImportError:
    HAS_DATASET_MANAGER = False
    # Create dummy functions

    def load_state():
        return {}

    def save_state(x):
        return None

    def count_samples(x):
        return 0

    CONFIG = {}


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def temp_dir():
    """Create temp directory, cleanup after test."""
    tmp = tempfile.mkdtemp()
    yield Path(tmp)
    shutil.rmtree(tmp)


@pytest.fixture
def mock_config(temp_dir, monkeypatch):
    """Mock CONFIG to use temp directory."""
    temp_config = {
        "active_train": temp_dir / "train.jsonl",
        "original_backup": temp_dir / "train.original.jsonl",
        "sanitized_data": temp_dir / "sanitized_train.jsonl",
        "state_file": temp_dir / ".dataset_state.json",
        "log_file": temp_dir / ".dataset_changes.log",
    }
    # Import here to avoid issues if dataset_manager doesn't exist
    try:
        import scripts.dataset_manager

        monkeypatch.setattr(scripts.dataset_manager, "CONFIG", temp_config)
    except ImportError:
        pass  # Skip if dataset_manager not available
    return temp_config


# ============================================================================
# TESTS: SANITIZATION
# ============================================================================


@pytest.mark.skipif(not HAS_SANITIZE, reason="data_sanitization not available")
class TestCodeDetection:
    """Test code vs natural language detection."""

    def test_detects_python_code(self):
        assert is_code_sample("def hello():\n    return 'hi'")

    def test_detects_natural_language(self):
        assert not is_code_sample("The cat sat on the mat.")

    def test_detects_javascript(self):
        assert is_code_sample("const x = () => true")


@pytest.mark.skipif(not HAS_SANITIZE, reason="data_sanitization not available")
class TestExtraction:
    """Test extracting verbose and compressed text."""

    def test_extracts_correctly(self):
        sample = {
            "messages": [
                {"role": "user", "content": "Compress: Hello world"},
                {"role": "assistant", "content": "hi world"},
            ]
        }
        verbose, compressed = extract_verbose_compressed(sample)
        assert verbose == "Hello world"
        assert compressed == "hi world"

    def test_handles_empty_messages(self):
        sample = {"messages": []}
        verbose, compressed = extract_verbose_compressed(sample)
        assert verbose == ""
        assert compressed == ""


@pytest.mark.skipif(not HAS_SANITIZE, reason="data_sanitization not available")
class TestCompressionRatio:
    """Test compression ratio calculation."""

    def test_basic_ratio(self):
        ratio = compute_compression_ratio("one two three four", "1 2")
        assert ratio == 2.0

    def test_zero_division_safety(self):
        # Should not crash on empty compressed
        ratio = compute_compression_ratio("hello", "")
        assert ratio == 0.0


@pytest.mark.skipif(not HAS_SANITIZE, reason="data_sanitization not available")
class TestRuleA:
    """Test Rule A: compression ratio validation."""

    def test_passes_good_ratio(self):
        passed, _ = rule_a_ratio_check("one two three", "1 2")
        assert passed

    def test_fails_bad_ratio(self):
        passed, _ = rule_a_ratio_check("hi", "hello there friend")
        assert not passed


@pytest.mark.skipif(not HAS_SANITIZE, reason="data_sanitization not available")
class TestRuleB:
    """Test Rule B: orphaned symbols."""

    def test_passes_clean_text(self):
        passed, _ = rule_b_orphaned_symbols("Paris @ France")
        assert passed

    def test_fails_symbol_at_start(self):
        passed, _ = rule_b_orphaned_symbols("→ bad start")
        assert not passed

    def test_allows_colon_at_end(self):
        passed, _ = rule_b_orphaned_symbols("function:")
        assert passed


@pytest.mark.skipif(not HAS_SANITIZE, reason="data_sanitization not available")
class TestRuleC:
    """Test Rule C: negation preservation."""

    def test_passes_no_negation(self):
        passed, _ = rule_c_negation_preservation("I like it", "like")
        assert passed

    def test_passes_preserved_negation(self):
        passed, _ = rule_c_negation_preservation("I do not like it", "not like")
        assert passed

    def test_fails_lost_negation(self):
        # "never" is a negation keyword - if lost, should fail
        passed, _ = rule_c_negation_preservation("I never eat meat", "eat meat")
        assert not passed


@pytest.mark.skipif(not HAS_SANITIZE, reason="data_sanitization not available")
class TestRuleD:
    """Test Rule D: semantic symbol usage."""

    def test_passes_location_with_at(self):
        passed, _ = rule_d_semantic_symbol_usage_nl("Paris is located in France", "Paris @ France")
        assert passed

    def test_fails_location_without_at(self):
        passed, _ = rule_d_semantic_symbol_usage_nl("Tokyo is located in Japan", "Tokyo Japan")
        assert not passed


# ============================================================================
# TESTS: DATASET MANAGER
# ============================================================================


@pytest.mark.skipif(not HAS_DATASET_MANAGER, reason="dataset_manager not available")
class TestState:
    """Test state management."""

    def test_load_default_state(self, mock_config):
        pytest.importorskip("scripts.dataset_manager")  # Skip if module not available
        state = load_state()
        assert state["current"] == "original"
        assert state["change_count"] == 0

    def test_save_and_load(self, mock_config):
        pytest.importorskip("scripts.dataset_manager")  # Skip if module not available
        test_state = {"current": "sanitized", "last_change": None, "change_count": 1}
        save_state(test_state)
        loaded = load_state()
        assert loaded == test_state


class TestFileOperations:
    """Test file utilities."""

    def test_count_samples(self, temp_dir):
        test_file = temp_dir / "test.jsonl"
        with open(test_file, "w") as f:
            f.write('{"test": 1}\n')
            f.write('{"test": 2}\n')
            f.write('{"test": 3}\n')

        assert count_samples(test_file) == 3

    def test_count_nonexistent_file(self, temp_dir):
        # Should not crash
        count = count_samples(temp_dir / "missing.jsonl")
        assert count == 0

    def test_count_empty_file(self, temp_dir):
        test_file = temp_dir / "empty.jsonl"
        test_file.touch()
        assert count_samples(test_file) == 0


# ============================================================================
# EDGE CASES
# ============================================================================


@pytest.mark.skipif(not HAS_SANITIZE, reason="data_sanitization not available")
class TestEdgeCases:
    """Test edge cases and safety."""

    def test_empty_compression_ratio(self):
        # Should not crash
        ratio = compute_compression_ratio("", "")
        assert ratio == 0.0

    def test_empty_symbol_check(self):
        passed, _ = rule_b_orphaned_symbols("")
        assert not passed

    def test_unicode_handling(self):
        passed, _ = rule_b_orphaned_symbols("test → result")
        assert passed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
