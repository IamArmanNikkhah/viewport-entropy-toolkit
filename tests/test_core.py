"""Basic tests for viewport entropy toolkit."""

from viewport_entropy_toolkit import core

def test_module_imports():
    """Test that the module imports successfully."""
    assert hasattr(core, 'analyze_viewport')
