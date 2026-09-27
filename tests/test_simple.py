"""Simple test to verify the modernized bot structure is working."""

def test_basic_imports():
    """Test that basic imports work."""
    from src.config import settings

    assert settings.BOT_NAME, "BOT_NAME must be configured"
    assert settings.SYMBOLS, "SYMBOLS must not be empty"
