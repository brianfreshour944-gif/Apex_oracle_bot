"""Simple test to verify the modernized bot structure is working."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_basic_imports():
    """Test that basic imports work."""
    try:
        from config import settings
        print("Configuration module imported successfully")
        print(f"Bot name: {settings.BOT_NAME}")
        print(f"Symbols: {settings.SYMBOLS}")
        return True
    except ImportError as e:
        print(f"Import failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing modernized bot structure...")
    print("=" * 50)

    if test_basic_imports():
        print("\nBasic structure test passed!")
        print("The modernized bot is properly set up.")
    else:
        print("\nBasic structure test failed!")
        print("Please check the installation.")
