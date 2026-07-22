"""Test to verify the database configuration is using PostgreSQL."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_database_configuration():
    """Test that database configuration is correct."""
    try:
        from config import settings

        print("Database Configuration:")
        print(f"Database URL: {settings.DATABASE_URL}")

        # Verify it's PostgreSQL
        if "postgresql" in settings.DATABASE_URL:
            print("Using PostgreSQL (correct!)")
            print("Database connection string is properly configured")
        else:
            print("Not using PostgreSQL (incorrect!)")
            return False

        # Show configuration summary
        print("\nConfiguration Summary:")
        print(f"Bot Name: {settings.BOT_NAME}")
        print(f"Database: PostgreSQL at {settings.DATABASE_URL}")
        print(f"Symbols: {settings.SYMBOLS}")

        return True
    except ImportError as e:
        print(f"Import failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing Database Configuration...")
    print("=" * 50)

    if test_database_configuration():
        print("\nDatabase configuration test passed!")
        print("The bot is properly configured to use PostgreSQL.")
    else:
        print("\nDatabase configuration test failed!")
        print("Please check the database settings.")
