# ============================================
# UNIQUE ID GENERATOR MODULE
# Generates human-readable, collision-resistant complaint IDs
# Format: CMP-XXXX-XXXX (e.g., CMP-8F3K-22)
# ============================================

import random
import string
import sqlite3
import time

# Configuration
ID_PREFIX = "CMP"
ID_LENGTH = 4  # Characters in the middle section
ID_SUFFIX_LENGTH = 2  # Year suffix (2 digits)

def generate_id_suffix():
    """Generate year suffix (last 2 digits of current year)"""
    return str(time.localtime().tm_year)[-2:]

def generate_random_segment(length=ID_LENGTH):
    """Generate a random alphanumeric segment"""
    # Use uppercase letters and digits (readable, avoids confusion like 0/O, 1/I)
    characters = string.ascii_uppercase.replace('O', '').replace('I', '') + string.digits.replace('0', '').replace('1', '')
    return ''.join(random.choices(characters, k=length))

def generate_complaint_id():
    """Generate a new complaint ID in format: CMP-XXXX-XXXX"""
    suffix = generate_id_suffix()
    middle = generate_random_segment(ID_LENGTH)
    return f"{ID_PREFIX}-{middle}-{suffix}"

def check_id_exists(db_path, complaint_id):
    """Check if a complaint ID already exists in database"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT 1 FROM grievances WHERE complaint_id = ?', (complaint_id,))
    exists = cursor.fetchone() is not None
    conn.close()
    return exists

def generate_unique_id(db_path='grievances.db', max_attempts=10):
    """
    Generate a unique complaint ID with collision handling.
    Retries up to max_attempts if ID already exists.
    """
    for attempt in range(max_attempts):
        new_id = generate_complaint_id()

        if not check_id_exists(db_path, new_id):
            return new_id

        # If collision, try again (very unlikely with our format)
        if attempt < max_attempts - 1:
            continue

    # Fallback: Use UUID if all attempts fail (extremely rare)
    import uuid
    return f"CMP-{uuid.uuid4().hex[:8].upper()}"


# ============================================
# TEST
# ============================================
if __name__ == '__main__':
    print("🆔 Complaint ID Generator Test")
    print("-" * 40)

    # Test unique ID generation
    print("\nGenerating 10 sample IDs:")
    for i in range(10):
        print(f"  {i+1}. {generate_unique_id()}")

    print("\n✅ ID Generator ready!")