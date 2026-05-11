# ============================================
# LANGUAGE DETECTION & TRANSLATION MODULE
# Detects if complaint is in Hindi
# and translates it to English for AI
# ============================================

from deep_translator import GoogleTranslator
from langdetect import detect

def detect_and_translate(text):
    """
    Detects language of text.
    If Hindi → translates to English.
    If English → returns as is.
    Returns: (translated_text, original_language)
    """
    try:
        # Detect language
        lang = detect(text)

        if lang == 'hi':
            # Translate Hindi to English
            translated = GoogleTranslator(
                source='hi',
                target='en'
            ).translate(text)
            return translated, 'Hindi'
        else:
            return text, 'English'

    except Exception as e:
        # If detection fails, return original
        return text, 'Unknown'


# Test it
if __name__ == '__main__':
    test_texts = [
        "हमारे क्षेत्र में 3 दिनों से पानी नहीं आ रहा है",
        "बिजली का तार गिरा हुआ है बहुत खतरनाक है",
        "सड़क पर बड़े गड्ढे हैं",
        "Water is not coming since 3 days",
        "Electric wire is sparking near my house",
    ]

    print("🧪 Testing Language Detection & Translation:")
    print("-" * 55)
    for text in test_texts:
        translated, lang = detect_and_translate(text)
        print(f"Original  : {text}")
        print(f"Language  : {lang}")
        print(f"Translated: {translated}")
        print()