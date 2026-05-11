# ============================================
# SENTIMENT ANALYSIS MODULE
# Detects how strongly negative a complaint is
# Combines with urgency for smarter priority
# ============================================

from textblob import TextBlob

def analyze_sentiment(text):
    """
    Analyzes sentiment of complaint text.
    Returns:
        - sentiment_score: -1.0 (very negative) to +1.0 (positive)
        - sentiment_label: VERY NEGATIVE / NEGATIVE / NEUTRAL
        - priority_level: HIGH / MEDIUM / LOW
    """
    blob = TextBlob(text)
    score = blob.sentiment.polarity

    # Classify sentiment
    if score < -0.5:
        sentiment_label = 'VERY NEGATIVE'
        priority_level = 'HIGH'
    elif score < -0.1:
        sentiment_label = 'NEGATIVE'
        priority_level = 'MEDIUM'
    else:
        sentiment_label = 'NEUTRAL'
        priority_level = 'LOW'

    return round(score, 3), sentiment_label, priority_level


def get_combined_priority(urgency, sentiment_label):
    """
    Combines keyword urgency + sentiment
    for a smarter final priority level
    """
    if urgency == 'URGENT':
        return '🔴 CRITICAL'
    elif urgency == 'NORMAL' and sentiment_label == 'VERY NEGATIVE':
        return '🟠 HIGH'
    elif urgency == 'NORMAL' and sentiment_label == 'NEGATIVE':
        return '🟡 MEDIUM'
    else:
        return '🟢 LOW'


# ============================================
# TEST IT
# ============================================
if __name__ == '__main__':
    test_complaints = [
        "There is no water since 3 days it is very bad and dangerous situation",
        "Electric wire is sparking near my house very dangerous please help immediately",
        "Garbage not collected for a week smell is terrible disgusting",
        "Pothole on road",
        "My pension has not come this month",
        "This is absolutely terrible nobody cares about us poor citizens",
    ]

    print("😊 Sentiment Analysis Test:")
    print("-" * 60)
    for complaint in test_complaints:
        score, label, priority = analyze_sentiment(complaint)
        combined = get_combined_priority('NORMAL', label)
        print(f"Complaint : {complaint[:55]}...")
        print(f"Score     : {score} | Sentiment: {label}")
        print(f"Priority  : {combined}")
        print()