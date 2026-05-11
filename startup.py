# ============================================
# STARTUP SCRIPT
# Runs when app first deploys on server
# Trains model if not already trained
# ============================================

import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import re
import nltk

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
from nltk.corpus import stopwords

def train_and_save():
    print("🤖 Training AI model...")

    complaints = [
        "No water supply in our area since 3 days",
        "Water pipe is broken and leaking on road",
        "Dirty contaminated water coming from tap",
        "No water coming from tap since morning",
        "Water supply cut off in our colony",
        "Underground water pipeline is damaged",
        "Water tanker has not come for 5 days",
        "Low water pressure in our area",
        "Water is yellowish and smells bad",
        "Borewell in our area is not working",
        "Water connection is broken outside my house",
        "Municipal water not supplied for a week",
        "Water meter is damaged and not working",
        "Tap water has mud and impurities",
        "Water supply timing is irregular",
        "Overhead tank in colony is empty",
        "Water leakage from main pipeline on street",
        "No drinking water available in our ward",
        "Water supply stopped without any notice",
        "Pipeline burst near market area",
        "Water coming for only 10 minutes daily",
        "New water connection application pending for months",
        "Water pump is not working in our building",
        "Sewage mixing with drinking water supply",
        "Water supply pipe broken near school",
        "Illegal water connections causing low pressure",
        "Water tank not cleaned for months",
        "No water in taps for entire street",
        "Water supply disrupted due to pipeline repair",
        "Old pipeline causing water loss in area",
        "No electricity in our area since yesterday",
        "Electric pole fallen on road is dangerous",
        "Street lights not working in our colony",
        "Power cut happening every day for 8 hours",
        "Electric wire sparking near my house",
        "Transformer has burnt in our area",
        "Getting very high voltage fluctuation",
        "Electricity bill is incorrect and too high",
        "No electricity supply since 2 days",
        "Streetlight bulb is fused near the park",
        "Electricity meter is not working properly",
        "Power supply is very unstable",
        "Electric pole is tilting and about to fall",
        "No power supply during exam time causing problems",
        "Overhead electric wire hanging dangerously low",
        "Electricity connection cut without prior notice",
        "New electricity connection application rejected",
        "Electric shock received from streetlight pole",
        "Frequent tripping of electricity in our area",
        "Substation near our area is making loud noise",
        "There are big potholes on the main road",
        "Road is completely broken and damaged",
        "No street lights on highway very dangerous",
        "Road divider is broken causing accidents",
        "Speed breaker damaged on school road",
        "Road construction work incomplete since months",
        "Footpath is broken and people are falling",
        "Road flooded due to rain and not draining",
        "Flyover has cracks and dangerous to use",
        "Road marking and signs not visible",
        "Manhole cover missing on main road",
        "Road near school has no safety barriers",
        "Bridge repair work pending since years",
        "Road dust causing respiratory problems",
        "Speed breaker not painted causing accidents at night",
        "Illegal encroachment on footpath",
        "Road blocked due to construction debris",
        "No pedestrian crossing near busy market",
        "Road accident prone area needs traffic signal",
        "Road repair done poorly and breaking again",
        "Garbage not being collected from our area",
        "Drainage blocked and dirty water overflowing",
        "Very bad smell from open garbage dump",
        "Sewage water flowing on the road",
        "Public toilet is dirty and not maintained",
        "Garbage dump near school causing health issues",
        "Drain is choked and water entering homes",
        "Dead animals lying on road not removed",
        "Garbage truck has not come since one week",
        "Open defecation happening near residential area",
        "Sanitation workers not coming to clean street",
        "Overflowing dustbin not emptied for days",
        "Mosquito breeding in stagnant water near drain",
        "Public toilet locked and not accessible",
        "Garbage being burnt causing air pollution",
        "Stray animals eating garbage on road",
        "Drainage work incomplete causing waterlogging",
        "No dustbin provided in our locality",
        "Cleaning of drain not done before monsoon",
        "Garbage disposal site too close to homes",
        "My ration card application pending since 6 months",
        "Government hospital has no doctors available",
        "Police not taking my complaint seriously",
        "Birth certificate has not been issued yet",
        "School has no teachers for last month",
        "Government office staff misbehaving with citizens",
        "Pension has not been received since 3 months",
        "Aadhar card correction not done despite visits",
        "Road permit office asking for bribe",
        "Death certificate application rejected without reason",
        "Income certificate not issued despite applying",
        "Domicile certificate application pending for months",
        "Government school has no drinking water",
        "Anganwadi not functioning properly",
        "MNREGA wages not paid to workers",
        "PM Awas Yojana benefit not received",
        "Land record correction pending for years",
        "Police verification for passport delayed",
        "Hospital not providing free medicines",
        "Scholarship amount not credited to student account",
    ]

    departments = (
        ['Water Supply'] * 30 +
        ['Electricity'] * 20 +
        ['Roads'] * 20 +
        ['Sanitation'] * 20 +
        ['Public Services'] * 20
    )

    def clean_text(text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        stop_words = set(stopwords.words('english'))
        words = text.split()
        words = [w for w in words if w not in stop_words]
        return ' '.join(words)

    df = pd.DataFrame({'complaint': complaints, 'department': departments})
    df['cleaned'] = df['complaint'].apply(clean_text)

    vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1,2))
    X = vectorizer.fit_transform(df['cleaned'])
    y = df['department']

    model = LogisticRegression(max_iter=1000, C=5)
    model.fit(X, y)

    os.makedirs('model', exist_ok=True)

    with open('model/classifier.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open('model/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

    print("✅ Model trained and saved!")

if __name__ == '__main__':
    train_and_save()