# ============================================
# GRIEVANCE CLASSIFICATION MODEL - IMPROVED
# ============================================

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
import nltk
import re

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
from nltk.corpus import stopwords

# ============================================
# TRAINING DATA — 30 complaints per department
# ============================================

complaints = [
    # --- WATER SUPPLY (30) ---
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

    # --- ELECTRICITY (30) ---
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
    "Power outage for 12 hours continuously",
    "Electric wire touching tree branches dangerously",
    "Electricity department not responding to complaints",
    "Meter reading being done incorrectly",
    "No electricity in government school building",
    "Underground cable damaged causing power failure",
    "Electricity supply disrupted due to storm",
    "High tension wire passing over residential area",
    "Electricity pole installed blocking road",
    "Generator in area making noise and polluting",

    # --- ROADS (30) ---
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
    "Drainage on roadside is overflowing on road",
    "Heavy vehicles damaging residential road",
    "Road near hospital always congested",
    "Street name board missing in our area",
    "No road in newly developed colony",
    "Road divider plants blocking visibility",
    "Mud road needs to be converted to concrete",
    "Road excavated for pipeline not repaired",
    "Cycle track blocked by parked vehicles",
    "Road near railway crossing very narrow",

    # --- SANITATION (30) ---
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
    "Sweeper not coming to clean our street",
    "Plastic waste dumped near water body",
    "Septic tank overflowing in residential area",
    "Community toilet has no water supply",
    "Garbage dumped on empty plot illegally",
    "Street dog menace increasing due to garbage",
    "Drain cleaning required before rainy season",
    "Industrial waste being dumped in open area",
    "No proper waste segregation in our ward",
    "Complaint about overflowing sewer line",

    # --- PUBLIC SERVICES (30) ---
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
    "Fair price shop not giving proper ration",
    "Electricity subsidy not applied to bill",
    "Caste certificate rejected multiple times",
    "Government contractor doing poor quality work",
    "RTI application not responded to in time",
    "Municipal tax receipt not issued",
    "Fire NOC application pending since long",
    "Senior citizen pension application rejected",
    "Government bus not running on scheduled route",
    "Mid day meal not served in government school",
]

departments = (
    ['Water Supply'] * 30 +
    ['Electricity'] * 30 +
    ['Roads'] * 30 +
    ['Sanitation'] * 30 +
    ['Public Services'] * 30
)

# ============================================
# CLEAN TEXT FUNCTION
# ============================================

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

df = pd.DataFrame({'complaint': complaints, 'department': departments})
df['cleaned'] = df['complaint'].apply(clean_text)

print("✅ Data loaded and cleaned!")
print(f"   Total complaints: {len(df)}")

# ============================================
# TRAIN MODEL
# ============================================

vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1,2))
X = vectorizer.fit_transform(df['cleaned'])
y = df['department']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=1000, C=5)
model.fit(X_train, y_train)

print("✅ Model trained successfully!")

# ============================================
# CHECK ACCURACY
# ============================================

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 Model Accuracy: {accuracy * 100:.2f}%")
print("\n📊 Detailed Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# ============================================
# URGENCY DETECTION
# ============================================

URGENT_KEYWORDS = [
    'dangerous', 'sparking', 'accident', 'emergency', 'fire',
    'electric shock', 'collapsed', 'dead', 'injury', 'contaminated',
    'burst', 'flooding', 'fallen', 'tilting', 'health', 'hospital'
]

def check_urgency(text):
    text_lower = text.lower()
    for keyword in URGENT_KEYWORDS:
        if keyword in text_lower:
            return '🔴 URGENT'
    return '🟢 NORMAL'

# ============================================
# SAVE MODEL
# ============================================

with open('model/classifier.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('model/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("\n✅ Model saved successfully!")

# ============================================
# TEST WITH SAMPLE COMPLAINTS
# ============================================

def predict(complaint_text):
    cleaned = clean_text(complaint_text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    confidence = model.predict_proba(vectorized).max() * 100
    urgency = check_urgency(complaint_text)
    return prediction, confidence, urgency

test_complaints = [
    "Water is not coming from tap since 2 days",
    "Electric wire is sparking near my house very dangerous",
    "There are big potholes on road near school",
    "Garbage has not been collected for a week",
    "My pension has not come this month",
    "Drainage is blocked and sewage overflowing on street",
    "No electricity since yesterday and food is spoiling",
]

print("\n🧪 Testing with sample complaints:")
print("-" * 60)
for complaint in test_complaints:
    dept, conf, urgency = predict(complaint)
    print(f"Complaint : {complaint}")
    print(f"Department: {dept} | Confidence: {conf:.1f}% | {urgency}")
    print()