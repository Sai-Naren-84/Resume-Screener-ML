import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from role_database import ROLE_SKILLS

# =====================
# LOAD DATA
# =====================

data = pd.read_csv("dataset.csv")

# =====================
# TRAIN MODEL
# =====================

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(data["skills"])

y = data["role"]

model = LogisticRegression(max_iter=1000)

model.fit(X, y)

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

# =====================
# USER INPUT
# =====================

print("=" * 40)
print("AI CAREER ADVISOR")
print("=" * 40)

skills = input("\nEnter skills: ").lower()

user_skills = set(skills.split())

new_resume = [skills]

new_resume_vector = vectorizer.transform(new_resume)

# =====================
# ROLE PREDICTION
# =====================

prediction = model.predict(new_resume_vector)

predicted_role = prediction[0]

probabilities = model.predict_proba(
    new_resume_vector
)[0]

confidence = max(probabilities) * 100

classes = model.classes_

results = list(zip(classes, probabilities))

results.sort(
    key=lambda x: x[1],
    reverse=True
)

# =====================
# TARGET ROLE
# =====================

print("\nAvailable Roles:")

for role in ROLE_SKILLS:
    print("-", role)

choice = input(
    "\nAre you targeting a role? (yes/no): "
).lower()

target_role = None

if choice == "yes":
    target_role = input(
        "Enter target role: "
    ).lower()

# =====================
# REPORT
# =====================

print("\n" + "=" * 40)
print("CAREER REPORT")
print("=" * 40)

print("\nPredicted Role:")
print(predicted_role)

print("\nConfidence:")
print(f"{confidence:.2f}%")

print("\nRole Alignment:")

for role, prob in results[:3]:
    print(
        f"{role} : {prob*100:.2f}%"
    )

# =====================
# TARGET ANALYSIS
# =====================

if target_role in ROLE_SKILLS:

    required_skills = set(
        ROLE_SKILLS[target_role]
    )

    matched_skills = (
        user_skills &
        required_skills
    )

    missing_skills = (
        required_skills -
        user_skills
    )

    readiness = (
        len(matched_skills)
        /
        len(required_skills)
    ) * 100

    print("\nTarget Role:")
    print(target_role)

    print("\nReadiness:")
    print(f"{readiness:.2f}%")

    print("\nSkills Found:")

    if matched_skills:
        for skill in sorted(matched_skills):
            print("✓", skill)
    else:
        print("None")

    print("\nMissing Skills:")

    if missing_skills:
        for skill in sorted(missing_skills):
            print("✗", skill)
    else:
        print("None")

    print("\nNext Skills To Learn:")

    for skill in sorted(missing_skills)[:3]:
        print("-", skill)

    if readiness < 30:
        print(
            "\nSuggestion: Focus on building "
            "foundational skills first."
        )

    elif readiness < 70:
        print(
            "\nSuggestion: You are on the right path. "
            "Fill the missing skill gaps."
        )

    else:
        print(
            "\nSuggestion: Strong alignment. "
            "Start building advanced projects."
        )

else:
    print(
        "\nTip: Choose a target role next time "
        "for personalized guidance."
    )

print("\n" + "=" * 40)