import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

# --- 1. Monitoring Engine (Data Preparation & Risk Classification) ---
def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    topics = ['Diabetes', 'Obesity among adults', 'High blood pressure']
    df = df[df['Question'].str.contains('|'.join(topics), case=False, na=False)]
    df = df.dropna(subset=['DataValue'])

    def classify_risk(value):
        if value < 7: return 0
        elif value < 15: return 1
        else: return 2

    df['risk_level'] = df['DataValue'].apply(classify_risk)
    return df

# --- 2. Risk Prediction Model Builder ---
def build_model(df):
    features = ['YearStart', 'LocationDesc', 'Stratification1', 'Question']
    target = 'risk_level'
    X = df[features]
    y = df[target]

    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['LocationDesc', 'Stratification1', 'Question'])
    ], remainder='passthrough')

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print("\n--- Model Evaluation ---")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=['Low Risk', 'Moderate Risk', 'High Risk']))

    return pipeline

# --- 3. Intervention Library ---
def recommend_interventions(risk_level, question):
    print("\n--- Intervention Recommendations ---")
    if risk_level == 2:
        print("⚠️ High Risk Detected.")
        if "diabetes" in question.lower():
            print("- Reduce sugar intake\n- Increase physical activity\n- Monitor blood glucose regularly")
        if "obesity" in question.lower():
            print("- Balanced diet\n- 30 minutes of exercise daily\n- Avoid junk food")
        if "blood pressure" in question.lower():
            print("- Reduce salt intake\n- Avoid stress\n- Regular BP checks")
    elif risk_level == 1:
        print("🔶 Moderate Risk. Early prevention is advised.")
    else:
        print("✅ Low Risk. Maintain your healthy lifestyle.")

# --- 4. Complete Prediction & Planning System ---
def chronic_disease_tracker(model, input_data):
    input_df = pd.DataFrame([input_data])
    risk_level = model.predict(input_df)[0]
    risk_map = ['Low', 'Moderate', 'High']
    print(f"\nPredicted Risk Level: {risk_map[risk_level]}")
    recommend_interventions(risk_level, input_data['Question'])

# === Run the System ===
file_path = "chronic_disease_sample_1percent.csv"  # Replace with your file
df = load_and_prepare_data(file_path)
model = build_model(df)

# Test with a new sample input
def get_user_input(valid_locations, valid_strats, valid_questions):
    print("Enter your health query information:")

    year = int(input("Year (e.g., 2021): "))

    location = input(f"Location (choose from: {', '.join(valid_locations[:5])}...): ")
    if location not in valid_locations:
        print("❌ Invalid location.")
        return None

    strat = input(f"Stratification (choose from: {', '.join(valid_strats)}): ")
    if strat not in valid_strats:
        print("❌ Invalid stratification.")
        return None

    print("\nChoose a health condition:")
    for idx, q in enumerate(valid_questions, start=1):
        print(f"{idx}. {q}")
    condition_choice = input(f"Enter choice (1-{len(valid_questions)}): ")

    if not condition_choice.isdigit() or int(condition_choice) not in range(1, len(valid_questions)+1):
        print("❌ Invalid condition choice.")
        return None

    question = valid_questions[int(condition_choice)-1]

    return {
        'YearStart': year,
        'LocationDesc': location,
        'Stratification1': strat,
        'Question': question
    }
valid_locations = df['LocationDesc'].unique().tolist()
valid_strats = df['Stratification1'].unique().tolist()
valid_questions = df['Question'].unique().tolist()

# Keep asking until valid input is received
while True:
    sample_input = get_user_input(valid_locations, valid_strats, valid_questions)
    
    if sample_input:
        chronic_disease_tracker(model, sample_input)
        break  # Exit after successful prediction
    else:
        print("❌ Invalid input. Please try again.\n")
