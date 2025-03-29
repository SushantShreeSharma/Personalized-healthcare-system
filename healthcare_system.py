# Import necessary  libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
try:
    data = pd.read_csv('medicine.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: File '.csv' not found. Ensure the file is in the correct directory.")
    exit()

# Explore the dataset
print("First 5 rows of the dataset:")
print(data.head())
print("\nDataset Summary:")
print(data.describe())

# Preprocess the data
# Separate features and target labels
try:
    X = data.drop('recommendation', axis=1)
    y = data['recommendation']
except KeyError:
    print("Error: 'recommendation' column not found in the dataset.")
    exit()

# Identify numerical and categorical features
numerical_features = ['age', 'blood_pressure', 'cholesterol', 'heart_rate']
categorical_features = ['gender', 'smoking_status', 'exercise_level']

# Create preprocessing pipelines
numerical_pipeline = Pipeline(steps=[
    ('scaler', StandardScaler())
])
categorical_pipeline = Pipeline(steps=[
    ('encoder', OneHotEncoder(drop='first'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_pipeline, numerical_features),
    ('cat', categorical_pipeline, categorical_features)
])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a complete pipeline with preprocessing and model training
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Train the model
model_pipeline.fit(X_train, y_train)

# Evaluate the model
y_pred = model_pipeline.predict(X_test)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Generate personalized recommendations
def generate_recommendations(patient_data):
    """
    Generate personalized recommendations for a patient.
    :param patient_data: DataFrame with patient details
    :return: Recommendation string
    """
    try:
        prediction = model_pipeline.predict(patient_data)
        # Map predictions to actual recommendations
        recommendation_mapping = {
            0: 'No action needed',
            1: 'Regular check-up',
            2: 'Lifestyle changes',
            3: 'Medication'
        }
        return recommendation_mapping.get(prediction[0], "Unknown Recommendation")
    except Exception as e:
        return f"Error in generating recommendation: {e}"

# Example patient data for recommendation generation
example_patient_data = pd.DataFrame({
    'age': [45],
    'gender': ['Male'],
    'blood_pressure': [130],
    'cholesterol': [200],
    'heart_rate': [80],
    'smoking_status': ['Non-smoker'],
    'exercise_level': ['Moderate']
})

# Generate and display the recommendation for the example patient
recommendation = generate_recommendations(example_patient_data)
print("\nPersonalized Recommendation for Example Patient:")
print(recommendation)
