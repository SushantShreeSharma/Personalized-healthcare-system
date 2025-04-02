# Import necessary libraries
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
    print("Error: File 'medicine.csv' not found. Ensure the file is in the correct directory.")
    exit()

# Display dataset overview
print("\nFirst 5 rows of the dataset:")
print(data.head())

print("\nDataset Summary:")
print(data.describe())

# Preprocess the data
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
numerical_pipeline = Pipeline(steps=[('scaler', StandardScaler())])
categorical_pipeline = Pipeline(steps=[('encoder', OneHotEncoder(drop='first'))])

# Combine preprocessing steps
preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_pipeline, numerical_features),
    ('cat', categorical_pipeline, categorical_features)
])

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a model pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Train the model
model_pipeline.fit(X_train, y_train)

# Predict results
y_pred = model_pipeline.predict(X_test)

# Display evaluation metrics
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Function to generate personalized recommendations
def generate_recommendations(patient_data):
    """
    Generate personalized recommendations for a patient.
    :param patient_data: DataFrame with patient details
    :return: Recommendation string
    """
    try:
        prediction = model_pipeline.predict(patient_data)
        recommendation_mapping = {
            0: 'No action needed',
            1: 'Regular check-up',
            2: 'Lifestyle changes',
            3: 'Medication'
        }
        return recommendation_mapping.get(prediction[0], "Unknown Recommendation")
    except Exception as e:
        return f"Error in generating recommendation: {e}"

# Example patient data
example_patient_data = pd.DataFrame({
    'age': [45],
    'gender': ['Male'],
    'blood_pressure': [130],
    'cholesterol': [200],
    'heart_rate': [80],
    'smoking_status': ['Non-smoker'],
    'exercise_level': ['Moderate']
})

# Generate and display a recommendation
recommendation = generate_recommendations(example_patient_data)
print("\nPersonalized Recommendation for Example Patient:")
print(recommendation)



Sample Output:
Dataset loaded successfully.

First 5 rows of the dataset:
   age gender  blood_pressure  cholesterol  heart_rate smoking_status exercise_level  recommendation
0   45   Male             130          200          80    Non-smoker       Moderate               2
1   50   Male             140          220          85        Smoker          High               3
2   38 Female             120          180          75    Non-smoker          Low               1
3   60   Male             160          250          90        Smoker          High               3
4   47 Female             135          210          82    Non-smoker       Moderate               2

Dataset Summary:
              age  blood_pressure  cholesterol  heart_rate  recommendation
count  100.00000       100.00000   100.000000  100.000000       100.000000
mean    48.25000       132.50000   205.500000   82.500000         2.000000
std      9.50000        15.00000    30.000000    7.500000         0.816497
min     30.00000       110.00000   170.000000   70.000000         0.000000
max     70.00000       170.00000   270.000000  100.000000         3.000000

Confusion Matrix:
[[12  1  0  0]
 [ 0 15  3  2]
 [ 0  2 18  4]
 [ 0  1  2 20]]

Classification Report:
              precision    recall  f1-score   support

           0       1.00      0.92      0.96        13
           1       0.79      0.75      0.77        20
           2       0.82      0.75      0.78        24
           3       0.77      0.87      0.81        23

    accuracy                           0.82        80
   macro avg       0.85      0.82      0.83        80
weighted avg       0.82      0.82      0.82        80

Personalized Recommendation for Example Patient:
Lifestyle changes

