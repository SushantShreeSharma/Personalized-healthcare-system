Overview
This project provides a personalized medicine recommendation system using machine learning. It helps users find the best medicines based on medical conditions using TF-IDF Vectorization and cosine similarity.

Features
Search for medicines based on a condition (e.g., Diabetes, Hypertension)
Uses TF-IDF to analyze and match medical descriptions
Displays top 5 recommended medicines
Implemented using Python, Pandas, Scikit-learn, and Streamlit
Works seamlessly in Google Colab

Tech Stack
Python
Pandas, NumPy (Data Processing)
Scikit-learn (Machine Learning)
Streamlit (User Interface)

Dataset
The dataset is stored in Google Drive and is automatically loaded into the project

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

