import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pickle

# Dataset load karo
data = pd.read_csv('Symptom2Disease.csv')

# Data dekho
print(data.head())
print("Shape of dataset:", data.shape)

# Saare unique symptoms collect karo
all_symptoms = set()
for col in ['Symptom_1', 'Symptom_2', 'Symptom_3']:
    all_symptoms.update(data[col].str.strip().unique())
all_symptoms.discard('')  # Empty strings hatao

# Unique symptoms dekho
print("Unique symptoms:", len(all_symptoms))

# Binary DataFrame banayein
binary_data = pd.DataFrame(0, index=data.index, columns=list(all_symptoms))
for idx, row in data.iterrows():
    for col in ['Symptom_1', 'Symptom_2', 'Symptom_3']:
        symptom = row[col].strip()
        if symptom in binary_data.columns:
            binary_data.at[idx, symptom] = 1
binary_data['disease'] = data['label']

# Features aur target alag karo
X = binary_data.drop('disease', axis=1)
y = binary_data['disease']

# Unique diseases dekho
print("Unique diseases:", y.nunique())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models define karo
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(kernel='rbf', random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "XGBoost": XGBClassifier(eval_metric='mlogloss', random_state=42)
}

# Target ko encode karo (XGBoost ke liye)
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Models train aur test karo
for name, model in models.items():
    if name == "XGBoost":
        model.fit(X_train, y_train_encoded)
        y_pred = model.predict(X_test)
        y_pred = label_encoder.inverse_transform(y_pred)
        accuracy = accuracy_score(y_test, y_pred)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy:.4f}")

# Best model (Random Forest) ko final model ke roop mein train karo
final_model = RandomForestClassifier(n_estimators=100, random_state=42)
final_model.fit(X_train, y_train)

# Test accuracy
y_pred = final_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Final Model (Random Forest) Accuracy on Test Data:", accuracy)

# Model save karo
with open('disease_model.pkl', 'wb') as f:
    pickle.dump(final_model, f)

# Symptom columns save karo
symptom_columns = X.columns.tolist()
with open('symptom_columns.pkl', 'wb') as f:
    pickle.dump(symptom_columns, f)

print("Model and symptom columns saved!")