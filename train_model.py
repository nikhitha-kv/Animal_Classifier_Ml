import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.metrics import accuracy_score
# Load datasets
zoo_data = pd.read_csv('data/zoo.csv')
class_data = pd.read_csv('data/class.csv')

# Merge datasets
merged_data = pd.merge(zoo_data, class_data, left_on='class_type', right_on='Class_Number', how='inner')
merged_data = merged_data.drop(['animal_name', 'Class_Number', 'Class_Type', 'Animal_Names'], axis=1)

# Separate features and target
X = merged_data.drop('class_type', axis=1)
y = merged_data['class_type']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: ',accuracy*100,'%')

# Save the trained model and scaler
with open('model/random_forest_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('model/scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

print("Model and scaler saved.")
