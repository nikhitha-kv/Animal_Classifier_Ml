import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the zoo dataset
zoo_data = pd.read_csv('data/zoo.csv')

# Drop unnecessary columns (if any)
# Assuming 'type' column corresponds to the class type or label, adjust accordingly
# You may need to adjust this based on your actual column names in zoo_data
zoo_data = zoo_data.drop(['class_type'], axis=1)

# Check for missing values
print("\nMissing Values:")
print(zoo_data.isnull().sum())

# Separate features and target variable
X = zoo_data.drop('animal_name', axis=1)
y = zoo_data['animal_name']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and test sets (for demonstration purposes, adjust as needed)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize the model (for demonstration, adjust this based on your model selection)
model = RandomForestClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Save the trained model and scaler
model_path = 'model/random_forest_model.pkl'
scaler_path = 'model/scaler.pkl'

with open(model_path, 'wb') as model_file:
    pickle.dump(model, model_file)

with open(scaler_path, 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

# Define input features
input_features = list(X.columns)

@app.route('/')
def index():
    return render_template('index.html', columns=input_features)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get form data
            features = []
            for column in input_features:
                feature_value = float(request.form[column])
                features.append(feature_value)
            
            # Prepare input for prediction
            input_data = pd.DataFrame([features], columns=input_features)
            
            # Feature scaling
            with open(scaler_path, 'rb') as scaler_file:
                scaler = pickle.load(scaler_file)
            
            scaled_features = scaler.transform(input_data)
            
            # Make prediction
            with open(model_path, 'rb') as model_file:
                model = pickle.load(model_file)
            
            predicted_animal_name = model.predict(scaled_features)[0]  # Predicted animal name
            
            if all(x == 0 for x in features):  # Check if all input features are zero
                return render_template('not_found.html')  # Render a template for "Animal not found" case
            else:
                return render_template('result.html', animal_name=predicted_animal_name)
        
        except Exception as e:
            return 'Error: {}'.format(e)

if __name__ == '__main__':
    app.run(debug=True)
