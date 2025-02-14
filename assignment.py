import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
import joblib

# Load data
data = pd.read_csv('scheduling_data.csv')

# Convert appointment_start and appointment_end to datetime
data['appointment_start'] = pd.to_datetime(data['appointment_start'])
data['appointment_end'] = pd.to_datetime(data['appointment_end'])

# Preprocess data
data['appointment_duration'] = (data['appointment_end'] - data['appointment_start']).dt.total_seconds() / 60
data = pd.get_dummies(data, columns=['appointment_type', 'provider'])

# Feature selection
features = ['appointment_duration', 'patient_age', 'no_show', 
            'appointment_type_checkup', 'appointment_type_emergency', 
            'appointment_type_followup', 'appointment_type_consultation', 
            'provider_Dr. Smith', 'provider_Dr. Johnson', 'provider_Dr. Lee']
X = data[features]
y = pd.to_datetime(data['appointment_scheduled_time']).astype(int) // 10**9

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Train the model with cross-validation


param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5, scoring='neg_mean_absolute_error')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

# Predict and evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')

# Save the model
joblib.dump(model, 'scheduling_model.pkl')

# Create billing and coding data
data = pd.read_csv('billing_coding_data.csv')

# Ensure 'diagnosis' and 'procedure' columns are of type string
data['diagnosis'] = data['diagnosis'].astype(str)
data['procedure'] = data['procedure'].astype(str)

# Combine 'diagnosis' and 'procedure' into a single text feature
data['combined_text'] = data['diagnosis'] + ' ' + data['procedure']
X = data['combined_text']
y = data['code']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and train the pipeline model
pipeline = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000, random_state=42))
pipeline.fit(X_train, y_train)

# Predict and evaluate
y_pred = pipeline.predict(X_test)
accuracy = (y_pred == y_test).mean()
print(f'Accuracy: {accuracy}')

# Save the model
joblib.dump(pipeline, 'billing_coding_model.pkl')

# Load the models
scheduling_model = joblib.load('scheduling_model.pkl')
billing_coding_model = joblib.load('billing_coding_model.pkl')

# Create simulated environment data
simulated_data = pd.read_csv('simulated_environment_data.csv')

# Print column names and first few rows
print("Simulated Data Columns:", simulated_data.columns)
print("Simulated Data Head:", simulated_data.head())

# Ensure 'appointment_start' and 'appointment_end' are of datetime type
simulated_data['appointment_start'] = pd.to_datetime(simulated_data['appointment_start'])
simulated_data['appointment_end'] = pd.to_datetime(simulated_data['appointment_end'])

# Preprocess simulated data similar to training data
simulated_data['appointment_duration'] = (simulated_data['appointment_end'] - simulated_data['appointment_start']).dt.total_seconds() / 60.0

# Create dummy variables for categorical features
simulated_data = pd.get_dummies(simulated_data, columns=['appointment_type', 'provider'])

# Ensure all required columns are present, filling missing columns with 0
required_columns = ['appointment_duration', 'patient_age', 'no_show', 
                    'appointment_type_checkup', 'appointment_type_emergency', 'appointment_type_followup', 'appointment_type_consultation', 
                    'provider_Dr. Smith', 'provider_Dr. Johnson', 'provider_Dr. Lee']

for col in required_columns:
    if col not in simulated_data.columns:
        simulated_data[col] = 0

# Predict scheduling time
simulated_data['predicted_scheduled_time'] = scheduling_model.predict(simulated_data[required_columns])

# Convert actual_scheduled_time to Unix timestamp
simulated_data['actual_scheduled_time'] = pd.to_datetime(simulated_data['actual_scheduled_time']).astype(int) // 10**9

# Evaluate performance
if 'actual_scheduled_time' in simulated_data.columns:
    scheduling_mae = mean_absolute_error(simulated_data['actual_scheduled_time'], simulated_data['predicted_scheduled_time'])
    print(f'Scheduling Mean Absolute Error: {scheduling_mae}')
else:
    print("'actual_scheduled_time' column not found in simulated data for evaluation.")


