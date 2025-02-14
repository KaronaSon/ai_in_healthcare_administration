import pandas as pd
from datetime import datetime, timedelta
import random

# Create a sample dataset
data = {
    'appointment_start': [datetime(2024, 7, 1, 9, 0) + timedelta(minutes=30*i) for i in range(20)],
    'appointment_end': [datetime(2024, 7, 1, 9, 30) + timedelta(minutes=30*i) for i in range(20)],
    'appointment_type': random.choices(['checkup', 'emergency', 'followup', 'consultation'], k=20),
    'provider': random.choices(['Dr. Smith', 'Dr. Johnson', 'Dr. Lee'], k=20),
    'patient_age': random.choices(range(20, 80), k=20),
    'no_show': random.choices([0, 1], k=20),
    'appointment_scheduled_time': [datetime(2024, 6, 28, 12, 0) + timedelta(days=i%3) for i in range(20)],
    'combined_text': ['diabetes blood test', 'flu flu shot', 'hypertension blood pressure measurement', 'asthma inhaler prescription', 'migraine pain relief'] * 4,
    'actual_code': ['E11', 'J10', 'I10', 'J45', 'G43'] * 4,
    'actual_scheduled_time': [datetime(2024, 6, 28, 12, 0) + timedelta(days=i%3) for i in range(20)]
}

df = pd.DataFrame(data)

# Save to CSV
df.to_csv('simulated_environment_data.csv', index=False)

print("Sample simulated_environment_data.csv created.")
