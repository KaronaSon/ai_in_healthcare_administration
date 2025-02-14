import pandas as pd

# Create a sample dataset
data = {
    'diagnosis': ['diabetes', 'flu', 'hypertension', 'asthma', 'migraine'],
    'procedure': ['blood test', 'flu shot', 'blood pressure measurement', 'inhaler prescription', 'pain relief'],
    'code': ['E11', 'J10', 'I10', 'J45', 'G43']
}

df = pd.DataFrame(data)

# Save to CSV
df.to_csv('billing_coding_data.csv', index=False)

print("Sample billing_coding_data.csv created.")
