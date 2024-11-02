import pandas as pd

# Load CSV file
data = pd.read_csv('matched_attractions_2.csv')

# Convert to JSON
data.to_json('matched_attractions_2.json', orient='records')

print("JSON created!")
