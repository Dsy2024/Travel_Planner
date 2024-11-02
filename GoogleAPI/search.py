from fuzzywuzzy import process
import pandas as pd
import chardet
import difflib

with open('schedule_2.csv', 'rb') as f:
    result = chardet.detect(f.read())

encoding = result['encoding']
schedule = pd.read_csv('schedule_2.csv', encoding=encoding)
schedule_name = schedule['basicName']
# print(schedule_name)

attractions = pd.read_csv('model_data/taipei/data.csv')
attractions.dropna(how='all', inplace=True)
# print(attractions)
attractions_name = attractions['basicName']
# print(attractions_name)

# Initialize an empty list to store matched attraction data
matched_attractions = []

# Find the closest match for each item in the schedule
for i, sched in enumerate(schedule_name):
    # Use difflib to get the closest match from the attractions list
    matches = difflib.get_close_matches(sched, attractions_name, n=1, cutoff=0.6)  # n=1 gives the best match
    if matches:
        match = matches[0]
        # Get the index of the match in the attractions DataFrame
        match_index = attractions_name[attractions_name == match].index[0]
        
        # Get the corresponding row from attractions DataFrame
        matched_attraction = attractions.loc[match_index].copy()
        matched_attraction['currentDay'] = schedule.loc[i, 'currentDay']
        matched_attraction['currentTime'] = schedule.loc[i, 'currentTime']
        
        # Append the matched row to the list
        matched_attractions.append(matched_attraction)
        
        print(f"Closest match for '{sched}' is '{match}' (index: {match_index})")
    else:
        print(f"No close match found for '{sched}'")

# Convert the list of matched attractions to a new DataFrame
matched_df = pd.DataFrame(matched_attractions)

print(matched_df)
matched_df.to_csv('matched_attractions_2.csv', index=True)