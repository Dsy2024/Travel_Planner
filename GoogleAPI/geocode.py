import pandas as pd
import requests
import time
import os
from dotenv import load_dotenv

current_dir = os.path.dirname(__file__)
target_file_path = os.path.join(current_dir, '..', 'Scrape', 'data/taipei/attractions.csv')
df = pd.read_csv(target_file_path)

load_dotenv()
API_KEY = os.getenv("API_KEY")
geocode_url = "https://maps.googleapis.com/maps/api/geocode/json"

def get_lat_lng(address, index):
    params = {
        'address': address,
        'key': API_KEY
    }
    response = requests.get(geocode_url, params=params)
    geocode_data = response.json()
    
    if geocode_data['status'] == 'OK':
        location = geocode_data['results'][0]['geometry']['location']
        return location['lat'], location['lng']
    else:
        print(f"Error for address '{address}': {geocode_data['status']}")
        df.drop(index, inplace=True) 
        return None, None
    
df['latitude'] = None
df['longitude'] = None

# 遍歷 DataFrame 並更新經緯度
for index, row in df.iterrows():
    address = row['address']
    lat, lng = get_lat_lng(address, index)
    df.at[index, 'latitude'] = lat
    df.at[index, 'longitude'] = lng
    
    # 添加延遲以避免超過 API 的請求速率限制
    time.sleep(0.1)

# days = ['mon', 'tues', 'wed', 'thurs', 'fri', 'sat', 'sun']
# df.drop(['openDate', 'openTime', 'address', 'review', 'url', 'img'] + [day + 'Time' for day in days], axis=1, inplace=True)
df.reset_index(drop=True)

df.to_csv('model_data/taipei/data.csv', index=False, encoding='utf-8-sig')