import requests
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.action_chains import ActionChains
from webdriver_manager.chrome import ChromeDriverManager
import time
import csv
import json

service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service)

urls = []

url = 'https://www.trip.com/travel-guide/attraction/taipei-360/tourist-attractions/?locale=en-US&curr=USD'
driver.get(url)
# request_url = 'https://www.trip.com/restapi/soa2/19913/getTripAttractionList'

# headers = {
#     'Content-Type': 'application/json',
#     'Accept': 'application/json',
#     'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36',
#     'Referer': 'https://www.trip.com/travel-guide/attraction/taipei-360/tourist-attractions/?locale=en-US&curr=USD',
#     'Origin': 'https://www.trip.com',
#     'Connection': 'keep-alive'
# }

# payload_template = {
#     "head": {
#         "extension": [
#             {"name": "platform", "value": "Online"},
#             {"name": "locale", "value": "en-US"}
#         ]
#     },
#     "cityId": 0,
#     "count": 10,
#     "districtId": 360,
#     "filter": {
#         "filterItems": [],
#         "coordinateFilter": {"coordinateType": "wgs84", "latitude": 0, "longitude": 0},
#         "itemTypes": ""
#     },
#     "index": 2,  # Start with the first page index
#     "keyword": None,
#     "pageId": "10650006153",
#     "returnModuleType": "product",
#     "scene": "gsDestination",
#     "sortType": 1,
#     "token": "LDgwNTk1LDEwNzU4Mjg5LDEwNTI0MjEyLDgwNTk0LDg1OTUxLDEwNTcyNTMwLDgwNTkzLDgwNTk4LDEwNTIyODcyLDIzODY1MzI0",  # Ensure this is the correct token
#     "traceId": "066a4bff-ee1f-50f3-30a9-125c92261401"
# }

# # Function to extract URLs from the response
# def extract_urls(response_content):
#     soup = BeautifulSoup(response_content, 'html.parser')
#     a_tags = soup.select("a.online-poi-item-card")
#     for a_tag in a_tags:
#         urls.append(a_tag.get('href'))

# Initial page extraction
# extract_urls(driver.page_source)

# Iterate through pages
# for i in range(2):
#     payload = payload_template.copy()
#     payload['index'] = i + 2  # Update the page index
#     print(f"index:{i+2}")
    
#     response = requests.post(
#         request_url,
#         headers=headers,
#         json=payload
#     )
    
#     if response.status_code == 200:
#         response_json = response.json()
#         response_html = response_json.get('data', {}).get('html', '')
#         extract_urls(response_html)
#     else:
#         print(f"Failed to get page {i+1}, status code: {response.status_code}")

for i in range(100):
    right_icon = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, 'li[title="Next Page"]'))
    )

    time.sleep(1)
    a_tags = driver.find_elements(By.CSS_SELECTOR, "a.online-poi-item-card")
    img_tags = driver.find_elements(By.CLASS_NAME, "bg-img")
    urls = [a_tag.get_attribute('href') for a_tag in a_tags]
    images = [img_tag.get_attribute('src') for img_tag in img_tags]
    # for a_tag in a_tags:
    #     urls.append(a_tag.get_attribute('href'))
    # driver.execute_script("arguments[0].scrollIntoView(true);", right_icon)
    
    # 保存URL到文件
    with open(f'urls/url_{i}.txt', 'w') as file:
        for url in urls:
            file.write(f"{url}\n")
    
    # 保存圖片SRC到文件
    with open(f'images/img_{i}.txt', 'w') as file:
        for img in images:
            file.write(f"{img}\n")

    actions = ActionChains(driver) # 使用ActionChains在指定位置進行點擊
    try:
        actions.move_to_element(right_icon).click().perform()
    except Exception as e:
        print(i)
        exit()
    time.sleep(3)
