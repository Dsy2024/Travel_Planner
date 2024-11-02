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
images = []
details_list = []

# urls for attractions
for i in range(30, 100):
    with open(f'urls/url_{i}.txt', 'r', encoding='utf-8') as file:
        urls.extend([line.strip() for line in file])
    with open(f'images/img_{i}.txt', 'r', encoding='utf-8') as file:
        images.extend([line.strip() for line in file])

# testing
# urls.append("https://www.trip.com/travel-guide/attraction/taipei/the-grand-hotel-taipei-west-secret-passage-140042904?curr=USD&locale=en-US")

# scraping elements of attractions
for url, img in zip(urls, images):

    driver.get(url)

    try:
        info = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, '//*[contains(@class, "TopBoxStyle")]'))
        )
    except TimeoutException:
        continue

    # info = soup.find("div", class_="TopBoxStyle")
    # with open('output.html', 'w', encoding='utf-8') as file:
    #     file.write(info.prettify())

    details = {}
    try:
        details['basicName'] = info.find_element(By.CLASS_NAME, 'basicName').text
    except Exception as e:
        details['basicName'] = "N/A"
    try:
        details['subTitleName'] = info.find_element(By.CLASS_NAME, 'subTitleName').text
    except Exception as e:
        details['subTitleName'] = "N/A"

    try:
        rating_element = driver.find_element(By.CLASS_NAME, 'gl-poi-detail-rating')
        rating_spans = rating_element.find_elements(By.TAG_NAME, 'span')
        details['rating'] = " " + rating_spans[0].text + rating_spans[1].text
    except Exception as e:
        details['rating'] = "N/A"

    details['closeDate'] = "N/A"
    details['closeTime'] = "N/A"
    details['openDate'] = []
    details['openTime'] = []
    try:
        time_element = driver.find_element(By.CSS_SELECTOR, 'i.gs-trip-iconfont.icon-opentime')
        time_element.click()
        # time.sleep(5)
        time_info = driver.find_element(By.CLASS_NAME, 'OnlinePopUpTime-jljzy3-10')
        open_dates = time_info.find_elements(By.CLASS_NAME, 'online-open-time-txt-title')
        open_times = time_info.find_elements(By.CLASS_NAME, 'online-open-time-txt-ctt')
        if open_dates:
            for open_date, open_time in zip(open_dates, open_times): 
                if "Closed" in open_time.text:
                    details['closeDate'] = open_date.text
                    details['closeTime'] = open_time.text
                else:
                    details['openDate'].append(open_date.text)
                    details['openTime'].append(open_time.text)
        elif open_times:
            details['openDate'].append("Open daily")
            details['openTime'].append(open_times[0].text)
        else:
            details['openDate'].append("N/A")
            details['openTime'].append("N/A")  
    except Exception as e:
        details['openDate'].append("N/A")
        details['openTime'].append("N/A")

    details['recommendStayTime'] = "N/A"
    details['address'] = "N/A"
    try:
        poi_top_infos = info.find_elements(By.CSS_SELECTOR, 'div.POITopInfo-jljzy3-11')
        for poi_top_info in poi_top_infos:
            try:
                time_span = poi_top_info.find_element(By.XPATH, './/span[text()="Recommended sightseeing time:"]')
                if time_span:
                    time_field = time_span.find_element(By.XPATH, './following-sibling::span[@class="field"]')
                    if time_field:
                        details['recommendStayTime'] = time_field.text
            except Exception as e:
                pass
            try:
                address_span = poi_top_info.find_element(By.CSS_SELECTOR, 'div.address-text-info')
                if address_span:
                    address_field = address_span.find_element(By.CSS_SELECTOR, 'span.field')
                    if address_field:
                        details['address'] = address_field.text
            except Exception as e:
                pass
    except Exception as e:
        pass

    try:
        price_element = driver.find_element(By.CLASS_NAME, 'TopPriceStyle-sc-1ono9bs-0')
        details['price'] = price_element.find_element(By.CSS_SELECTOR, '.tour-price span').text
    except Exception as e:
        details['price'] = "Free"

    try: 
        review_element = driver.find_element(By.CLASS_NAME, 'comment-box')
        details['review'] = review_element.find_element(By.TAG_NAME, 'p').text
    except Exception as e:
        details['review'] = "N/A"

    details['url'] = url
    details['img'] = img
    details_list.append(details)

fieldnames = ['basicName', 'subTitleName', 'rating', 'openDate', 'openTime', 'closeDate', 'closeTime', 'recommendStayTime', 'address', 'price', 'label', 'review', 'url', 'img']

# writing the elements into csv
# with open('attractions.csv', mode='w', newline='', encoding='utf-8') as file:
with open('details/taipei/attractions_2.csv', mode='w', newline='', encoding='utf-8-sig') as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    
    for detail in details_list:
        writer.writerow(detail)