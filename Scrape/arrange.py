import pandas as pd
import numpy as np
import chardet
import ast
import re


with open('details/taipei/attractions_3.csv', 'rb') as f:
    result = chardet.detect(f.read())

encoding = result['encoding']
df = pd.read_csv('details/taipei/attractions_3.csv', encoding=encoding)

# remove useless data
df['rating'] = df['rating'].str.replace('/5', '')
df['openDate'] = df['openDate'].apply(lambda x: x.replace(':', ''))
df = df[df['openTime'] != "['undefined']"]
df = df[~df['subTitleName'].str.contains('《', na=False)]
df.drop(columns=['closeDate', 'closeTime'], inplace=True)
df.replace("['N/A']", np.nan, inplace=True)
df.dropna(subset=['openDate', 'openTime', 'review'], how='all', inplace=True)
df.reset_index(drop=True, inplace=True)
df['review'] = df.apply(lambda row: "(Business hours are inaccurate) " + str(row['review'])
                        if not pd.isna(row['review']) and pd.isna(row['openDate']) or pd.isna(row['openTime'])
                        else row['review'], axis=1)

# arrange openDate
weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
def expand_days(day_range):
    # 將多個範圍或單個天數分割開來
    day_parts = day_range.split(',')
    expanded_days = []
    
    for part in day_parts:
        if '-' in part:
            # 如果包含範圍，則擴展範圍
            start_day, end_day = part.split('-')
            start_index = weekdays.index(start_day.strip())
            end_index = weekdays.index(end_day.strip())
            
            if start_index <= end_index:
                # 正常範圍內的天數
                expanded_days.extend(weekdays[start_index:end_index + 1])
            else:
                # 如果範圍跨過了週末（如 Friday-Monday）
                expanded_days.extend(weekdays[start_index:] + weekdays[:end_index + 1])
        else:
            # 如果是單個天數，直接添加
            expanded_days.append(part.strip())
    
    return expanded_days
    
days = ['mon', 'tues', 'wed', 'thurs', 'fri', 'sat', 'sun']
result = pd.DataFrame(columns=days)
for index, row in df.iterrows():
    open_dates = row['openDate']
    open_times = row['openTime']
    if pd.notna(open_dates):  # 檢查是否為 NaN
        open_dates = ast.literal_eval(open_dates)
    else:
        open_dates = []  

    if pd.notna(open_times):  # 檢查是否為 NaN
        open_times = ast.literal_eval(open_times)
        row = pd.DataFrame({day: ['0'] for day in days})  # 每行初始化為空
    else:
        open_times = ['None']  
        row = pd.DataFrame({day: [None] for day in days})  # 每行初始化為空
    for date, time in zip(open_dates, open_times):
        time = re.sub(r'\s*\(.*\)', '', time)
        if 'Open daily' in date:
            for day in days:
                row[day] = time
        else:
            if '-' in date:
                date = expand_days(date)
            if 'Monday' in date:
                row['mon'] = time
            if 'Tuesday' in date:
                row['tues'] = time
            if 'Wednesday' in date:
                row['wed'] = time
            if 'Thursday' in date:
                row['thurs'] = time
            if 'Friday' in date:
                row['fri'] = time
            if 'Saturday' in date:
                row['sat'] = time
            if 'Sunday' in date:
                row['sun'] = time
    result = pd.concat([result, row], ignore_index=True)
# print(result)
df = pd.concat([df.iloc[:, :5], result, df.iloc[:, 5:]], axis=1)

def time_to_minutes(time_str):
    if time_str is None:
        return None, None
    if 'Open all day' in time_str:
        return 0, 24 * 60  # 24小時開放，轉換為分鐘
    elif '-' in time_str:
        ranges = time_str.split(';')
        for i, time_range in enumerate(ranges):
            start_time, end_time = time_range.split('-')
            start_minutes = int(start_time.split(':')[0]) * 60 + int(start_time.split(':')[1])
            end_minutes = int(end_time.split(':')[0]) * 60 + int(end_time.split(':')[1])
            if i == 0:
                open_times = start_minutes
            close_times = end_minutes
        if open_times != 0 and close_times < open_times:
            close_times += 24 * 60
        return open_times, close_times
    return 0, 0  # 無法解析時返回0

# 將每一天的時間轉換為數值
for i, day in enumerate(['mon', 'tues', 'wed', 'thurs', 'fri', 'sat', 'sun']):
    df[day + 'Open'], df[day + 'Close'] = zip(*df[day].apply(time_to_minutes))

def convert_to_hours(time_str):
    if time_str is None or pd.isna(time_str):
        return None
    result = None
    if 'day' in time_str:
        time_str = time_str.split(' ')[0]
        if '-' in time_str:
            start_time, end_time = time_str.split('-')
            average_time = (float(start_time) + float(end_time)) / 2
            result = average_time * 24
        else:
            result = float(time_str) * 24
    if 'hour' in time_str:
        time_str = time_str.split(' ')[0]
        if '-' in time_str:
            start_time, end_time = time_str.split('-')
            average_time = (float(start_time) + float(end_time)) / 2
            result = average_time
        else:
            result = float(time_str)
    if 'minute' in time_str:
        time_str = time_str.split(' ')[0]
        if '-' in time_str:
            start_time, end_time = time_str.split('-')
            average_time = (float(start_time) + float(end_time)) / 2
            result = average_time / 60
        else:
            result = float(time_str) / 60
    
    result += 0.25
    result *= 10
    temp = result % 5
    result = int(result - temp)
    result = float(result) / 10
    if result < 0.5:
        result = 0.5
    if result > 10:
        result = 10
    
    return result

df['recommend'] = df['recommendStayTime'].apply(convert_to_hours)

def convert_price(price):
    if price == "Free":
        return 0
    elif isinstance(price, str) and price.startswith('$'):
        # 假設當前的價格是以美元為單位，轉換為 NTD 的匯率假設為 30
        exchange_rate = 30
        # 去掉 $ 符號，轉換為數字並乘以匯率
        price_in_usd = float(price.replace('$', ''))
        return price_in_usd * exchange_rate
    return price

# 應用於 DataFrame 的價格列
df['price'] = df['price'].apply(convert_price)
df.drop(['recommendStayTime'] + days, axis=1, inplace=True)

df.to_csv('arranged_data.csv', index=False, encoding='utf-8-sig')