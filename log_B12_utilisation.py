import pandas as pd
# import matplotlib.pyplot as plt # for plotting data

from bs4 import BeautifulSoup
import requests # fetches html content of a website
from urllib.request import urlopen
from urllib.request import URLError

import time # for sleep function
from datetime import datetime # for timestamp

import re # for regular expressions
import os

save2file = "log.csv"

def webpage2soup(url, parser="html.parser"):
    headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 6.1)\
     AppleWebKit/537.36 (KHTML, like Gecko)\
     Chrome/41.0.2228.0 Safari/537.36',
    }
    res = requests.get(url, headers=headers)
    res.raise_for_status()

    soup = BeautifulSoup(res.text, parser)
    return soup
            
def online(url="http://google.com"):
    Online = False
    try:
        urlopen(url, timeout=1)
        Online = True
    except URLError as err:
        Online = False
    return Online

def get_current_capacity():
    js_query = 'https://111.webclimber.de/de/trafficlight?\
    callback=WebclimberTrafficlight.insertTrafficlight&\
    key=184xNhv6RRU7H2gVg8QFyHCYxym8DKve&\
    hid=111&\
    container=trafficlightContainer&\
    type=&\
    area='
    
    soup = webpage2soup(js_query)
    percentage_str, num_str = [str(x) for x in soup.div.find_all("div")[-2:]]
    
    
    percentage_template = r"[0-9]+%"
    regex = re.compile(percentage_template)
    m = regex.search(percentage_str)
    percentage = int(m.group()[:-1])

    num_template = r"[0-9]+"
    regex = re.compile(num_template)
    m = regex.search(num_str)
    num = int(m.group())
    return num, percentage

def str2datetime(string, fmt="%H:%M"):
    if type(string) is str:
        return datetime.strptime(string, fmt)
    if type(string) is list:
        return [datetime.strptime(x, fmt) for x in string]

def is_open(timestamp=datetime.now(), return_state=True):
    state = False
    opening_times = {"Mon":["09:30","23:00"], 
                     "Tue":["09:30","23:00"], 
                     "Wed":["08:30","23:00"], 
                     "Thu":["12:30","23:00"], 
                     "Fri":["09:30","23:00"], 
                     "Sat":["10:00","22:00"], 
                     "Sun":["10:00","21:30"]}
    weekdays = list(opening_times.keys())
    
    day = weekdays[timestamp.weekday()]
    opening_time, closing_time = str2datetime(opening_times[day])
    
    if return_state:
        if timestamp.time() > opening_time.time() and timestamp.time() < closing_time.time():
            state = True
        else:
            state = False
        return state
    else:
        # needs to be tested
        opening_time, closing_time = opening_times[day]
        next_day = weekdays[(timestamp.weekday()+1)%7]
        next_opening_time, next_closing_time = str2datetime(opening_times[next_day])
        
        return closing_time, next_opening_time        
log = pd.DataFrame(columns=["Free Spots", "Capacity", "Time", "Day", "DateTime"])

if os.path.isfile(save2file):
    print("logfile already exists, adding new data to existing file.")
    log = pd.read_csv(save2file, sep=";")

while True:
    now = datetime.now() # neccesary?
    if is_open(now):
        if online():
            timestamp = datetime.now()
            timestamp_str = datetime.strftime(timestamp, "%H:%M, %m/%d/%Y")
            weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            timeofday = datetime.strftime(timestamp, "%H:%M")
            day = weekdays[timestamp.weekday()]

            print("fetching data from B12 API @ " + timestamp_str, end=" --> ")
            payload = get_current_capacity()
            absolute, relative = payload
            update = pd.DataFrame({"Free Spots":[absolute], 
                                   "Capacity":[relative], 
                                   "Time":[timeofday], 
                                   "Weekday":[day], 
                                   "DateTime":[timestamp_str]
                                  })
            print("Free Spots = {0}, Capacity = {1}%.".format(absolute, relative))
            #print("updating logfile")
            log = log.append(update, ignore_index=True, sort=False)
            log.to_csv(save2file, index=False, sep=";")
            time.sleep(60*5)
        else:
            print("Error while trying to reach internet. Waiting 5min then trying again...")
            time.sleep(60*5)
    else:
        now = datetime.now()
        closing_time, next_opening_time = is_open(now, return_state=False)
        print("The B12 is currently closed. Waiting till it opens again @ " 
              + datetime.strftime(next_opening_time, "%H:%M") + "."
             )
        time.sleep((next_opening_time-now).seconds)
