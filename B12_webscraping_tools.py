import pandas as pd
import numpy as np
import re

from bs4 import BeautifulSoup
import requests # fetches html content of a website
from urllib.request import HTTPError # for catching timeout for website response
from urllib.request import urlopen
from urllib.request import URLError

import time # for sleep function
from datetime import datetime, date, time, timedelta

from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import matplotlib.pyplot as plt 
    
def import_logged_data(loc="./log.csv"):
    data = pd.read_csv(loc, sep=";")
    data = data.set_index("datetime")
    data.index = pd.to_datetime(data.index)
    data = data.sort_index()
    return data

def deploy_data_logger():
    log = pd.DataFrame(columns=["datetime", "free spots", "capacity"])
    save2file = "log.csv"

    while True:
        if online():
            abs_cap, rel_cap = get_current_capacity()
            timestamp = datetime.now()
            timestamp_str = datetime.strftime(timestamp, "%H:%M, %m/%d/%Y")

            update = pd.DataFrame({"datetime":[timestamp_str], "free spots":[abs_cap], "capacity":[rel_cap]})
            log = log.append(update, ignore_index=True, sort=False)
            log.to_csv(save2file, index=False, sep=";")
            print( "[Success] " + timestamp_str + " - Logfile was updated. Free Spots = {0}, Capacity = {1}%.".format(abs_cap, rel_cap) )

            time.sleep(10)
        else:
            timestamp = datetime.now()
            timestamp_str = datetime.strftime(timestamp, "%H:%M, %m/%d/%Y")
            print("[Failure] " + timestamp_str + " - Internet is not reachable. Retrying in 5min.")
            time.sleep(10)

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


def simulate_scraper(year=2021, month=5, day=23, hour=10, minute=0, update_every=15, n_updates=100000, return_as="dataframe"):
    weekdays_dict = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}

    d = date(year, month, day)
    t = time(hour, minute)
    
    datetimes = np.array([datetime.combine(d, t) + n*timedelta(minutes=update_every) for n in range(n_updates)])
    weekdays_str = np.array([weekdays_dict[datetime.weekday()] for datetime in datetimes])
    weekdays_numeric = np.array([datetime.weekday() for datetime in datetimes])
    times = np.array([datetime.time() for datetime in datetimes])

    hours = np.array([datetime.hour for datetime in datetimes])
    capacity = 30+5*(weekdays_numeric == 4) +5*(weekdays_numeric == 2) + 40*np.sin(np.pi*(hours+1)/24) + 10*np.random.randn(n_updates).astype('int64')

    if "frame" in return_as.lower():
        data = np.vstack([datetimes, times, weekdays_str, capacity])
        df = pd.DataFrame(data.T, columns=["datetime", "time", "weekday", "capacity"])
        df = df.astype({"capacity": int}).set_index("datetime")
        return df
    if "array" in return_as.lower():
        data = np.vstack([datetimes, times, weekdays_numeric, capacity])
        return data

def plot_avg_capacity(data, start_datetime=None, end_datetime=None, time_binsize="7D", quantity="capacity"):
    
    if start_datetime != None:
        data = data[start_datetime:]
    if end_datetime != None:
        data = data[:end_datetime]

    avg_per_timedelta = pd.Series(index=data.index, data=np.array(data.capacity)).resample(time_binsize).mean()
    avg_per_timedelta_df = pd.DataFrame(avg_per_timedelta, columns=[quantity])
    avg_per_timedelta_df.plot()
    plt.show()

def plot_capacity_matrix(data, figsize=(15,15), y_axis="weekday", x_axis="hour", quantity="capacity", start_datetime=None, end_datetime=None):
    
    time_intervalls = {"hour":"H", "day":"D", "min":"T", "month":"M", "week":"W", "weekday": "D"}
    
    mean_cap = []
    yticklabels = None

    if start_datetime != None:
        data = data[start_datetime:]
    if end_datetime != None:
        data = data[:end_datetime]

    binned_cap = pd.Series(index=data.index, data=np.array(data[quantity])).resample(time_intervalls[x_axis]).mean()
    if "weekday" in y_axis.lower():
        for i in range(7):
            weekday_data = binned_cap[binned_cap.index.weekday == i]
            weekday_cap = weekday_data.groupby(weekday_data.index.time).mean()
            padding = pd.Series([np.nan]*(24-len(weekday_cap)), dtype=float)
            weekday_cap = pd.concat([weekday_cap, padding])
            mean_cap.append(weekday_cap)

    if "week" == y_axis.lower():
        for i in range(52):
            week_data = binned_cap[binned_cap.index.isocalendar().week == i]
            weekly_cap = week_data.groupby(week_data.index.time).mean()
            padding = pd.Series([np.nan]*(24-len(weekly_cap)), dtype=float)
            weekly_cap = pd.concat([weekly_cap, padding])

            if "weekday" in x_axis.lower():
                weekly_cap = week_data.groupby(week_data.index.weekday).mean()
                padding = pd.Series([np.nan]*(7-len(weekly_cap)), dtype=float)
                weekly_cap = pd.concat([weekly_cap, padding])
            if "day" == x_axis.lower():
                weekly_cap = week_data.groupby(week_data.index.day).mean()
                padding = pd.Series([np.nan]*(31-len(weekly_cap)), dtype=float)
                weekly_cap = pd.concat([weekly_cap, padding])
            
            mean_cap.append(weekly_cap)

    if "month" in y_axis.lower():
        for i in range(12):
            month_data = binned_cap[binned_cap.index.month == i]
            monthly_cap = month_data.groupby(month_data.index.time).mean()
            padding = pd.Series([np.nan]*(24-len(monthly_cap)), dtype=float)
            monthly_cap = pd.concat([monthly_cap, padding])

            if "weekday" in x_axis.lower():
                monthly_cap = month_data.groupby(month_data.index.weekday).mean()
                padding = pd.Series([np.nan]*(7-len(monthly_cap)), dtype=float)
                monthly_cap = pd.concat([monthly_cap, padding])

            if "day" == x_axis.lower():
                monthly_cap = month_data.groupby(month_data.index.day).mean()
                padding = pd.Series([np.nan]*(31-len(monthly_cap)), dtype=float)
                monthly_cap = pd.concat([monthly_cap, padding])
            
            if "month" in x_axis.lower():
                monthly_cap = month_data.groupby(month_data.index.month).mean()
                padding = pd.Series([np.nan]*(12-len(monthly_cap)), dtype=float)
                monthly_cap = pd.concat([monthly_cap, padding])
            
            if "week" == x_axis.lower():
                monthly_cap = month_data.groupby(month_data.index.isocalendar().week).mean()
                padding = pd.Series([np.nan]*(5-len(monthly_cap)), dtype=float)
                monthly_cap = pd.concat([monthly_cap, padding])

            mean_cap.append(monthly_cap)

    mean_cap = np.vstack(mean_cap)
    
    plt.figure(figsize=figsize)
    ax = plt.gca()
    
    im = ax.imshow(mean_cap, cmap="jet", vmin=0, vmax=100)

    if "hour" in x_axis.lower():
        keys = list(range(24))
        timestamps = ["0{}:00".format(i) for i in keys if i < 10] + ["{}:00".format(i) for i in keys if i >= 10]
        hours_dct = dict(zip(keys, timestamps))
        hours2time = [hours_dct.get(t, ax.get_xticks()[i]) for i,t in enumerate(ax.get_xticks())]
        ax.set_xticklabels(hours2time)

    if "month" in y_axis.lower():
        keys = list(range(12))
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        hours_dct = dict(zip(keys, months))
        months2time = [hours_dct.get(t, ax.get_yticks()[i]) for i,t in enumerate(ax.get_yticks())]
        ax.set_yticklabels(months2time)

    if "month" in x_axis.lower():
        keys = list(range(12))
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        hours_dct = dict(zip(keys, months))
        months2time = [hours_dct.get(t, ax.get_xticks()[i]) for i,t in enumerate(ax.get_xticks())]
        ax.set_xticklabels(months2time)

    if "weekday" in y_axis.lower():
        weekdays_dict = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
        yticklabels = [weekdays_dict.get(t, ax.get_yticks()[i]) for i,t in enumerate(ax.get_yticks())]
        ax.set_yticklabels(yticklabels)
    
    if "weekday" in x_axis.lower():
        weekdays_dict = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
        xticklabels = [weekdays_dict.get(t, ax.get_xticks()[i]) for i,t in enumerate(ax.get_xticks())]
        ax.set_xticklabels(xticklabels)
    
    ax.set_xlabel("time")

    plt.title("Capacity of the B12 on different {}s and {}s.".format(y_axis, x_axis))
    plt.tight_layout()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.20)
    plt.colorbar(im, cax=cax)
    plt.show()

