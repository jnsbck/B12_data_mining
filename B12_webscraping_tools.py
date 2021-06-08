import pandas as pd
import numpy as np
import re

from bs4 import BeautifulSoup
import requests  # fetches html content of a website

# for catching timeout for website response
from urllib.request import HTTPError
from urllib.request import urlopen
from urllib.request import URLError

from time import sleep  # for sleep function
from datetime import datetime, date, time, timedelta

from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import matplotlib.pyplot as plt

# types
from pandas import DataFrame
from bs4 import BeautifulSoup
from typing import Tuple, List, Optional
from numpy import ndarray


def import_logged_data(loc: str = "./log.csv") -> DataFrame:
    """Import looged capacity data for B12 as DataFrame.

    Args:
        loc: Location of logfile.

    Returns:
        data: DataFrame containing the data.
    """
    data = pd.read_csv(loc, sep=";")
    data = data.set_index("datetime")
    data.index = pd.to_datetime(data.index)
    data = data.sort_index()
    return data


def deploy_data_logger(
    update_interval: Optional[int] = 15, save2file: Optional[str] = "log.csv"
):
    """Starts logging of the capacity of the B12 Bouldering Hall in Tuebingen.

    Requests current capacity from B12 Website in specified timeintervalls 
    and keeps track of it. Catches Errors and tests for proper connection to
    the internet during operation.

    Args:
        update_interval: Number of minutes between subsequent requests.
        save2file: Location of the file, which the fetched data is saved to.
    """
    log = pd.DataFrame(columns=["datetime", "free spots", "capacity"])

    while True:
        timestamp = datetime.now()
        timestamp_str = datetime.strftime(timestamp, "%H:%M, %m/%d/%Y")
        if online():
            try:
                abs_cap, rel_cap = get_current_capacity()

                dct = {
                    "datetime": [timestamp_str],
                    "free spots": [abs_cap],
                    "capacity": [rel_cap],
                }
                update = pd.DataFrame(dct)

                log = log.append(update, ignore_index=True, sort=False)
                log.to_csv(save2file, index=False, sep=";")

                msg = "[Success] {} - Logfile was updated. \
                    Free Spots = {}, Capacity = {}%.".format(
                    timestamp_str, abs_cap, rel_cap
                )
                print(msg)

            except HTTPError:
                msg = "[Failure] {} - Encountered some unknown error. \
                    Retrying again in {} mins".format(
                    timestamp_str, update_interval
                )
                print(msg)

            except URLError:
                msg = "[Failure] {} - Encountered some unknown error. \
                    Retrying again in {} mins".format(
                    timestamp_str, update_interval
                )
                print(msg)

            sleep(60 * update_interval)

        else:
            msg = "[Failure] {} - Internet is not reachable. \
                Retrying in {} mins.".format(
                timestamp_str, update_interval
            )
            print(msg)

            sleep(60 * update_interval)


def webpage2soup(url: str, parser: Optional[str] = "html.parser") -> BeautifulSoup:
    """Fetches HTML content of a webpage and returns soup.

    Args:
        url: Url to fetch.
        parser: Tells BeatifulSoup which paser to use.

    Returns:
        soup: Parsed HTML tree.
    """

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 6.1)\
     AppleWebKit/537.36 (KHTML, like Gecko)\
     Chrome/41.0.2228.0 Safari/537.36",
    }
    res = requests.get(url, headers=headers)
    res.raise_for_status()

    soup = BeautifulSoup(res.text, parser)
    return soup


def online(url: Optional[str] = "http://google.com") -> bool:
    """Checks for a connection to the internet.

    Args:
        url: Which url to use as check.

    Returns:
        Online: Online status.
    """
    Online = False
    try:
        urlopen(url, timeout=1)
        Online = True
    except URLError as err:
        Online = False
    return Online


def get_current_capacity() -> Tuple[int, int]:
    """Makes B12 API query for the current capacity.

    Returns:
        num: Absolute number of available free spots.
        percentage: Relative number of available free spots.
    """
    js_query = "https://111.webclimber.de/de/trafficlight?\
    callback=WebclimberTrafficlight.insertTrafficlight&\
    key=184xNhv6RRU7H2gVg8QFyHCYxym8DKve&\
    hid=111&\
    container=trafficlightContainer&\
    type=&\
    area="

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


def str2datetime(string: str, fmt: str = "%H:%M") -> datetime or List[datetime]:
    """Translates timestamp from string to datetime object.

    Args:
        string: Timestamp or List of timestamps provided as string.
        fmt: Format string of the format that the provided timestamp has.

    Returns:
        Single or list of datetime objects.
    """
    if type(string) is str:
        return datetime.strptime(string, fmt)
    if type(string) is list:
        return [datetime.strptime(x, fmt) for x in string]


def simulate_scraper(
    year: int = 2021,
    month: int = 5,
    day: int = 23,
    hour: int = 10,
    minute: int = 0,
    update_every: int = 15,
    n_updates: int = 100000,
    return_as: str = "dataframe",
) -> DataFrame or ndarray:
    """Simulates B12 scraper.

    This function can be used to quickly generate
    a lot of scraper like fake data for testing
    and developing analysis tools around the data.

    Args:
        year: Starting year.
        month: Starting month.
        day: Starting day.
        hour: Starting hour.
        minute: Starting minute.
        update_every: Update interval in minutes.
        n_updates: Number of datapoints.
        return_as: Whether to return the data in the form of a DataFrame or array.

    Returns:
        Generated fake data in the form of an array or DataFrame.
    """
    weekdays_dict = {
        0: "Mon",
        1: "Tue",
        2: "Wed",
        3: "Thu",
        4: "Fri",
        5: "Sat",
        6: "Sun",
    }

    d = date(year, month, day)
    t = time(hour, minute)

    datetimes = np.array(
        [
            datetime.combine(d, t) + n * timedelta(minutes=update_every)
            for n in range(n_updates)
        ]
    )
    weekdays_str = np.array(
        [weekdays_dict[datetime.weekday()] for datetime in datetimes]
    )
    weekdays_numeric = np.array([datetime.weekday() for datetime in datetimes])
    times = np.array([datetime.time() for datetime in datetimes])

    hours = np.array([datetime.hour for datetime in datetimes])
    capacity = (
        30
        + 5 * (weekdays_numeric == 4)
        + 5 * (weekdays_numeric == 2)
        + 40 * np.sin(np.pi * (hours + 1) / 24)
        + 10 * np.random.randn(n_updates).astype("int64")
    )

    if "frame" in return_as.lower():
        data = np.vstack([datetimes, times, weekdays_str, capacity])
        df = pd.DataFrame(data.T, columns=["datetime", "time", "weekday", "capacity"])
        df = df.astype({"capacity": int}).set_index("datetime")
        return df
    if "array" in return_as.lower():
        data = np.vstack([datetimes, times, weekdays_numeric, capacity])
        return data


def plot_avg_capacity(
    data: DataFrame,
    start_datetime: Optional[str] = None,
    end_datetime: Optional[str] = None,
    time_binsize: str = "7D",
    quantity: Optional[str] = "capacity",
    figsize: Tuple[int, int] = (15, 10),
):
    """Plots the timecourse of the avg capacity in the B12.

    Depending on the interest, the relative or absolute capacity can be plotted.
    This can be done for a specified time intervall.

    The starting and end times need to be formated as 'HH:MM, mm/dd/yyyy'.

    Args:
        data: The data to be plotted.
        start_datetime: The starting timestamp can be specified. If none is 
            specified the first datapoint is chosen.
        end_datetime: The ending timestamp can be specified. If none is specified
            the last datapoint is chosen.
        time_binsize: XM, XH, XD, XW specify the time in minutes, hours, days or 
            weeks, where X can be a number, i.e. 7D.
        quantity: The quantity to plot. Usually either capacity or free spots.
            Can however also be a custom quantity.
        figsize: Specifies figure dimensions.
    """

    if start_datetime != None:
        data = data[start_datetime:]
    if end_datetime != None:
        data = data[:end_datetime]

    avg_per_timedelta = (
        pd.Series(index=data.index, data=np.array(data.capacity))
        .resample(time_binsize)
        .mean()
    )
    avg_per_timedelta_df = pd.DataFrame(avg_per_timedelta, columns=[quantity])
    avg_per_timedelta_df.plot(figsize=figsize)
    plt.show()


def plot_capacity_matrix(
    data: DataFrame,
    figsize: Tuple[int, int] = (15, 15),
    y_axis: str = "weekday",
    x_axis: str = "hour",
    quantity: str = "capacity",
    start_datetime: Optional[str] = None,
    end_datetime: Optional[str] = None,
):
    """Visualise the occupancy of the B12 in different time chunks.

    Depedning on the bin width chosen for the bins in y and x dimensions,
    the z dimension is then used to colourcode the resulting bins based on
    the capacity at this time on average. Data gets averaged in both x and y
    dims.

    Possible time-binsizes are: hour, day, min, month, week, weekday.

    Args:
        data: B12 usage data.
        figsize: Specifies figure dimensions.
        y_axis: Size of timebins in y dimension.
        x_axis: Size of timebins in x dimension.
        quantity: The quantity to plot. Usually either capacity or free spots.
            Can however also be a custom quantity.
        start_datetime: The starting timestamp can be specified. If none is 
            specified the first datapoint is chosen.
        end_datetime: The ending timestamp can be specified. If none is specified
            the last datapoint is chosen.
    """

    time_intervalls = {
        "hour": "H",
        "day": "D",
        "min": "T",
        "month": "M",
        "week": "W",
        "weekday": "D",
    }

    mean_cap = []
    yticklabels = None

    if start_datetime != None:
        data = data[start_datetime:]
    if end_datetime != None:
        data = data[:end_datetime]

    binned_cap = (
        pd.Series(index=data.index, data=np.array(data[quantity]))
        .resample(time_intervalls[x_axis])
        .mean()
    )
    if "weekday" in y_axis.lower():
        for i in range(7):
            weekday_data = binned_cap[binned_cap.index.weekday == i]
            weekday_cap = weekday_data.groupby(weekday_data.index.time).mean()
            padding = pd.Series([np.nan] * (24 - len(weekday_cap)), dtype=float)
            weekday_cap = pd.concat([weekday_cap, padding])
            mean_cap.append(weekday_cap)

    if "week" == y_axis.lower():
        for i in range(52):
            week_data = binned_cap[binned_cap.index.isocalendar().week == i]
            weekly_cap = week_data.groupby(week_data.index.time).mean()
            padding = pd.Series([np.nan] * (24 - len(weekly_cap)), dtype=float)
            weekly_cap = pd.concat([weekly_cap, padding])

            if "weekday" in x_axis.lower():
                weekly_cap = week_data.groupby(week_data.index.weekday).mean()
                padding = pd.Series([np.nan] * (7 - len(weekly_cap)), dtype=float)
                weekly_cap = pd.concat([weekly_cap, padding])
            if "day" == x_axis.lower():
                weekly_cap = week_data.groupby(week_data.index.day).mean()
                padding = pd.Series([np.nan] * (31 - len(weekly_cap)), dtype=float)
                weekly_cap = pd.concat([weekly_cap, padding])

            mean_cap.append(weekly_cap)

    if "month" in y_axis.lower():
        for i in range(12):
            month_data = binned_cap[binned_cap.index.month == i]
            monthly_cap = month_data.groupby(month_data.index.time).mean()
            padding = pd.Series([np.nan] * (24 - len(monthly_cap)), dtype=float)
            monthly_cap = pd.concat([monthly_cap, padding])

            if "weekday" in x_axis.lower():
                monthly_cap = month_data.groupby(month_data.index.weekday).mean()
                padding = pd.Series([np.nan] * (7 - len(monthly_cap)), dtype=float)
                monthly_cap = pd.concat([monthly_cap, padding])

            if "day" == x_axis.lower():
                monthly_cap = month_data.groupby(month_data.index.day).mean()
                padding = pd.Series([np.nan] * (31 - len(monthly_cap)), dtype=float)
                monthly_cap = pd.concat([monthly_cap, padding])

            if "month" in x_axis.lower():
                monthly_cap = month_data.groupby(month_data.index.month).mean()
                padding = pd.Series([np.nan] * (12 - len(monthly_cap)), dtype=float)
                monthly_cap = pd.concat([monthly_cap, padding])

            if "week" == x_axis.lower():
                monthly_cap = month_data.groupby(
                    month_data.index.isocalendar().week
                ).mean()
                padding = pd.Series([np.nan] * (5 - len(monthly_cap)), dtype=float)
                monthly_cap = pd.concat([monthly_cap, padding])

            mean_cap.append(monthly_cap)

    mean_cap = np.vstack(mean_cap)

    plt.figure(figsize=figsize)
    ax = plt.gca()

    im = ax.imshow(mean_cap, cmap="jet", vmin=0, vmax=100)

    if "hour" in x_axis.lower():
        keys = list(range(24))
        timestamps = ["0{}:00".format(i) for i in keys if i < 10] + [
            "{}:00".format(i) for i in keys if i >= 10
        ]
        hours_dct = dict(zip(keys, timestamps))
        hours2time = [
            hours_dct.get(t, ax.get_xticks()[i]) for i, t in enumerate(ax.get_xticks())
        ]
        ax.set_xticklabels(hours2time)

    if "month" in y_axis.lower():
        keys = list(range(12))
        months = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]
        hours_dct = dict(zip(keys, months))
        months2time = [
            hours_dct.get(t, ax.get_yticks()[i]) for i, t in enumerate(ax.get_yticks())
        ]
        ax.set_yticklabels(months2time)

    if "month" in x_axis.lower():
        keys = list(range(12))
        months = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]
        hours_dct = dict(zip(keys, months))
        months2time = [
            hours_dct.get(t, ax.get_xticks()[i]) for i, t in enumerate(ax.get_xticks())
        ]
        ax.set_xticklabels(months2time)

    if "weekday" in y_axis.lower():
        weekdays_dict = {
            0: "Mon",
            1: "Tue",
            2: "Wed",
            3: "Thu",
            4: "Fri",
            5: "Sat",
            6: "Sun",
        }
        yticklabels = [
            weekdays_dict.get(t, ax.get_yticks()[i])
            for i, t in enumerate(ax.get_yticks())
        ]
        ax.set_yticklabels(yticklabels)

    if "weekday" in x_axis.lower():
        weekdays_dict = {
            0: "Mon",
            1: "Tue",
            2: "Wed",
            3: "Thu",
            4: "Fri",
            5: "Sat",
            6: "Sun",
        }
        xticklabels = [
            weekdays_dict.get(t, ax.get_xticks()[i])
            for i, t in enumerate(ax.get_xticks())
        ]
        ax.set_xticklabels(xticklabels)

    ax.set_xlabel("time")

    plt.title("Capacity of the B12 on different {}s and {}s.".format(y_axis, x_axis))
    plt.tight_layout()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.20)
    plt.colorbar(im, cax=cax)
    plt.show()
