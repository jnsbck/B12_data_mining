# data handling
import pandas as pd
import numpy as np
import re
import os

# webscraping
from bs4 import BeautifulSoup
import requests  # fetches html content of a website

# for catching timeout for website response
from urllib.request import HTTPError
from urllib.request import urlopen
from urllib.request import URLError

# plotting
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt

# other
from time import sleep  # for sleep function
from datetime import datetime, date, time, timedelta

# types
from pandas import DataFrame
from bs4 import BeautifulSoup
from typing import Tuple, List, Optional, Callable
from numpy import ndarray


def prune_data(
    data: DataFrame, outside_of: Optional[Tuple[int]] = None, buffer=15
) -> DataFrame:
    """Gets rid of data aquired while the B12 was not open.

    Removes datapoints outside of a specified time intervall, or if
    nothing is specified, the opening times from the B12 website are used.

    This function can be used to free up disk / memory space, by overwriting
    the existing logfiles, that contain datapoints, when the B12 is closed.

    Args:
        data: Occupancy data.
        outside_of: Outside of this intervall data will be disregarged.
            If nothing is specified, the opening times are used.
        buffer: Allows for some time before opening and after closing to be considered.

    Returns:
        data with uneccessary data points removed.
    """
    if outside_of != None:
        after_opening = np.array(data.index.time > time(outside_of[0], 0))
        b4_closing = np.array(data.index.time < time(outside_of[1], 0))
        is_open = np.logical_and(after_opening, b4_closing)

        return data[is_open]

    else:
        opening_times = {
            "Mon": ["09:30", "23:00"],
            "Tue": ["09:30", "23:00"],
            "Wed": ["08:30", "23:00"],
            "Thu": ["12:30", "23:00"],
            "Fri": ["09:30", "23:00"],
            "Sat": ["10:00", "22:00"],
            "Sun": ["10:00", "21:30"],
        }

        def map_openinghours2time(x, idx):
            return datetime.strptime(opening_times[x][idx], "%H:%M") + (
                2 * idx - 1
            ) * timedelta(minutes=buffer)

        def day2time_mapper(x):
            return [
                map_openinghours2time(x, 0).time(),
                map_openinghours2time(x, 1).time(),
            ]

        weekday = np.array(list(opening_times.keys()))
        weekdays = weekday[data.index.weekday]
        open4theday = np.array(list(map(day2time_mapper, weekdays)))

        after_opening = data.index.time > open4theday[:, 0]
        b4_closing = data.index.time < open4theday[:, 1]
        is_open = np.logical_and(after_opening, b4_closing)

        return data[is_open]


def import_logged_data(loc: str = "./log.csv") -> DataFrame:
    """Import logged capacity data for B12 as DataFrame.

    Converts column with datetime str into datetime index.

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
    update_interval: Optional[int] = 5,
    save2file: Optional[str] = "./log.csv",
    run_every: int = 3,
    update_func: Optional[Callable] = None,
    *args
):
    """Starts logging of the capacity of the B12 Bouldering Hall in Tuebingen.

    Requests current capacity from B12 Website in specified timeintervalls 
    and keeps track of it. Catches Errors and tests for proper connection to
    the internet during operation.

    Args:
        update_interval: Number of minutes between subsequent requests.
        save2file: Location of the file, which the fetched data is saved to.
        run_every: How many intervals until update function is called.
        update_func: Can be used to execute whatever within loop. Can be used to
        update plots for example.
        *args: Args for the update_func.
    """
    # seed logfile with header if logfile is empty.
    if not os.path.exists(save2file):
        timestamp = datetime.now()
        timestamp_str = datetime.strftime(timestamp, "%H:%M, %m/%d/%Y")
        header = pd.DataFrame(columns=["datetime", "free spots", "capacity"])
        header.to_csv(save2file, index=False, sep=";", mode="a")
        msg = "[Success] {} - New logfile was created @ {}".format(
            timestamp_str, save2file
        )
        print(msg)

    # Check occupancy in regular intervals, while catching network errors.
    counter = 0
    while True:
        timestamp = datetime.now()
        timestamp_str = datetime.strftime(timestamp, "%H:%M, %m/%d/%Y")
        if online():
            try:
                abs_cap, rel_cap = get_current_capacity()
                update = pd.DataFrame([[timestamp_str, abs_cap, rel_cap]])
                update.to_csv(save2file, index=False,
                              sep=";", header=False, mode="a")

                msg = "[Success] {} - Logfile was updated. Free Spots = {}, Capacity = {}%.".format(
                    timestamp_str, abs_cap, rel_cap
                )
                print(msg)

            except HTTPError:
                msg = "[Failure] {} - Encountered some unknown error. Retrying in {} mins".format(
                    timestamp_str, update_interval
                )
                print(msg)

            except URLError:
                msg = "[Failure] {} - Encountered some unknown error. Retrying in {} mins".format(
                    timestamp_str, update_interval
                )
                print(msg)

        else:
            msg = "[Failure] {} - Internet is not reachable. Retrying in {} mins.".format(
                timestamp_str, update_interval
            )
            print(msg)

        if update_func != None:
            counter += 1
            if counter == run_every:
                update_func(*args)
                counter = 0

        sleep(60*update_interval)


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
     Chrome/41.0.2228.0 Safari/537.36"
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
        df = pd.DataFrame(
            data.T, columns=["datetime", "time", "weekday", "capacity"])
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
        time_binsize: XT, XH, XD, XW specify the time in minutes, hours, days or 
            weeks, where X can be a number, i.e. 7D.
        quantity: The quantity to plot. Usually either capacity or free spots.
            Can however also be a custom quantity.
        figsize: Specifies figure dimensions.

    Returns:
        Axes object.
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
    ax = avg_per_timedelta_df.plot(figsize=figsize, grid=True)
    return ax


def plot_capacity_matrix(
    data: DataFrame,
    figsize: Tuple[int, int] = (15, 15),
    y_axis: str = "weekday",
    x_axis: str = "hour",
    x_increment: int = 1,
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
        x_increment: Increments the x timebins.
            Example x_axis = min, x_increment = 30 --> 30min increments.
        quantity: The quantity to plot. Usually either capacity or free spots.
            Can however also be a custom quantity.
        start_datetime: The starting timestamp can be specified. If none is 
            specified the first datapoint is chosen.
        end_datetime: The ending timestamp can be specified. If none is specified
            the last datapoint is chosen.

    Returns:
        Fig object.
        Axes object.
    """
    yticklabels = None
    if "weekday" in x_axis.lower():
        x_increment = 1

    mean_cap = capacity_matrix(
        data, y_axis, x_axis, x_increment, quantity, start_datetime, end_datetime
    )

    fig = plt.figure(figsize=figsize)
    ax = plt.gca()

    im = ax.imshow(mean_cap, cmap="jet", vmin=0, vmax=100, aspect="auto")
    ax.grid(True)

    if "hour" in x_axis.lower():
        keys = range(int(25/x_increment))
        timestamps = ["0{}:00".format(int(i*x_increment)) for i in keys if int(i*x_increment) < 10] + [
            "{}:00".format(int(i*x_increment)) for i in keys if int(i*x_increment) >= 10
        ]

        ax.locator_params(axis='x', nbins=len(keys))
        ax.set_xticks(keys)
        ax.set_xticklabels(timestamps)

    if "min" in x_axis.lower():
        keys = range(int(25*60/x_increment))
        timestamps = [
            "0{}:00".format(int(i*x_increment / 60)) for i in keys if int(i * x_increment / 60) < 10
        ] + ["{}:00".format(int(i*x_increment / 60)) for i in keys if int(i * x_increment / 60) >= 10]

        ax.locator_params(axis='x', nbins=len(keys[::6]))
        ax.set_xticks(keys[::6])
        ax.set_xticklabels(timestamps[::6])

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
        ax.set_yticks(ax.get_yticks().tolist())
        ax.set_yticklabels(months2time)
        ax.set_ylim(0, 11)

    if "month" in x_axis.lower():
        keys = list(range(int(12/x_increment)))
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
        lables = [months[int(x_increment*key)] for key in keys]
        ax.locator_params(axis='x', nbins=6)

        ax.locator_params(axis='x', nbins=len(keys))
        ax.set_xticks(keys)
        ax.set_xticklabels(lables)

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
        ax.set_yticks(ax.get_yticks().tolist())
        ax.set_yticklabels(yticklabels)
        ax.set_ylim(0, 6)

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
        ax.locator_params(axis='x', nbins=len(keys))
        ax.set_xticks(ax.get_xticks().tolist())
        ax.set_xticklabels(xticklabels)
        ax.set_xlim(0, 6)

    # x_incrementation does not work for days

    ax.set_xlabel("time")

    plt.title("Capacity of the B12 on different {}s and {}s.".format(y_axis, x_axis))
    plt.tight_layout()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.20)
    plt.colorbar(im, cax=cax)

    return fig, ax


def capacity_matrix(
    data: DataFrame,
    y_axis: str = "weekday",
    x_axis: str = "hour",
    x_increment: int = 1,
    quantity: str = "capacity",
    start_datetime: Optional[str] = None,
    end_datetime: Optional[str] = None,
) -> ndarray:
    """Computes the occupancy of the B12 for different time chunks.

    Depedning on the bin width chosen for the bins in y and x dimensions,
    the z dimension is then used to colourcode the resulting bins based on
    the capacity at this time on average. Data gets averaged in both x and y
    dims.

    Possible time-binsizes are: hour, day, min, month, week, weekday.

    Args:
        data: B12 usage data.
        y_axis: Size of timebins in y dimension.
        x_axis: Size of timebins in x dimension.
        x_increment: Increments the x timebins.
            Example x_axis = min, x_increment = 30 - -> 30min increments.
        quantity: The quantity to plot. Usually either capacity or free spots.
            Can however also be a custom quantity.
        start_datetime: The starting timestamp can be specified. If none is
            specified the first datapoint is chosen.
        end_datetime: The ending timestamp can be specified. If none is specified
            the last datapoint is chosen.

    Returns:
        Matrix containing mean of the occupancy per 2D timebin (i,j).
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

    if start_datetime != None:
        data = data[start_datetime:]
    if end_datetime != None:
        data = data[:end_datetime]

    binned_cap = (
        pd.Series(index=data.index, data=np.array(data[quantity]))
        .resample(str(x_increment) + time_intervalls[x_axis])
        .mean()
    )
    time_padding = pd.Series(
        np.nan,
        index=pd.date_range(
            start="01/01/2021",
            end="01/01/2022",
            freq=str(x_increment) + time_intervalls[x_axis],
        ),
    )

    if "weekday" in y_axis.lower():
        for i in range(7):
            occupancy = time_padding.groupby(time_padding.index.time).mean()
            weekday_data = binned_cap[binned_cap.index.weekday == i]
            weekday_cap = weekday_data.groupby(weekday_data.index.time).mean()
            occupancy[weekday_cap.index] = weekday_cap.values
            mean_cap.append(occupancy)

    if "week" == y_axis.lower():
        for i in range(52):
            occupancy = time_padding.groupby(time_padding.index.time).mean()
            week_data = binned_cap[binned_cap.index.isocalendar().week == i]
            weekly_cap = week_data.groupby(week_data.index.time).mean()
            occupancy[weekly_cap.index] = weekly_cap.values

            if "weekday" in x_axis.lower():
                occupancy = time_padding.groupby(
                    time_padding.index.weekday).mean()
                weekly_cap = week_data.groupby(week_data.index.weekday).mean()
                occupancy[weekly_cap.index] = weekly_cap.values
            if "day" == x_axis.lower():
                occupancy = time_padding.groupby(time_padding.index.day).mean()
                weekly_cap = week_data.groupby(week_data.index.day).mean()
                occupancy[weekly_cap.index] = weekly_cap.values

            mean_cap.append(occupancy)

    if "month" in y_axis.lower():
        for i in range(12):
            occupancy = time_padding.groupby(time_padding.index.time).mean()
            month_data = binned_cap[binned_cap.index.month == i]
            monthly_cap = month_data.groupby(month_data.index.time).mean()
            occupancy[monthly_cap.index] = monthly_cap.values

            if "weekday" in x_axis.lower():
                occupancy = time_padding.groupby(
                    time_padding.index.weekday).mean()
                monthly_cap = month_data.groupby(
                    month_data.index.weekday).mean()
                occupancy[monthly_cap.index] = monthly_cap.values

            if "day" == x_axis.lower():
                occupancy = time_padding.groupby(time_padding.index.day).mean()
                monthly_cap = month_data.groupby(month_data.index.day).mean()
                occupancy[monthly_cap.index] = monthly_cap.values

            if "month" in x_axis.lower():
                occupancy = time_padding.groupby(
                    time_padding.index.month).mean()
                monthly_cap = month_data.groupby(month_data.index.month).mean()
                occupancy[monthly_cap.index] = monthly_cap.values

            if "week" == x_axis.lower():
                occupancy = time_padding.groupby(
                    time_padding.index.week).mean()
                monthly_cap = month_data.groupby(
                    month_data.index.isocalendar().week
                ).mean()
                occupancy[monthly_cap.index] = monthly_cap.values

            mean_cap.append(occupancy)

    mean_cap = np.vstack(mean_cap)
    return mean_cap
