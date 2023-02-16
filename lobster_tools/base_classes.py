"""Main module."""
############################
############################
# import modules
import logging
import os
import glob
import shutil
import re
from enum import Enum, auto
from typing import List, Tuple, Union
from dataclasses import dataclass, field
import pprint as pp
from tqdm import tqdm
import numpy as np
import pandas as pd
import time
import datetime
import pickle
import functools
from collections import Counter
import matplotlib.pyplot as plt

############################
############################
# set logging
logger = logging.getLogger("lobster")
logger.handlers = []
logger.setLevel(logging.DEBUG)

consoleHandler = logging.StreamHandler()
consoleHandler.setLevel(logging.DEBUG)

fileHandler = logging.FileHandler("log.log", mode="a")
fileHandler.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
consoleHandler.setFormatter(formatter)
fileHandler.setFormatter(formatter)

logger.handlers = []
logger.addHandler(consoleHandler)
logger.addHandler(fileHandler)

############################
############################
# enum classes


class Event(Enum):
    UNKNOWN = 0
    SUBMISSION = 1
    CANCELLATION = 2
    DELETION = 3
    EXECUTION = 4
    HIDDEN_EXECUTION = 5
    CROSS_TRADE = 6
    TRADING_HALT = 7
    OTHER = 8
    RESUME_QUOTE = 9
    TRADING_RESUME = 10


@dataclass
class EventGroup:
    EXECUTIONS = [Event.EXECUTION, Event.HIDDEN_EXECUTION]
    HALTS = [Event.TRADING_HALT, Event.RESUME_QUOTE, Event.TRADING_RESUME]
    CANCELLATIONS = [Event.CANCELLATION, Event.DELETION]


class Direction(Enum):
    BUY = 1
    SELL = -1
    NONE = 0


############################
############################
@dataclass
class Data:
    directory_path: str = "../data/raw"
    ticker: str = "AIG"
    date_range: Union[str, Tuple[str, str]] = None
    levels: int = None
    nrows: int = None

    def __post_init__(self) -> None:
        # ticker path
        tickers = glob.glob(f"{self.directory_path}/*")
        ticker_path = [t for t in tickers if self.ticker in t]
        assert len(ticker_path) == 1
        self.ticker_path = ticker_path[0]

        # levels
        if not self.levels:
            self.levels = int(self.ticker_path.split("_")[-1])
            assert self.levels >= 1

        # infer date range from ticker folder name
        if not self.date_range:
            self.date_range = tuple(
                re.findall(pattern="\d\d\d\d-\d\d-\d\d",
                           string=self.ticker_path)
            )
            assert len(self.date_range) == 2

        # book and message paths
        tickers = glob.glob(f"{self.ticker_path}/*")
        tickers_end = list(map(os.path.basename, tickers))

        if isinstance(self.date_range, tuple):
            # get all dates in folder
            dates = set(
                [
                    re.findall(pattern="\d\d\d\d-\d\d-\d\d", string=file)[0]
                    for file in tickers_end
                ]
            )
            # filter for dates within specified range
            dates = sorted(
                list(
                    filter(
                        lambda date: self.date_range[0] <= date <= self.date_range[1],
                        dates,
                    )
                )
            )

            self.dates = dates
            self.date_range = (min(self.dates), max(self.dates))

        elif isinstance(self.date_range, str):
            self.dates, self.date_range = [self.date_range], (
                self.date_range,
                self.date_range,
            )

        # messages and book filepath dictionaries
        def _create_date_to_path_dict(keyword: str) -> dict:
            filter_keyword_tickers = list(
                filter(lambda x: keyword in x, tickers_end))
            date_path_dict = {}
            for date in self.dates:
                filter_date_tickers = list(
                    filter(lambda x: date in x, filter_keyword_tickers)
                )
                assert len(filter_date_tickers) == 1
                date_path_dict[date] = os.path.join(
                    self.ticker_path, filter_date_tickers[0]
                )
            return date_path_dict

        self.book_paths = _create_date_to_path_dict("book")
        self.messages_paths = _create_date_to_path_dict("message")

        # set pkl filename
        self.pkl_filename = (
            f"{self.ticker}_{self.date_range[0]}_{self.date_range[1]}_{self.levels}.pkl"
        )


# get tickers list from data folder
def tickers_from_folder(folder="data"):
    def _get_uppercase(string):
        return "".join(list(filter(lambda x: x.isupper(), string)))

    return list(map(_get_uppercase, glob.glob(f"{folder}/*")))


class DatetimeDataFrame(pd.DataFrame):
    "pd.DataFrame subclass with methods operating on datetime index column."

    @property
    def _constructor(self):
        return DatetimeDataFrame

    def date_filter(self, date):
        return self[self.index.date == pd.to_datetime(date).date()]

    def _memory_usage_in_mb(self):
        return self.memory_usage(index=True, deep=True).sum() / 2**20

    def add_ticker(self, ticker):
        return self.assign(ticker=ticker).astype(
            dtype={
                "ticker": pd.CategoricalDtype(
                    categories=tickers_from_folder(folder="data")
                )
            }
        )

    def _add_ticker_column(self, ticker):
        self.insert(0, "ticker", ticker)
        self.ticker = self.ticker.astype("category")


class Messages(DatetimeDataFrame):
    "Limit order book: messages data class."

    @property
    def _constructor(self):
        return Messages

    def filter_(self, event=None, direction=None):
        df = self
        if direction:
            df = df[df.direction.eq(direction)]
        if event:
            if isinstance(event, list):
                df = df[df.event.isin(event)]
            elif isinstance(event, Event):
                df = df[df.event.eq(event)]
        return df

    def intraday_plot(self, y, date):
        df = self[self.index.date == date]
        df["time"] = df.index.time
        df.plot(x="time", y=y, title="Plot Title", rot=45)
        plt.show()

# book class
class Book(DatetimeDataFrame):
    "Limit order book data: book data class."

    @property
    def _constructor(self):
        return Book

    def intraday_plot(self):
        pass

    def first_n_levels(self, n: int):
        return self.iloc[:, : 4 * n]


@dataclass
class Lobster:
    "Lobster data class for a single symbol of Lobster data."
    data: Data = Data()

    def __post_init__(self):
        dfs = []
        for date, filepath in self.data.messages_paths.items():
            # load messages
            df = pd.read_csv(
                filepath,
                header=None,
                nrows=self.data.nrows,
                usecols=list(range(6)),
                names=["time", "event", "order_id",
                       "size", "price", "direction"],
                index_col=False,
                dtype={
                    "time": "float64",
                    "event": "int8",
                    "price": "int64",
                    "direction": "int8",
                    "order_id": pd.Int64Dtype(),
                    "size": pd.Int64Dtype(),
                },
            )

            # set index as datetime
            df["datetime"] = pd.to_datetime(date, format="%Y-%m-%d") + df.time.apply(
                lambda x: pd.to_timedelta(x, unit="s")
            )
            df.set_index("datetime", drop=True, inplace=True)
            df.drop(columns="time", inplace=True)
            dfs.append(df)
        df = pd.concat(dfs)

        # convert integer events to Enums
        df.event = df.event.apply(lambda x: Event(x))

        # convert integer directions to enums
        df.loc[df.event.ne(Event.TRADING_HALT), "direction"] = df.loc[
            df.event.ne(Event.TRADING_HALT), "direction"
        ].apply(lambda x: Direction(x))

        assert df.loc[df.event.eq(Event.TRADING_HALT),
                      "direction"].eq(-1).all()
        df.loc[df.event.eq(Event.TRADING_HALT), "direction"] = Direction.NONE

        # process trading halts
        def trading_halt_type(price):
            return {
                -1: Event.TRADING_HALT,
                0: Event.RESUME_QUOTE,
                1: Event.TRADING_RESUME,
            }[price]

        df.loc[df.event.eq(Event.TRADING_HALT), "event"] = df.loc[
            df.event.eq(Event.TRADING_HALT), "price"
        ].apply(lambda x: trading_halt_type(x))

        df.loc[df.event.isin(EventGroup.HALTS), ["order_id", "size", "price"], ] = [
            pd.NA,
            pd.NA,
            np.NaN,
        ]

        # set price in dollars
        df.price = df.price.apply(lambda x: x / 10_000)

        df = df.astype(
            dtype={
                "event": pd.CategoricalDtype(categories=list(Event)),
                "direction": pd.CategoricalDtype(categories=list(Direction)),
            }
        )
        self.messages = Messages(df)

        # load ordebook
        col_names = []
        for level in range(1, self.data.levels + 1):
            for col_type in ["ask_price", "ask_size", "bid_price", "bid_size"]:
                col_name = f"{col_type}_{level}"
                col_names.append(col_name)

        col_dtypes = {
            col_name: pd.Int64Dtype() if ("size" in col_name) else "float"
            for col_name in col_names
        }

        dfs = []
        for filename in self.data.book_paths.values():
            df = pd.read_csv(
                filename,
                header=None,
                nrows=self.data.nrows,
                usecols=list(range(4 * self.data.levels)),
                names=col_names,
                dtype=col_dtypes,
                na_values=[-9999999999, 9999999999, 0],
            )

            dfs.append(df)
        df = pd.concat(dfs)
        df.set_index(self.messages.index, inplace=True, drop=True)


        price_cols = df.columns.str.contains("price")
        df.loc[:, price_cols] = df.loc[:, price_cols].apply(
            lambda x: x / 10_000)

        self.book = Book(df)

        # memory usage attributes
        memory_usage = {
            "messages": self.messages._memory_usage_in_mb(),
            "book": self.messages._memory_usage_in_mb(),
        }
        memory_usage["total"] = memory_usage["messages"] + memory_usage["book"]
        memory_usage = {key: f"{val:.1f} MB" for key,
                        val in memory_usage.items()}
        self.memory_usage = memory_usage

    def __repr__(self) -> None:
        return f"Lobster data for ticker: {self.data.ticker} for date range: {self.data.date_range[0]} to {self.data.date_range[1]}."

    # reload Book and Messages classes
    def reload_classes(self) -> None:
        self.book = Book(self.book)
        self.messages = Messages(self.messages)

    def add_ticker(self) -> None:
        ticker = self.data.ticker
        self.book = self.book.add_ticker(ticker)
        self.messages = self.messages.add_ticker(ticker)

    def add_ticker_column(self) -> None:
        self.book._add_ticker_column(self.data.ticker)
        self.messages._add_ticker_column(self.data.ticker)


def myFunc(x):
    return x+1

ALMOST_PI = 3.14