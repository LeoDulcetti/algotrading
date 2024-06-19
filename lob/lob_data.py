import datetime
from enum import Enum
import pandas as pd
from matching_engine import *
from lob_order import *


class LOBupdate:
    """
    Represents an update to the Limit Order Book (LOB).

    This class captures the state of the order book at a given time, including bid and ask prices,
    their corresponding volumes, and any trades that might have occurred.

    Attributes:
        time (datetime.datetime): The timestamp of the order book update.
        bid_prices (list[float]): A list of bid prices.
        bid_volumes (list[int]): A list of volumes corresponding to the bid prices.
        ask_prices (list[float]): A list of ask prices.
        ask_volumes (list[int]): A list of volumes corresponding to the ask prices.
        orders (list[Order]): The list of orders that produced the update.
    """

    def __init__(self, time, bid_prices, bid_volumes, ask_prices, ask_volumes, orders):
        self.time = time
        self.bid_prices = bid_prices
        self.bid_volumes = bid_volumes
        self.ask_prices = ask_prices
        self.ask_volumes = ask_volumes
        self.orders = orders

    def __str__(self):
        return (f"Update Time: {self.time}\n"
                f"Bid Prices: {self.bid_prices}\n"
                f"Bid Volumes: {self.bid_volumes}\n"
                f"Ask Prices: {self.ask_prices}\n"
                f"Ask Volumes: {self.ask_volumes}\n"
                f"Orders: {[str(order) for order in self.orders]}")
