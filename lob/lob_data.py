class LOBupdate:
    """
    Represents an update to the Limit Order Book (LOB).

    This class captures the state of the order book at a given time, including bid and ask prices, 
    their corresponding volumes, and any trades that might have occurred.

    Attributes:
        time (datetime): The timestamp of the order book update.
        bid_prices (list[float]): A list of bid prices.
        bid_volumes (list[int]): A list of volumes corresponding to the bid prices.
        ask_prices (list[float]): A list of ask prices.
        ask_volumes (list[int]): A list of volumes corresponding to the ask prices.
        orders (list[Order]): The list of orders that produced the update
    """

    def __init__(self, time, bid_prices, bid_volumes, ask_prices, ask_volumes, orders):
        self.time = time
        self.bid_prices = bid_prices
        self.bid_volumes = bid_volumes
        self.ask_prices = ask_prices
        self.ask_volumes = ask_volumes
        self.orders = orders
