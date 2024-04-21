from ..lob.lob_data import LOBupdate
from ..lob.matching_engine import LOBmatching

class TradingAlgo:
    """
    A trading algorithm class that interacts with a market or limit order book (LOB) system.

    This class represents a trading strategy and includes methods to handle different types of data and events
    from the market such as market updates, order events, and time-based actions.

    Attributes:
        market (LOBmatching): An instance of the LOBmatching class to which this algorithm subscribes for updates.
    """

    def __init__(self, market):
        """
        Initializes the trading algorithm with a reference to a market.

        Parameters:
            market (LOBmatching): The market system (typically an instance of LOBmatching) to which the algorithm will subscribe.
        """
        self.market = market
        market.subscribeToLOB(self)  

    def onData(self, data):
        """
        Handles incoming market data.

        This method is called when new data is received from the market. Subclasses should implement this
        method to specify how the algorithm reacts to market data.

        Parameters:
            data (LOBupdate): The data received from the market, typically including bids, asks, and trades.
        """
        pass

    def onOrderEvent(self, time, id, size, price):
        """
        Handles events related to orders made by the algorithm.

        This method is called when there is an update about an order placed by the algorithm such as
        execution or cancellation.

        Parameters:
            time (datetime): The timestamp of the event.
            id (int): The identifier of the order.
            size (float): The size of the order affected.
            price (float): The price at which the order was placed.
        """
        pass

    def onTime(self, time):
        """
        Performs actions based on a time trigger.

        This method is called regularly and can be used to perform actions at regular intervals, such as
        checking for time-specific market conditions or managing timed orders.

        Parameters:
            time (datetime): The current time at which the method is triggered.
        """
        pass

