from lob_order import *

class LOBmatching:
    """
    Class for matching orders in a Limit Order Book (LOB).
    This class provides functionality for subscribing an algorithm to the LOB,
    sending orders to the LOB, and running the matching process.
    """
    
    def __init__(self):
        """Initializes the LOB matching system."""
        pass

    def subscribeToLOB(self, algo):
        """
        Subscribes an algorithm to the LOB system to receive updates and interact with the order book.
        Parameters:
            algo (Algorithm): The trading algorithm that needs to interact with the LOB.
        """
        pass

    def sendOrder(self, order):
        """
        Sends an order to the LOB for processing.
        Parameters:
            order (Order): The order object to be processed by the LOB.
        """
        pass

    def run(self):
        """
        Runs the matching engine. This method should be called to start processing orders and matching them according to the LOB rules.
        """
        pass



    
