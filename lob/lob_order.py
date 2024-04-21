from enum import Enum

class OrderType(Enum):
    """Enum for different types of orders."""
    LIMIT = 1
    MARKET = 2
    PARTIAL_CANCEL = 3
    FULL_CANCEL = 4

class OrderStatus(Enum):
    """Enum for tracking the status of an order."""
    NEW = 1
    POSTED = 2
    EXECUTED = 3
    CANCELLED = 4

class OrderSide(Enum):
    """Enum for buy or sell orders"""
    BUY = 1
    SELL = -1

class Order:
    """Base class for different types of orders.
    Attributes:
        type (OrderType): The type of the order.
        status (OrderStatus): The current status of the order, initially set to NEW.
    """
    def __init__(self, type):
        self.type = type
        self.status = OrderStatus.NEW

class LimitOrder(Order):
    """Represents a limit order
    Attributes:
        side (OrderSide): The side of the order 
        size (float): The quantity of the asset to buy or sell.
        price (float): The price limit for the order.
    """
    def __init__(self, side, size, price):
        super().__init__(OrderType.LIMIT)
        self.side = side
        self.size = size
        self.price = price

class MarketOrder(Order):
    """Represents a market order.
    Attributes:
        side (OrderSide): The side of the order 
        size (float): The quantity of the asset to buy or sell.
    """
    def __init__(self, side, size):
        super().__init__(OrderType.MARKET)
        self.side = side
        self.size = size

class PartialCancel(Order):
    """Represents a request to partially cancel an order.
    Attributes:
        id (int): The identifier of the order to be partially canceled.
        size (float): The remaining size of the order after the partial cancellation.
    """
    def __init__(self, id, size):
        super().__init__(OrderType.PARTIAL_CANCEL)
        self.id = id
        self.size = size

class FullCancel(Order):
    """Represents a request to fully cancel an order.
    
    Attributes:
        id (int): The identifier of the order to be canceled.
    """
    def __init__(self, id):
        super().__init__(OrderType.FULL_CANCEL)
        self.id = id
