import pandas as pd
import numpy as np

from hawkeslib.model.mv_exp import MultivariateExpHawkesProcess as MVHP
from scipy.stats import gamma, expon, norm


'''
Plan for this file:
Goal: Model via statistical modeling (Multivariate Hawkes Process) the different orders arrival. 
Type of orders are:
        1: Submission of a new limit order
        2: Cancellation (Partial deletion 
            of a limit order)
        3: Deletion (Total deletion of a limit order)
        4: Execution of a visible limit order
        
order_data = from src.transform import OrdersDataset that contains the following columns:
    'Time', 'Type', 'OrderID', 'Size', 'Price', 'Direction', 'Ask Price 1', 'Ask Size 1', 'Bid Price 1', 'Bid Size 1',
    'Ask Price 2', 'Ask Size 2', 'Bid Price 2', 'Bid Size 2', 'Ask Price 3', 'Ask Size 3', 'Bid Price 3', 'Bid Size 3',
    'Ask Price 4', 'Ask Size 4', 'Bid Price 4', 'Bid Size 4', 'Ask Price 5', 'Ask Size 5', 'Bid Price 5', 'Bid Size 5',
    'Ask Price 6', 'Ask Size 6', 'Bid Price 6', 'Bid Size 6', 'Ask Price 7', 'Ask Size 7', 'Bid Price 7', 'Bid Size 7',
    'Ask Price 8', 'Ask Size 8', 'Bid Price 8', 'Bid Size 8', 'Ask Price 9', 'Ask Size 9', 'Bid Price 9', 'Bid Size 9',
    'Ask Price 10', 'Ask Size 10', 'Bid Price 10', 'Bid Size 10', 'Mid Price', 'Spread'

Plan:
1) Create ArrivalProcessModel containing the following methods:
    - __init__(self, order_data)
    - fit_multivariate_hawkes_process(self) --> CHECK: https://github.com/canerturkmen/hawkeslib/blob/master/examples/MV%20Exp%20Example.ipynb
    - extract_order_arrival_times(self)
2) Create OrderParameterModel which will handle fitting the ditributions to order parameters: price = Gaussian, size = TBD
    - __init__(self, order_data)
    - fit_price_distribution(self)
    - fit_size_distribution(self)
    - extract_price_deviation(self)
    - extract_size_deviation(self)
'''


class ArrivalProcessModel:
    """
    A class to model the arrival of different types of orders using a Multivariate Hawkes Process.

    Attributes:
    -----------
    order_data : DataFrame
        The order data containing order arrival information.
    hawkes_model : MultivariateExpHawkesProcess
        The fitted multivariate Hawkes process model.
    """

    def __init__(self, order_data):
        """
        Initializes the ArrivalProcessModel with order data.

        Parameters:
        -----------
        order_data : DataFrame
            The order data to be used for fitting the Hawkes process.
        """
        self.order_data = order_data
        self.arrival_times = self.extract_order_arrival_times()
        self.hawkes_model = None

    def fit_multivariate_hawkes_process(self):
        """
        Fits a Multivariate Hawkes Process to the order arrival data.

        The Hawkes process models the self-exciting nature of order arrivals, where the arrival of one order 
        can increase the likelihood of subsequent orders arriving soon after.
        """

    def extract_order_arrival_times(self):
        """
        Extracts arrival times for each type of order from the order data.

        Returns:
        --------
        list of lists
            A list containing lists of arrival times for each order type.
        """


class OrderParameterModel:
    """
    A class to model the parameters of orders, such as price and size, using statistical distributions.

    Attributes:
    -----------
    order_data : DataFrame
        The order data containing information on order prices and sizes.
    price_distribution : tuple
        Parameters of the fitted price distribution.
    size_distribution : tuple
        Parameters of the fitted size distribution.
    """

    def __init__(self, order_data):
        """
        Initializes the OrderParameterModel with order data.

        Parameters:
        -----------
        order_data : DataFrame
            The order data to be used for fitting the parameter distributions.
        """
        self.order_data = order_data
        self.price_distribution = None
        self.size_distribution = None

    def fit_price_distribution(self):
        """
        Fits a statistical distribution to the price deviations of orders.

        The price deviation is calculated as the difference between the order price and the mid-price.
        A Gamma distribution is fitted to the price deviations.
        """
        # Extract price deviations from mid-price

    def fit_size_distribution(self):
        """
        Fits a statistical distribution to the sizes of orders.

        An Exponential distribution is fitted to the order sizes.
        """

    def extract_price_deviation(self):
        """
        Extracts the price deviation from the mid-price for each order.

        The price deviation is positive if it's a buy order and negative if it's a sell order.

        Returns:
        --------
        numpy.ndarray
            An array of price deviations.
        """

    def extract_size_deviation(self):
        """
        Extracts the sizes of the orders.

        Returns:
        --------
        numpy.ndarray
            An array of order sizes.
        """
