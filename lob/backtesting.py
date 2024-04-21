from matching_engine import LOBmatching

def runBacktest(messages, orderbook, algos, start, end, time_res, opening_time, closing_time, tick_size):
    """
    Runs a backtest simulation on a set of trading algorithms using historical market data.

    Parameters:
        messages (dataframe): messages dataset formatted as in LOBSTER
        orderbook (dataframe): orderbook dataset formatted as in LOBSTER
        algos (list): A list of trading algorithms that will interact with the orderbook. Each algorithm should be capable of making decisions based on orderbook updates.
        start (datetime): The start datetime of the backtest period.
        end (datetime): The end datetime of the backtest period.
        time_res (timedelta): The time resolution at which the backtest should process messages. This determines how often the orderbook is updated.
        opening_time (time): The opening time of the market each trading day.
        closing_time (time): The closing time of the market each trading day.
        tick_size (float): The minimum price movement of the market, defining the granularity of price levels in the orderbook.

    Returns:
        orders_data: returns the orders data of the trading sessions simulated 

    Example:
        # Example of how you might call runBacktest with hypothetical parameters
        runBacktest(messages=my_messages, orderbook=my_orderbook, algos=my_algos,
                    start=datetime(2022, 1, 1), end=datetime(2022, 1, 31),
                    time_res=timedelta(minutes=1), opening_time=time(9, 0),
                    closing_time=time(17, 0), tick_size=0.01)
    """
    pass
