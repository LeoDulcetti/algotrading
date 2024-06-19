from datetime import datetime, timedelta
import pandas as pd
from matching_engine import LOBMatching  # Ensure this is the correct class name and import path
from lob_order import *

def validate_data(messages, orderbook):
    # Check if essential columns are present in the dataframes
    required_message_columns = {'datetime', 'type', 'price', 'size'}
    required_orderbook_columns = {'datetime', 'bid_prices', 'ask_prices', 'bid_volumes', 'ask_volumes'}

    if not required_message_columns.issubset(messages.columns):
        raise ValueError("Messages dataframe is missing one or more required columns.")
    if not required_orderbook_columns.issubset(orderbook.columns):
        raise ValueError("Orderbook dataframe is missing one or more required columns.")

def runBacktest(
    messages,
    orderbook,
    algos,
    start,
    end,
    time_res,
    opening_time='09:30:00',
    closing_time='16:00:00',
    tick_size=0.01,
):
    """
    Simulates a trading session using historical LOB data and various trading algorithms.
    """
    # Validate input data
    validate_data(messages, orderbook)

    # Convert string times to datetime.time objects if necessary
    opening_time = pd.to_datetime(opening_time).time() if isinstance(opening_time, str) else opening_time
    closing_time = pd.to_datetime(closing_time).time() if isinstance(closing_time, str) else closing_time

    # Initialize the LOB with settings
    lob = LOBMatching(
        opening_time=str(opening_time),
        closing_time=str(closing_time),
        tick_size=tick_size,
        time_resolution=time_res
    )

    # Subscribe each algorithm to the LOB
    for algo in algos:
        lob.subscribeToLOB(algo)

    # Initialize variables for the simulation period
    current_time = datetime.combine(start, opening_time)
    end_time = datetime.combine(end, closing_time)
    simulated_days = []

    # Simulation loop from start to end date
    while current_time <= end_time:
        # Reset LOB for the start of each day
        lob.resetForNewDay()

        # Filter messages and order book data for the current day
        daily_messages = messages[messages['datetime'].dt.floor('D') == current_time.date()]
        daily_orderbook = orderbook[orderbook['datetime'].dt.floor('D') == current_time.date()]

        # Initialize the day in the LOB with the first snapshot of orderbook data
        lob.initializeWithSnapshot(daily_orderbook.iloc[0])

        # Process messages for the current day
        for _, message in daily_messages.iterrows():
            lob.processMessage(message)

        # Execute the end-of-day operations
        daily_results = lob.captureEndOfDayState()
        simulated_days.append({
            'date': current_time.date(),
            'results': daily_results
        })

        # Move to the next day
        current_time += timedelta(days=1)
        current_time = datetime.combine(current_time.date(), opening_time)  # Reset time to opening

    return simulated_days