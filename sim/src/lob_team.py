from enum import Enum
import pandas as pd
from market_sim2_xgb import SimulationRunner

# Enums and Order class as per your descriptions


class OrderType(Enum):
    LIMIT = 1
    MARKET = 2
    PARTIAL_CANCEL = 3
    FULL_CANCEL = 4


class OrderStatus(Enum):
    NEW = 1
    POSTED = 2
    EXECUTED = 3
    CANCELLED = 4


class OrderSide(Enum):
    BUY = 1
    SELL = -1


class Order:
    def __init__(self, order_type, side, price, volume):
        self.type = order_type
        self.side = side
        self.price = price
        self.volume = volume
        self.status = OrderStatus.NEW

# Matching engine and its methods


class LOBmatching:
    def __init__(self):
        self.orders = []

    def process_order(self, order):
        # Simulated processing (would be more complex in reality)
        print(
            f"Processing order: Type={order.type}, Side={order.side}, Price={order.price}, Volume={order.volume}")
        order.status = OrderStatus.EXECUTED
        self.orders.append(order)


# Initialize the LOB system
lob = LOBmatching()

# Load the CSV file
# Modify with the actual path to your CSV file
# csv_path = "../../data/logs/simulation_results.csv"


num_simulations = 2  # Specify the number of simulations to run
simulation_runner = SimulationRunner(num_simulations)
dataframe_sim = simulation_runner.run_simulations(concat=True)

# csv_data = pd.read_csv(csv_path)

# Process each order from the CSV data
for index, row in dataframe_sim.iterrows():
    order = Order(
        order_type=OrderType(row['OrderType']),
        side=OrderSide.BUY if row['Direction'] == 1 else OrderSide.SELL,
        price=row['Price'],
        volume=row['Volume']
    )
    lob.process_order(order)

# Output the processed orders (for example purposes)
for order in lob.orders:
    print(f"Order Type: {order.type.name}, Side: {order.side.name}, Price: {order.price}, Volume: {order.volume}, Status: {order.status.name}")
