from lob_order import *
import heapq
from collections import defaultdict
import datetime
import time
from lob_data import LOBupdate

class LOBMatching:
    def __init__(self, opening_time, closing_time, tick_size, time_resolution):
        self.opening_time = datetime.datetime.strptime(opening_time, '%H:%M:%S').time()
        self.closing_time = datetime.datetime.strptime(closing_time, '%H:%M:%S').time()
        self.tick_size = float(tick_size)
        self.time_resolution = float(time_resolution)
        self.bids = []
        self.asks = []
        self.orders = {}
        self.subscribers = defaultdict(list)
        self.order_id_counter = 1

    def resetForNewDay(self):
        self.bids = []
        self.asks = []
        self.orders = {}
        self.order_id_counter = 1
    
    def processMessage(self, message):
        OrderClass = LimitOrder if message['type'] in ['ask', 'bid'] else None
        if OrderClass:
            if 'order_id' not in message:
                message['order_id'] = self.order_id_counter
                self.order_id_counter += 1
            order_side = 'SELL' if message['type'] == 'ask' else 'BUY'
            order = OrderClass(message['order_id'], OrderSide[order_side], message['size'], message['price'])
            heap = self.asks if message['type'] == 'ask' else self.bids
            heapq.heappush(heap, (order.price * (-1 if message['type'] == 'bid' else 1), order))
            self.orders[order.order_id] = order
            
    def update_lob_state(self):
        # Local import to avoid circular dependency
        from lob_data import LOBupdate
        bid_prices, bid_volumes = zip(*[(-price, order.size) for price, order in self.bids]) if self.bids else ([], [])
        ask_prices, ask_volumes = zip(*[(price, order.size) for price, order in self.asks]) if self.asks else ([], [])
        orders = list(self.orders.values())
        update = LOBupdate(datetime.datetime.now(), list(bid_prices), list(bid_volumes), list(ask_prices), list(ask_volumes), orders)
        self.notify_subscribers(update)

            
    def subscribeToLOB(self, subscriber):
        """ Subscribes a trading algorithm to receive updates from the LOB. """
        self.subscribers[subscriber.id].append(subscriber)

    def sendOrder(self, order):
        """ Dispatches orders to appropriate handling methods based on their type. """
        order.order_id = self.order_id_counter
        self.order_id_counter += 1
        self.orders[order.order_id] = order
        order.status = OrderStatus.POSTED

        if order.type == OrderType.LIMIT:
            self.process_limit_order(order)
        elif order.type == OrderType.MARKET:
            self.process_market_order(order)
        elif order.type == OrderType.PARTIAL_CANCEL:
            self.process_partial_cancel(order)
        elif order.type == OrderType.FULL_CANCEL:
            self.process_full_cancel(order)
        
        return order.order_id

    def process_limit_order(self, order):
        """Processes limit orders by attempting to match or placing them in the order book."""
        opposite_heap = self.asks if order.side == OrderSide.BUY else self.bids
        own_heap = self.bids if order.side == OrderSide.BUY else self.asks
        price_key = -order.price if order.side == OrderSide.BUY else order.price

        matched = False
        while opposite_heap and order.size > 0:
            best_price, best_order = opposite_heap[0]
            if (order.side == OrderSide.BUY and -price_key <= best_price) or \
            (order.side == OrderSide.SELL and price_key >= -best_price):
                trade_size = min(order.size, best_order.size)
                self.execute_trade(order, best_order, trade_size)
                if best_order.size == 0:
                    heapq.heappop(opposite_heap)
                matched = True
            else:
                break

        if not matched and order.size > 0:
            heapq.heappush(own_heap, (price_key, order))


    def process_market_order(self, order):
        """Processes market orders by matching with the best available opposite order."""
        opposite_heap = self.asks if order.side == OrderSide.BUY else self.bids

        while opposite_heap and order.size > 0:
            best_price, best_order = heapq.heappop(opposite_heap)
            trade_size = min(order.size, best_order.size)
            self.execute_trade(order, best_order, trade_size)

            if best_order.size > 0:
                heapq.heappush(opposite_heap, (best_price, best_order))

        # Update status to EXECUTED if completely filled
        if order.size == 0:
            order.status = OrderStatus.EXECUTED


    def process_partial_cancel(self, order):
        """ Adjusts the size of an existing order based on a partial cancellation request. """
        original_order = self.orders.get(order.id)
        if original_order and original_order.size > order.size:
            original_order.size -= order.size
        elif original_order:
            self.process_full_cancel(FullCancel(original_order.order_id))

    def process_full_cancel(self, order):
        """ Completely removes an order from the order book and updates its status. """
        original_order = self.orders.pop(order.id, None)
        if original_order:
            original_order.status = OrderStatus.CANCELLED
            self.remove_order_from_heap(original_order)


    def remove_order_from_heap(self, order):
        """ Removes an order from the appropriate heap. """
        heap = self.bids if order.side == OrderSide.BUY else self.asks
        try:
            heap.remove((-order.price if order.side == OrderSide.BUY else order.price, order))
            heapq.heapify(heap)
        except ValueError:
            pass  # Order not found or already executed

    def execute_trade(self, buy_order, sell_order, trade_size):
        """Executes a trade and updates order statuses."""
        print(f"Executing trade: {trade_size} units at price {sell_order.price}")
        buy_order.size -= trade_size
        sell_order.size -= trade_size

        # Check and update the status if the order size reaches zero
        if buy_order.size == 0:
            buy_order.status = OrderStatus.EXECUTED
            print(f"Buy order executed: Order ID {buy_order.order_id}")
        if sell_order.size == 0:
            sell_order.status = OrderStatus.EXECUTED
            print(f"Sell order executed: Order ID {sell_order.order_id}")

        self.notify_subscribers(buy_order)
        self.notify_subscribers(sell_order)


    def update_lob_state(self):
        # Extracts the current state of bids and asks to create a snapshot
        bid_prices, bid_volumes = zip(*[(-price, order.size) for price, order in self.bids]) if self.bids else ([], [])
        ask_prices, ask_volumes = zip(*[(price, order.size) for price, order in self.asks]) if self.asks else ([], [])
        orders = list(self.orders.values())

        # Generate the LOB update
        update = LOBupdate(datetime.datetime.now(), list(bid_prices), list(bid_volumes),
                        list(ask_prices), list(ask_volumes), orders)

        self.notify_subscribers(update)

    def notify_subscribers(self, update):
        for subs in self.subscribers.values():
            for sub in subs:
                sub.onData(update)

    def run(self):
        while datetime.datetime.now().time() >= self.opening_time and datetime.datetime.now().time() <= self.closing_time:
            time.sleep(self.time_resolution)
            self.update_lob_state()
            for _, algos in self.subscribers.items():
                for subscriber in algos:
                    subscriber.onTime(datetime.datetime.now().time())
