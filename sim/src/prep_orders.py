import pandas as pd


class OrdersDataset:
    def __init__(self, orderbook_data_path, message_data_path):
        self.order_data = pd.read_csv(orderbook_data_path, header=None)
        self.message_data = pd.read_csv(message_data_path, header=None)

    def print_shapes(self):
        print(f'Order data shape: {self.order_data.shape}')
        print(f'Message data shape: {self.message_data.shape}')

    def preprocessing(self):
        # Debugging: Print shapes before renaming columns
        self.print_shapes()

        # Ensure the number of columns in order_data matches the length of columns list
        self.order_data.columns = [
            'Ask Price 1', 'Ask Size 1', 'Bid Price 1', 'Bid Size 1',
            'Ask Price 2', 'Ask Size 2', 'Bid Price 2', 'Bid Size 2',
            'Ask Price 3', 'Ask Size 3', 'Bid Price 3', 'Bid Size 3',
            'Ask Price 4', 'Ask Size 4', 'Bid Price 4', 'Bid Size 4',
            'Ask Price 5', 'Ask Size 5', 'Bid Price 5', 'Bid Size 5',
            'Ask Price 6', 'Ask Size 6', 'Bid Price 6', 'Bid Size 6',
            'Ask Price 7', 'Ask Size 7', 'Bid Price 7', 'Bid Size 7',
            'Ask Price 8', 'Ask Size 8', 'Bid Price 8', 'Bid Size 8',
            'Ask Price 9', 'Ask Size 9', 'Bid Price 9', 'Bid Size 9',
            'Ask Price 10', 'Ask Size 10', 'Bid Price 10', 'Bid Size 10'
        ]

        self.message_data.columns = ['Time', 'Type',
                                     'OrderID', 'Size', 'Price', 'Direction']
        self.message_data['Direction'] = self.message_data['Direction'].replace(
            {-1: 'Sell limit order', 1: 'Buy limit order'}
        )

        # Including time
        self.order_data['Time'] = self.message_data['Time']

        # Merging the two datasets on the time column
        merged_data = pd.merge(self.message_data, self.order_data, on='Time')

        # Excluding type 5
        # merged_data = merged_data[merged_data['Type'] != 5]

        # Calculating Mid Price and Spread
        merged_data['Mid Price'] = (
            merged_data['Ask Price 1'] + merged_data['Bid Price 1']) / 2
        merged_data['Spread'] = merged_data['Ask Price 1'] - \
            merged_data['Bid Price 1']

        return merged_data


# Usage example:
if __name__ == '__main__':
    orderbook_data_path = 'sim/data/orderbook_data.csv'
    message_data_path = 'sim/data/message_data.csv'
    orders_dataset = OrdersDataset(orderbook_data_path, message_data_path)
    processed_data = orders_dataset.preprocessing()
    # Debugging: Print the first few rows of processed data
    print(processed_data.head())
