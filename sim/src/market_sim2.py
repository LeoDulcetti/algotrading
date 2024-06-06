import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import weibull_min, gamma
import random
import time
import os


class MarketSim:
    def __init__(self):
        '''
        Initialize the MarketSim class with the following parameters:
        - message_data_path: str, path to the message data CSV
        - orderbook_data_path: str, path to the order book data CSV
        '''
        self.message_data_path = "/Users/lucazosso/Desktop/IE_Course/Term_3/Algorithmic_Trading/ie_mbd_sept23/sim/data/message_data.csv"
        self.orderbook_data_path = "/Users/lucazosso/Desktop/IE_Course/Term_3/Algorithmic_Trading/ie_mbd_sept23/sim/data/orderbook_data.csv"
        self.message_data = pd.read_csv(self.message_data_path, header=None)
        self.lob_data = pd.read_csv(self.orderbook_data_path, header=None)
        self.process_data()

    def process_data(self):
        '''
        Preprocess data by setting column names, merging datasets, and calculating necessary values.
        '''
        self.message_data.columns = ['Time', 'Type',
                                     'OrderID', 'Size', 'Price', 'Direction']
        self.rename_lob_columns()
        self.lob_data['Time'] = self.message_data['Time']
        self.merge_df = self.lob_data.merge(
            self.message_data, on='Time', how='inner')
        self.filter_and_calculate()

    def rename_lob_columns(self):
        cols = ['ask_price', 'ask_size', 'bid_price', 'bid_size']
        new_column_names = []
        num_levels = len(self.lob_data.columns) // len(cols)
        for i in range(num_levels):
            new_column_names.extend(f"{name}_{i+1}" for name in cols)
        self.lob_data.columns = new_column_names

    def filter_and_calculate(self):
        self.merge_df = self.merge_df[self.merge_df['Type'] != 5]
        self.merge_df['mid_price'] = (
            self.merge_df['ask_price_1'] + self.merge_df['bid_price_1']) / 2
        self.merge_df['calc_direction'] = np.where(
            self.merge_df['mid_price'] < self.merge_df['Price'], 'Buy', 'Sell')
        self.merge_df.drop('Direction', axis=1, inplace=True)
        self.create_event_type()

    def create_event_type(self):
        self.merge_df['event_type'] = self.merge_df['Type'].astype(
            str) + '_' + self.merge_df['calc_direction'].astype(str)
        self.merge_df['event_type'] = self.merge_df['event_type'].replace({
            '1_Sell': 1, '1_Buy': 2, '2_Sell': 3, '2_Buy': 4,
            '3_Sell': 5, '3_Buy': 6, '4_Sell': 7, '4_Buy': 8
        })

    def calculate_event_rates(self):
        time_duration = self.merge_df['Time'].max(
        ) - self.merge_df['Time'].min()
        event_counts = {
            1: self.merge_df[self.merge_df['event_type'] == 1].shape[0],
            2: self.merge_df[self.merge_df['event_type'] == 2].shape[0],
            3: self.merge_df[self.merge_df['event_type'] == 3].shape[0],
            4: self.merge_df[self.merge_df['event_type'] == 4].shape[0],
            5: self.merge_df[self.merge_df['event_type'] == 5].shape[0],
            6: self.merge_df[self.merge_df['event_type'] == 6].shape[0],
            7: self.merge_df[self.merge_df['event_type'] == 7].shape[0],
            8: self.merge_df[self.merge_df['event_type'] == 8].shape[0],
        }
        self.event_rates = {
            etype: count / time_duration for etype, count in event_counts.items()}
        print("Event Rates:")
        for etype, rate in self.event_rates.items():
            print(f"Event Type {etype}: {rate}")

    def generate_poisson_events(self, rate, time_duration):
        num_events = np.random.poisson(rate * time_duration)
        inter_arrival_times = np.random.exponential(1.0 / rate, num_events)
        event_times = np.cumsum(inter_arrival_times)
        return num_events, event_times, inter_arrival_times

    def poisson_simulation(self, rate, time_duration):
        return self.generate_poisson_events(rate, time_duration)

    def run_poisson_simulations(self):
        sim_results = {}
        count_types_list = [1, 2, 3, 4, 5, 6, 7, 8]

        for event_type in count_types_list:
            rate = self.event_rates.get(event_type, 0)
            num_events, event_times, inter_arrival_times = self.poisson_simulation(
                rate, self.merge_df['Time'].max() - self.merge_df['Time'].min())
            sim_results[event_type] = {
                'num_events': num_events,
                'event_times': event_times,
                'inter_arrival_times': inter_arrival_times
            }
        print("Simulation Results:")
        for event_type, results in sim_results.items():
            print(f"Event Type {event_type}: {results['num_events']} events")
        return sim_results

    def calculate_probabilities(self, sim_results):
        '''
        Calculate the probability of each event type at uniform time intervals based on simulated results.
        '''
        time_duration = self.merge_df['Time'].max(
        ) - self.merge_df['Time'].min()
        time_duration_int = int(time_duration)
        timestamps = np.linspace(0, time_duration, num=time_duration_int + 1)
        probabilities = pd.DataFrame(index=timestamps[:-1])
        total_counts_per_bin = np.zeros(len(timestamps) - 1)

        for event_type, data in sim_results.items():
            event_times = data['event_times']
            hist, _ = np.histogram(event_times, bins=timestamps)
            total_counts_per_bin += hist

        for event_type, data in sim_results.items():
            event_times = data['event_times']
            hist, _ = np.histogram(event_times, bins=timestamps)
            probabilities[event_type] = np.where(
                total_counts_per_bin > 0, hist / total_counts_per_bin, 0)

        row_sums = probabilities.sum(axis=1)
        if np.allclose(row_sums, 1, atol=1e-2):
            print("Rows sum to 1 within tolerance.")
        else:
            print("Rows do not sum to 1, check calculations.")

        return probabilities

    ############################## Price Simulation ##############################

    def resample_price_data(self):
        '''
        Resample the price data for 'Buy' and 'Sell' directions into 1-minute bins.
        '''
        price_sell_df = self.merge_df[self.merge_df['calc_direction'] == 'Sell'][[
            'Time', 'mid_price']]
        price_buy_df = self.merge_df[self.merge_df['calc_direction'] == 'Buy'][[
            'Time', 'mid_price']]

        price_sell_df['Time'] = pd.to_datetime(price_sell_df['Time'], unit='s')
        price_sell_df.set_index('Time', inplace=True)
        price_sell_df = price_sell_df.resample('1min').ohlc()
        price_sell_df = price_sell_df['mid_price']
        price_sell_df = price_sell_df.add_suffix('_mid_price')

        price_buy_df['Time'] = pd.to_datetime(price_buy_df['Time'], unit='s')
        price_buy_df.set_index('Time', inplace=True)
        price_buy_df = price_buy_df.resample('1min').ohlc()
        price_buy_df = price_buy_df['mid_price']
        price_buy_df = price_buy_df.add_suffix('_mid_price')

        price_sell_df['returns'] = (price_sell_df['close_mid_price'] -
                                    price_sell_df['close_mid_price'].shift(1)) / price_sell_df['close_mid_price'].shift(1)
        price_buy_df['returns'] = (price_buy_df['close_mid_price'] -
                                   price_buy_df['close_mid_price'].shift(1)) / price_buy_df['close_mid_price'].shift(1)

        self.price_sell_df = price_sell_df
        self.price_buy_df = price_buy_df
        return price_sell_df, price_buy_df

    def sampling_return_price_weibull(self, df, df_price_col, num_samples=390):
        '''
        Sample return prices using a Weibull distribution.
        '''
        seed = 42
        np.random.seed(seed)
        data_clean = df[df_price_col].dropna()
        c, loc, scale = weibull_min.fit(data_clean, floc=0)
        sample_weibull_returns = weibull_min.rvs(
            c, loc, scale, size=num_samples)
        return sample_weibull_returns

    def simulate_prices(self):
        '''
        Simulate the prices for 'Buy' and 'Sell' directions using Weibull distribution.
        '''
        sim_returns_sell_wei = self.sampling_return_price_weibull(
            self.price_sell_df, 'returns', num_samples=len(self.price_sell_df))
        sim_returns_buy_wei = self.sampling_return_price_weibull(
            self.price_buy_df, 'returns', num_samples=len(self.price_buy_df))

        return sim_returns_sell_wei, sim_returns_buy_wei

    ############################## Volume Simulation ##############################

    def resample_volume_data(self):
        '''
        Resample the volume data for different event types into 1-minute bins.
        '''
        vol_buy_lim_df = self.merge_df[self.merge_df['event_type'] == 1][[
            'Time', 'Size']]
        vol_sell_lim_df = self.merge_df[self.merge_df['event_type'] == 2][[
            'Time', 'Size']]
        vol_buy_mrkt_df = self.merge_df[self.merge_df['event_type'] == 7][[
            'Time', 'Size']]
        vol_sell_mrkt_df = self.merge_df[self.merge_df['event_type'] == 8][[
            'Time', 'Size']]

        vol_buy_lim_df['Time'] = pd.to_datetime(
            vol_buy_lim_df['Time'], unit='s')
        vol_buy_lim_df.set_index('Time', inplace=True)
        vol_buy_lim_df = vol_buy_lim_df.resample('1min').ohlc()
        vol_buy_lim_df = vol_buy_lim_df['Size']
        vol_buy_lim_df = vol_buy_lim_df.add_suffix('Size')

        vol_sell_lim_df['Time'] = pd.to_datetime(
            vol_sell_lim_df['Time'], unit='s')
        vol_sell_lim_df.set_index('Time', inplace=True)
        vol_sell_lim_df = vol_sell_lim_df.resample('1min').ohlc()
        vol_sell_lim_df = vol_sell_lim_df['Size']
        vol_sell_lim_df = vol_sell_lim_df.add_suffix('Size')

        vol_buy_mrkt_df['Time'] = pd.to_datetime(
            vol_buy_mrkt_df['Time'], unit='s')
        vol_buy_mrkt_df.set_index('Time', inplace=True)
        vol_buy_mrkt_df = vol_buy_mrkt_df.resample('1min').ohlc()
        vol_buy_mrkt_df = vol_buy_mrkt_df['Size']
        vol_buy_mrkt_df = vol_buy_mrkt_df.add_suffix('Size')

        vol_sell_mrkt_df['Time'] = pd.to_datetime(
            vol_sell_mrkt_df['Time'], unit='s')
        vol_sell_mrkt_df.set_index('Time', inplace=True)
        vol_sell_mrkt_df = vol_sell_mrkt_df.resample('1min').ohlc()
        vol_sell_mrkt_df = vol_sell_mrkt_df['Size']
        vol_sell_mrkt_df = vol_sell_mrkt_df.add_suffix('Size')

        self.vol_buy_lim_df = vol_buy_lim_df
        self.vol_sell_lim_df = vol_sell_lim_df
        self.vol_buy_mrkt_df = vol_buy_mrkt_df
        self.vol_sell_mrkt_df = vol_sell_mrkt_df

        return vol_buy_lim_df, vol_sell_lim_df, vol_buy_mrkt_df, vol_sell_mrkt_df

    def sampling_volume_gamma(self, df, df_vol_col, num_samples=390):
        '''
        Sample volumes using a Gamma distribution.
        '''
        seed = 42
        np.random.seed(seed)
        data_clean = df[df_vol_col].dropna()
        c, loc, scale = gamma.fit(data_clean)
        sample_gamma_volume = gamma.rvs(c, loc, scale, size=num_samples)
        sample_gamma_volume = np.round(
            sample_gamma_volume, 0)  # Round to 0 decimal places
        return sample_gamma_volume

    def simulate_volumes(self):
        '''
        Simulate the volumes for different event types using Gamma distribution.
        '''
        sim_vol_buy_lim_gam = self.sampling_volume_gamma(
            self.vol_buy_lim_df, 'closeSize', num_samples=len(self.vol_buy_lim_df))
        sim_vol_sell_lim_gam = self.sampling_volume_gamma(
            self.vol_sell_lim_df, 'closeSize', num_samples=len(self.vol_sell_lim_df))
        sim_vol_buy_mrkt_gam = self.sampling_volume_gamma(
            self.vol_buy_mrkt_df, 'closeSize', num_samples=len(self.vol_buy_mrkt_df))
        sim_vol_sell_mrkt_gam = self.sampling_volume_gamma(
            self.vol_sell_mrkt_df, 'closeSize', num_samples=len(self.vol_sell_mrkt_df))

        return sim_vol_buy_lim_gam, sim_vol_sell_lim_gam, sim_vol_buy_mrkt_gam, sim_vol_sell_mrkt_gam

    ############################## ALGO SIMULATION ##############################

    def simulate_market(self, start_price, poisson_df,
                        sim_returns_buy_wei, sim_returns_sell_wei,
                        sim_vol_buy_lim_gam, sim_vol_sell_lim_gam,
                        sim_vol_buy_mrkt_gam, sim_vol_sell_mrkt_gam):
        '''
        Simulate the market based on the given probabilities, return prices, and volumes.
        '''
        results = []

        # Iterate through each row and column properly
        for index, row in poisson_df.iterrows():
            for col in poisson_df.columns:
                if row[col] > 0.1:  # Check if the probability is greater than 0.1
                    # Assuming column names are strings that represent integers
                    order_type = int(col)
                    if order_type == 1:
                        # Buy Limit
                        return_price = random.choice(sim_returns_buy_wei)
                        price = start_price * (1 + return_price)
                        volume = random.choice(sim_vol_buy_lim_gam)
                        results.append(
                            {'Time': index, 'OrderType': 1, 'Price': price, 'Volume': volume})

                    elif order_type == 2:
                        # Sell Limit
                        return_price = random.choice(sim_returns_sell_wei)
                        price = start_price * (1 + return_price)
                        volume = random.choice(sim_vol_sell_lim_gam)
                        results.append(
                            {'Time': index, 'OrderType': 2, 'Price': price, 'Volume': volume})

                    elif order_type == 7:
                        # Buy Market
                        return_price = random.choice(sim_returns_buy_wei)
                        price = start_price * (1 + return_price)
                        volume = random.choice(sim_vol_buy_mrkt_gam)
                        results.append(
                            {'Time': index, 'OrderType': 7, 'Price': price, 'Volume': volume})

                    elif order_type == 8:
                        # Sell Market
                        return_price = random.choice(sim_returns_sell_wei)
                        price = start_price * (1 + return_price)
                        volume = random.choice(sim_vol_sell_mrkt_gam)
                        results.append(
                            {'Time': index, 'OrderType': 8, 'Price': price, 'Volume': volume})

                    # Update the starting price for the next iteration based on the last transaction
                    start_price = price

        return results

    ############################## Orchestration ##############################

    def run_market_simulation(self, start_price=585.0):
        '''
        Run the complete simulation from data processing to probability calculations.
        - Doing PreProcessing
        - Calculating Event Rates
        - Running Poisson Simulations
        - Calculating Probabilities
        - Resampling Data
        - Simulating Prices
        - Simulating Volumes
        - Simulating Market
        '''
        self.calculate_event_rates()
        sim_results = self.run_poisson_simulations()
        probabilities = self.calculate_probabilities(sim_results)

        # Resample price data before simulating prices
        self.resample_price_data()
        sim_returns_sell_wei, sim_returns_buy_wei = self.simulate_prices()

        # Resample volume data before simulating volumes
        self.resample_volume_data()
        sim_vol_buy_lim_gam, sim_vol_sell_lim_gam, sim_vol_buy_mrkt_gam, sim_vol_sell_mrkt_gam = self.simulate_volumes()

        # Simulate market
        sim_data = self.simulate_market(start_price, probabilities, sim_returns_buy_wei, sim_returns_sell_wei,
                                        sim_vol_buy_lim_gam, sim_vol_sell_lim_gam,
                                        sim_vol_buy_mrkt_gam, sim_vol_sell_mrkt_gam)

        return probabilities, sim_returns_sell_wei, sim_returns_buy_wei, sim_vol_buy_lim_gam, sim_vol_sell_lim_gam, sim_vol_buy_mrkt_gam, sim_vol_sell_mrkt_gam, sim_data


class SimulationRunner:
    def __init__(self, output_file, num_simulations):
        '''
        Initialize the SimulationRunner with the specified output file and number of simulations.
        - output_file: str, path to the output CSV file
        - num_simulations: int, number of simulations to run
        '''
        self.output_file = output_file
        self.num_simulations = num_simulations

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    def run_simulations(self):
        '''
        Run the specified number of market simulations and store the results in a CSV file.
        '''
        start_price = 585.0  # Initial starting price

        for i in range(self.num_simulations):
            print(f"Running simulation {i+1}/{self.num_simulations}")
            market_sim_instance = MarketSim()
            probabilities, sim_returns_sell_wei, sim_returns_buy_wei, sim_vol_buy_lim_gam, sim_vol_sell_lim_gam, sim_vol_buy_mrkt_gam, sim_vol_sell_mrkt_gam, sim_data = market_sim_instance.run_market_simulation(
                start_price)

            # Convert sim_data to DataFrame
            sim_data_df = pd.DataFrame(sim_data)

            # Append to CSV file
            if not os.path.isfile(self.output_file):
                sim_data_df.to_csv(self.output_file, index=False)
            else:
                sim_data_df.to_csv(self.output_file, mode='a',
                                   header=False, index=False)

            # Update start_price for the next simulation
            if not sim_data_df.empty:
                start_price = sim_data_df.iloc[-1]['Price']

            print(
                f"Simulation {i+1} completed and results stored in CSV file.")
            time.sleep(1)  # Delay before the next simulation


# Instantiate and run the simulation runner
output_folder = "/Users/lucazosso/Desktop/IE_Course/Term_3/Algorithmic_Trading/ie_mbd_sept23/sim/data/logs"
# Change to your desired path
output_file = output_folder + '/simulation_results.csv'
num_simulations = 10  # Specify the number of simulations to run
simulation_runner = SimulationRunner(output_file, num_simulations)
simulation_runner.run_simulations()
