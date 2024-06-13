import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import weibull_min, gamma, norm
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import random
import time
import os
import sys
pd.set_option('future.no_silent_downcasting', True)


class MarketSim:
    '''
    Class that contains the different methods to simulate the market based on the given data.
    Workflow: Data Preprocessing → XGBoost Classifier Order Sim. → Price Sim. → Volume Sim. → Algo simulation → run_market_simulation (Orchestration)
    '''

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
        '''
        Rename the columns of the LOB data to match the order book data.
        '''
        cols = ['ask_price', 'ask_size', 'bid_price', 'bid_size']
        new_column_names = []
        num_levels = len(self.lob_data.columns) // len(cols)
        for i in range(num_levels):
            new_column_names.extend(f"{name}_{i+1}" for name in cols)
        self.lob_data.columns = new_column_names

    def filter_and_calculate(self):
        '''
        Filter out Type 5 messages and calculate the mid price and event type.
        '''
        self.merge_df = self.merge_df[self.merge_df['Type'] != 5]
        self.merge_df['mid_price'] = (
            self.merge_df['ask_price_1'] + self.merge_df['bid_price_1']) / 2
        self.merge_df['calc_direction'] = np.where(
            self.merge_df['mid_price'] < self.merge_df['Price'], 'Buy', 'Sell')
        self.merge_df.drop('Direction', axis=1, inplace=True)
        self.create_event_type()

    def create_event_type(self):
        '''
        Create the event type based on the Type and calc_direction columns.
        '''
        self.merge_df['event_type'] = self.merge_df['Type'].astype(
            str) + '_' + self.merge_df['calc_direction'].astype(str)
        self.merge_df['event_type'] = self.merge_df['event_type'].replace({
            '1_Sell': 1, '1_Buy': 2, '2_Sell': 3, '2_Buy': 4,
            '3_Sell': 5, '3_Buy': 6, '4_Sell': 7, '4_Buy': 8
        }).astype(int)

    ############################## XGBoost Classifier Integration ##############################

    def simulate_event_xgb_prob(self, df_r, num_steps) -> pd.DataFrame:
        """
        Simulate future event probabilities using the XGBoost model, starting from the last known data.

        Parameters:
        - df_r: DataFrame containing the processed data with the same preprocessing applied as during the model's training.
        - num_steps: Length of the simulation (number of future data points to predict).

        Returns:
        - DataFrame containing the probabilities of event types for each step.
        """
        # Import xgb model from pickle file
        current_dir = os.path.dirname(__file__)
        model_path = os.path.join(current_dir, 'xgboost_model.pkl')
        model = joblib.load(model_path)  # "sim/src/xgboost_model.pkl"

        # Preprocess the DataFrame to match the model training process
        le = LabelEncoder()
        df_r['calc_direction'] = le.fit_transform(df_r['calc_direction'])

        # Create lag features for autocorrelation catching
        for lag in range(1, 4):
            df_r.loc[:, f'Time_lag{lag}'] = df_r['Time'].shift(lag)
            df_r.loc[:, f'Size_lag{lag}'] = df_r['Size'].shift(lag)
            df_r.loc[:, f'Price_lag{lag}'] = df_r['Price'].shift(lag)
            df_r.loc[:, f'mid_price_lag{lag}'] = df_r['mid_price'].shift(lag)
            df_r.loc[:, f'calc_direction_lag{lag}'] = df_r['calc_direction'].shift(
                lag)

        df_r = df_r.dropna()

        scaler = StandardScaler()
        features = [col for col in df_r.columns if 'lag' in col]
        df_r.loc[:, features] = scaler.fit_transform(df_r[features])

        initial_features = df_r[features].iloc[-1].values

        current_features = initial_features.copy()
        probabilities_list = []

        for _ in range(num_steps):
            model_input = np.array(current_features).reshape(1, -1)
            next_event_prob = model.predict_proba(model_input)[0]
            probabilities_list.append(next_event_prob)
            current_features = np.roll(current_features, -1)
            current_features[-1] = np.argmax(next_event_prob)

        probabilities = pd.DataFrame(probabilities_list, columns=[
                                     f'event_type_{i}' for i in range(1, 9)])
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

    def sampling_return_norm(self, df, df_price_col, num_samples=390):
        # seed = 42
        # np.random.seed(seed)
        # Compute w/mle
        mean_returns_mle, std_returns_mle = norm.fit(df[df_price_col].dropna())

        # Generate sample directly from the gaussian distribution
        sample_gaussian_returns = np.random.normal(
            loc=mean_returns_mle, scale=std_returns_mle, size=num_samples)

        return sample_gaussian_returns

    def simulate_prices(self):
        '''
        Simulate the prices for 'Buy' and 'Sell' directions using Weibull distribution.
        '''
        price_sell_df, price_buy_df = self.resample_price_data()

        sim_returns_sell_norm = self.sampling_return_norm(
            price_sell_df, 'returns', num_samples=len(price_sell_df))  # len(price_sell_df)
        sim_returns_buy_norm = self.sampling_return_norm(
            price_buy_df, 'returns', num_samples=len(price_buy_df))  # len(price_buy_df)

        return sim_returns_sell_norm, sim_returns_buy_norm

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
        if data_clean.empty:
            print(f"Data clean is empty for {df_vol_col}")
            # Return zeros if data is empty to avoid fitting errors
            return np.zeros(num_samples)

        try:
            c, loc, scale = gamma.fit(data_clean)
            sample_gamma_volume = gamma.rvs(c, loc, scale, size=num_samples)
            sample_gamma_volume = np.round(
                sample_gamma_volume, 0)  # Round to 0 decimal places
        except (ValueError, RuntimeError) as e:
            print(f"Error fitting Gamma distribution: {e}")
            return np.zeros(num_samples)  # Return zeros if there's an error

        return sample_gamma_volume

    def simulate_volumes(self):
        '''
        Simulate the volumes for different event types using Gamma distribution.
        '''
        vol_buy_lim_df, vol_sell_lim_df, vol_buy_mrkt_df, vol_sell_mrkt_df = self.resample_volume_data()

        sim_vol_buy_lim_gam = self.sampling_volume_gamma(
            vol_buy_lim_df, 'closeSize', num_samples=len(vol_buy_lim_df))
        sim_vol_sell_lim_gam = self.sampling_volume_gamma(
            vol_sell_lim_df, 'closeSize', num_samples=len(vol_sell_lim_df))
        sim_vol_buy_mrkt_gam = self.sampling_volume_gamma(
            vol_buy_mrkt_df, 'closeSize', num_samples=len(vol_buy_mrkt_df))
        sim_vol_sell_mrkt_gam = self.sampling_volume_gamma(
            vol_sell_mrkt_df, 'closeSize', num_samples=len(vol_sell_mrkt_df))

        return sim_vol_buy_lim_gam, sim_vol_sell_lim_gam, sim_vol_buy_mrkt_gam, sim_vol_sell_mrkt_gam

    ############################## ALGO SIMULATION ##############################
    def simulate_market(self, start_price, probabilities_df,
                        sim_returns_buy_norm, sim_returns_sell_norm,
                        sim_vol_buy_lim_gam, sim_vol_sell_lim_gam,
                        sim_vol_buy_mrkt_gam, sim_vol_sell_mrkt_gam):
        '''
        Simulate the market based on the given probabilities, return prices, and volumes.
        '''
        results = []

        # Ensure the arrays are not empty
        if sim_returns_buy_norm.size == 0 or sim_returns_sell_norm.size == 0:
            raise ValueError(
                "Simulation arrays are empty. Ensure proper data is provided for simulation.")

        # Iterate through each row and column properly
        for index, row in probabilities_df.iterrows():
            for col in probabilities_df.columns:
                if row[col] > 0.5:  # Check if the probability is greater than 0.1
                    # Extract numeric part from column name
                    order_type = int(col.split('_')[-1])
                    if order_type == 1:
                        # Sell Limit
                        return_price = random.choice(sim_returns_sell_norm)
                        price = start_price * (1 + return_price)
                        volume = random.choice(sim_vol_sell_lim_gam)
                        results.append(
                            {'Time': index, 'OrderType': 1, 'Price': price, 'Volume': volume})
                        start_price = price

                    elif order_type == 2:
                        # Buy Limit
                        return_price = random.choice(sim_returns_buy_norm)
                        price = start_price * (1 + return_price)
                        volume = random.choice(sim_vol_buy_lim_gam)
                        results.append(
                            {'Time': index, 'OrderType': 2, 'Price': price, 'Volume': volume})
                        start_price = price

                    elif order_type == 7:
                        # Sell Market
                        return_price = random.choice(sim_returns_sell_norm)
                        price = start_price * (1 + return_price)
                        volume = random.choice(sim_vol_sell_mrkt_gam)
                        results.append(
                            {'Time': index, 'OrderType': 7, 'Price': price, 'Volume': volume})
                        start_price = price

                    elif order_type == 8:
                        # Sell Market
                        return_price = random.choice(sim_returns_buy_norm)
                        price = start_price * (1 + return_price)
                        volume = random.choice(sim_vol_buy_mrkt_gam)
                        results.append(
                            {'Time': index, 'OrderType': 8, 'Price': price, 'Volume': volume})
                        # Update the starting price for the next iteration based on the last transaction
                        start_price = price

                    ############################## Partial and Full Cancelation ##############################
                    elif order_type == 3:
                        # Partial cancellation of a sell order
                        # Assume 50% of the order is canceled
                        volume = random.choice(sim_vol_sell_lim_gam) * 0.5
                        results.append(
                            {'Time': index, 'OrderType': 3, 'Price': start_price, 'Volume': volume})

                    elif order_type == 4:
                        # Partial cancellation of a buy order
                        # Assume 50% of the order is canceled
                        volume = random.choice(sim_vol_buy_lim_gam) * 0.5
                        results.append(
                            {'Time': index, 'OrderType': 4, 'Price': start_price, 'Volume': volume})

                    elif order_type == 5:
                        # Full cancellation of a sell order
                        volume = 0  # Full cancellation implies volume is reduced to 0
                        results.append(
                            {'Time': index, 'OrderType': 5, 'Price': start_price, 'Volume': volume})

                    elif order_type == 6:
                        # Full cancellation of a buy order
                        volume = 0  # Full cancellation implies volume is reduced to 0
                        results.append(
                            {'Time': index, 'OrderType': 6, 'Price': start_price, 'Volume': volume})

        return results

    ############################## Orchestration ##############################
    # 390 min but 34,200 seconds
    def run_market_simulation(self, start_price=585.0, num_steps=34200):
        '''
        Run the complete simulation from data processing to probability calculations.
        - Doing PreProcessing
        - Calculating Event Rates
        - Running XGB Simulations that output directly the probabilities
        - Resampling Data
        - Simulating Prices
        - Simulating Volumes
        - Simulating Market
        '''
        # Use the correct DataFrame for the XGBoost function
        df_r = self.merge_df[['Time', 'Size', 'Price',
                              'mid_price', 'calc_direction', 'event_type']].copy()

        probabilities = self.simulate_event_xgb_prob(df_r, num_steps)

        # print(f"Probabilities head:\n{probabilities.head()}")

        # Resample price data before simulating prices
        self.resample_price_data()
        sim_returns_sell_norm, sim_returns_buy_norm = self.simulate_prices()

        # print(f"Sim returns sell wei:\n{sim_returns_sell_wei}")
        # print(f"Sim returns buy wei:\n{sim_returns_buy_wei}")

        # Resample volume data before simulating volumes
        self.resample_volume_data()
        sim_vol_buy_lim_gam, sim_vol_sell_lim_gam, sim_vol_buy_mrkt_gam, sim_vol_sell_mrkt_gam = self.simulate_volumes()

        # print(f"Sim volume buy limit gamma:\n{sim_vol_buy_lim_gam}")
        # print(f"Sim volume sell limit gamma:\n{sim_vol_sell_lim_gam}")
        # print(f"Sim volume buy market gamma:\n{sim_vol_buy_mrkt_gam}")
        # print(f"Sim volume sell market gamma:\n{sim_vol_sell_mrkt_gam}")

        # Simulate market
        sim_data = self.simulate_market(start_price, probabilities, sim_returns_buy_norm, sim_returns_sell_norm,
                                        sim_vol_buy_lim_gam, sim_vol_sell_lim_gam,
                                        sim_vol_buy_mrkt_gam, sim_vol_sell_mrkt_gam)

        return probabilities, sim_returns_sell_norm, sim_returns_buy_norm, sim_vol_buy_lim_gam, sim_vol_sell_lim_gam, sim_vol_buy_mrkt_gam, sim_vol_sell_mrkt_gam, sim_data


class SimulationRunner:
    def __init__(self, num_simulations, output_path=None):
        '''
        Initialize the SimulationRunner with the specified output file and number of simulations.
        - output_file: str, path to the output CSV file
        - num_simulations: int, number of simulations to run (Day of Trading)
        '''
        if output_path is None:
            base_dir = os.path.dirname(__file__)
            output_path = os.path.join(
                base_dir, '..', '..', 'sim', 'data', 'logs', 'simulation_results.csv')
            output_path = os.path.abspath(output_path)
        print(f"Output path: {output_path}")

        self.output_file = output_path
        self.num_simulations = num_simulations

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Clear the log file everytime the class is run
        self.clear_output_file()

    def clear_output_file(self):
        '''
        Clear the content of the output file.
        '''
        with open(self.output_file, 'w') as file:
            pass

    def run_simulations(self, start_price=585.0, concat=True):
        '''
        Run the specified number of market simulations and store the results in a CSV file and also return a list of dataframes.
        '''
        # start_price = 200.0  # Initial starting price
        results_sim_list_df = []  # to list the dataframes
        for i in range(self.num_simulations):
            print(f"Running simulation {i+1}/{self.num_simulations}")
            for _ in range(3):
                sys.stdout.write('.')
                sys.stdout.flush()
                time.sleep(1)
            print()
            market_sim_instance = MarketSim()
            probabilities, sim_returns_sell_norm, sim_returns_buy_norm, sim_vol_buy_lim_gam, sim_vol_sell_lim_gam, sim_vol_buy_mrkt_gam, sim_vol_sell_mrkt_gam, sim_data = market_sim_instance.run_market_simulation(
                start_price)

            # Convert sim_data to DataFrame
            sim_data_df = pd.DataFrame(sim_data)
            header = ['Time', 'OrderType', 'Price', 'Volume']
            sim_data_df = sim_data_df[header]

            # Append to CSV file
            if os.path.getsize(self.output_file) == 0:
                sim_data_df.to_csv(self.output_file, index=False)
            else:
                sim_data_df.to_csv(self.output_file, mode='a',
                                   header=False, index=False)

            # Update start_price for the next simulation
            if not sim_data_df.empty:
                start_price = sim_data_df['Price'].iloc[-1]

            print(
                f"Simulation {i+1} completed and results stored in CSV file.")
            print(
                f"Data Volume: {sim_data_df.shape}")
            time.sleep(1)  # Delay before the next simulation

            results_sim_list_df.append(sim_data_df)
        if concat == True:
            print("Concatenating all the dataframes...")
            result = pd.concat(results_sim_list_df)
            print(f"Final Data Volume: {result.shape}")
        else:
            result = results_sim_list_df
        print("All simulations completed.")

        print("""
            Mapping the OrderType for Reference:\n
            OrderType    |Type                   |Direction
            ------------------------------------------------------
            1            |Limit                  |Sell
            2            |Limit                  |Buy
            3            |Partial Cancelation    |Sell
            4            |Partial Cancelation    |Buy
            5            |Full Cancelation       |Sell
            6            |Full Cancelation       |Buy
            7            |Market                 |Sell
            8            |Market                 |Buy
            """)

        return result


if __name__ == '__main__':
    # Instantiate and run the simulation runner
    # output_folder = "/Users/lucazosso/Desktop/IE_Course/Term_3/Algorithmic_Trading/ie_mbd_sept23/sim/data/logs"
    # Change to your desired path
    # output_file = output_folder + '/simulation_results.csv'
    num_simulations = 2  # Specify the number of simulations to run
    simulation_runner = SimulationRunner(num_simulations)
    simulation_runner.run_simulations(concat=True)
