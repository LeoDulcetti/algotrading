import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import norm, weibull_min, gamma
import random

class DataLoader:
    def __init__(self, message_data_path, orderbook_data_path):
        self.message_data_path = message_data_path
        self.orderbook_data_path = orderbook_data_path

    def load_data(self):
        message_data = pd.read_csv(self.message_data_path, header=None)
        lob_data = pd.read_csv(self.orderbook_data_path, header=None)
        message_data.columns = ['Time', 'Type', 'OrderID', 'Size', 'Price', 'Direction']
        lob_data = self.rename_lob_columns(lob_data)
        lob_data['Time'] = message_data['Time']
        return message_data, lob_data

    @staticmethod
    def rename_lob_columns(data):
        cols = ['ask_price', 'ask_size', 'bid_price', 'bid_size']
        new_column_names = []
        num_levels = len(data.columns) // len(cols)
        for i in range(num_levels):
            new_column_names.extend(f"{name}_{i+1}" for name in cols)
        data.columns = new_column_names
        return data

class DataProcessor:
    def __init__(self, message_data, lob_data):
        self.message_data = message_data
        self.lob_data = lob_data

    def process_data(self):
        merge_df = self.lob_data.merge(self.message_data, on='Time', how='inner')
        merge_df = merge_df[merge_df['Type'] != 5]
        merge_df['mid_price'] = (merge_df['ask_price_1'] + merge_df['bid_price_1']) / 2
        merge_df['calc_direction'] = np.where(merge_df['mid_price'] < merge_df['Price'], 'Buy', 'Sell')
        merge_df.drop('Direction', axis=1, inplace=True)
        merge_df['event_type'] = merge_df['Type'].astype(str) + '_' + merge_df['calc_direction'].astype(str)
        merge_df['event_type'] = merge_df['event_type'].replace(
            {'1_Sell': 1, '1_Buy': 2, '2_Sell': 3, '2_Buy': 4,
             '3_Sell': 5, '3_Buy': 6, '4_Sell': 7, '4_Buy': 8}
        )
        return merge_df[['Time', 'event_type']]

class PoissonSimulator:
    @staticmethod
    def generate_poisson_events(rate, time_duration):
        num_events = np.random.poisson(rate * time_duration)
        inter_arrival_times = np.random.exponential(1.0 / rate, num_events)
        event_times = np.cumsum(inter_arrival_times)
        return num_events, event_times, inter_arrival_times

    @staticmethod
    def plot_non_sequential_poisson(num_events, event_times, inter_arrival_times, rate, time_duration):
        fig, axs = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'Poisson Process Simulation (λ = {rate}, Duration = {time_duration} seconds)\n', fontsize=16)

        axs[0].step(event_times, np.arange(1, num_events + 1), where='post', color='blue')
        axs[0].set_xlabel('Time')
        axs[0].set_ylabel('Event Number')
        axs[0].set_title(f'Poisson Process Event Times\nTotal: {num_events} events\n')
        axs[0].grid(True)

        axs[1].hist(inter_arrival_times, bins=20, color='green', alpha=0.5)
        axs[1].set_xlabel('Inter-Arrival Time')
        axs[1].set_ylabel('Frequency')
        axs[1].set_title(f'Histogram of Inter-Arrival Times\nMEAN: {np.mean(inter_arrival_times):.2f} | STD: {np.std(inter_arrival_times):.2f}\n')
        axs[1].grid(True, alpha=0.5)
        
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_sequential_poisson(num_events_list, event_times_list, inter_arrival_times_list, rate, time_duration):
        fig, axs = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'Poisson Process Simulation (Duration = {time_duration} seconds)\n', fontsize=16)

        axs[0].set_xlabel('Time')
        axs[0].set_ylabel('Event Number')
        axs[0].set_title(f'Poisson Process Event Times')
        axs[0].grid(True)

        axs[1].set_xlabel('Inter-Arrival Time')
        axs[1].set_ylabel('Frequency')
        axs[1].set_title(f'Histogram of Inter-Arrival Times')
        axs[1].grid(True, alpha=0.5)

        color_palette = plt.get_cmap('tab20')
        colors = [color_palette(i) for i in range(len(rate))]

        for n, individual_rate in enumerate(rate):
            num_events = num_events_list[n]
            event_times = event_times_list[n]
            inter_arrival_times = inter_arrival_times_list[n]

            axs[0].step(event_times, np.arange(1, num_events + 1), where='post', color=colors[n], label=f'λ = {individual_rate}, Total Events: {num_events}')
            axs[1].hist(inter_arrival_times, bins=20, color=colors[n], alpha=0.5, label=f'λ = {individual_rate}, MEAN: {np.mean(inter_arrival_times):.2f}, STD: {np.std(inter_arrival_times):.2f}')

        axs[0].legend()
        axs[1].legend()

        plt.tight_layout()
        plt.show()

    def poisson_simulation(self, rate, time_duration, show_visualization=True):
        if isinstance(rate, (int, float)):
            num_events, event_times, inter_arrival_times = self.generate_poisson_events(rate, time_duration)
            if show_visualization:
                self.plot_non_sequential_poisson(num_events, event_times, inter_arrival_times, rate, time_duration)
            return num_events, event_times, inter_arrival_times

        elif isinstance(rate, list):
            num_events_list = []
            event_times_list = []
            inter_arrival_times_list = []

            for individual_rate in rate:
                num_events, event_times, inter_arrival_times = self.generate_poisson_events(individual_rate, time_duration)
                num_events_list.append(num_events)
                event_times_list.append(event_times)
                inter_arrival_times_list.append(inter_arrival_times)

            if show_visualization:
                self.plot_sequential_poisson(num_events_list, event_times_list, inter_arrival_times_list, rate, time_duration)
            return num_events_list, event_times_list, inter_arrival_times_list

class PriceSampler:
    @staticmethod
    def sampling_return_price(df, df_price_col, num_samples=390):
        seed = 42
        np.random.seed(seed)
        mean_returns_mle, std_returns_mle = norm.fit(df[df_price_col].dropna())
        sample_gaussian_returns = np.random.normal(loc=mean_returns_mle, scale=std_returns_mle, size=num_samples)
        return sample_gaussian_returns

    @staticmethod
    def sampling_return_price_weibull(df, df_price_col, num_samples=390):
        seed = 42
        np.random.seed(seed)
        data_clean = df[df_price_col].dropna()
        c, loc, scale = weibull_min.fit(data_clean, floc=0)
        sample_weibull_returns = weibull_min.rvs(c, loc, scale, size=num_samples)
        return sample_weibull_returns

class VolumeSampler:
    @staticmethod
    def sampling_volume_gamma(df, df_vol_col, num_samples=390):
        seed = 42
        np.random.seed(seed)
        data_clean = df[df_vol_col].dropna()
        c, loc, scale = gamma.fit(data_clean)
        sample_gamma_volume = gamma.rvs(c, loc, scale, size=num_samples)
        return sample_gamma_volume

class MarketSimulator:
    def __init__(self, poisson_df, start_price):
        self.poisson_df = poisson_df
        self.start_price = start_price

    def simulate_market(self, sim_returns_buy_wei, sim_returns_sell_wei, 
                        sim_vol_buy_lim_gam, sim_vol_sell_lim_gam, 
                        sim_vol_buy_mrkt_gam, sim_vol_sell_mrkt_gam):
        results = []
        for index, row in self.poisson_df.iterrows():
            for col in self.poisson_df.columns:
                if row[col] > 0.5:
                    order_type = int(col)
                    if order_type == 1:
                        return_price = random.choice(sim_returns_buy_wei)
                        price = self.start_price * (1 + return_price)
                        volume = random.choice(sim_vol_buy_lim_gam)
                        results.append({'Time': index, 'OrderType': 1, 'Price': price, 'Volume': volume})

                    elif order_type == 2:
                        return_price = random.choice(sim_returns_sell_wei)
                        price = self.start_price * (1 + return_price)
                        volume = random.choice(sim_vol_sell_lim_gam)
                        results.append({'Time': index, 'OrderType': 2, 'Price': price, 'Volume': volume})

                    elif order_type == 3:
                        return_price = random.choice(sim_returns_buy_wei)
                        price = self.start_price * (1 + return_price)
                        volume = random.choice(sim_vol_buy_mrkt_gam)
                        results.append({'Time': index, 'OrderType': 3, 'Price': price, 'Volume': volume})

                    elif order_type == 4:
                        return_price = random.choice(sim_returns_sell_wei)
                        price = self.start_price * (1 + return_price)
                        volume = random.choice(sim_vol_sell_mrkt_gam)
                        results.append({'Time': index, 'OrderType': 4, 'Price': price, 'Volume': volume})
        return pd.DataFrame(results)

# Example usage
data_loader = DataLoader('/Users/constantinwiederin/Documents/GitHub/ie_mbd_sept23/sim/data/message_data.csv', '/Users/constantinwiederin/Documents/GitHub/ie_mbd_sept23/sim/data/orderbook_data.csv')
message_data, lob_data = data_loader.load_data()

data_processor = DataProcessor(message_data, lob_data)
processed_data = data_processor.process_data()

poisson_simulator = PoissonSimulator()
rates = [1, 2, 3]  # Example rates
time_duration = 1  # Example time duration
poisson_events = poisson_simulator.poisson_simulation(rates, time_duration)

price_sampler = PriceSampler()
sampled_returns_buy_wei = price_sampler.sampling_return_price_weibull(processed_data, 'mid_price')
sampled_returns_sell_wei = price_sampler.sampling_return_price_weibull(processed_data, 'mid_price')

volume_sampler = VolumeSampler()
sampled_volumes_buy_lim_gam = volume_sampler.sampling_volume_gamma(processed_data, 'Size')
sampled_volumes_sell_lim_gam = volume_sampler.sampling_volume_gamma(processed_data, 'Size')
sampled_volumes_buy_mrkt_gam = volume_sampler.sampling_volume_gamma(processed_data, 'Size')
sampled_volumes_sell_mrkt_gam = volume_sampler.sampling_volume_gamma(processed_data, 'Size')

market_simulator = MarketSimulator(poisson_events, 100.0)  # Example start price
simulated_market = market_simulator.simulate_market(
    sampled_returns_buy_wei, sampled_returns_sell_wei, 
    sampled_volumes_buy_lim_gam, sampled_volumes_sell_lim_gam, 
    sampled_volumes_buy_mrkt_gam, sampled_volumes_sell_mrkt_gam
)
