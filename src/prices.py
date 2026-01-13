import numpy as np
import matplotlib.pyplot as plt

class Prices:
    ELECTRICITY_PRICE_BASE = 20
    PERCENTILE_MIN = 1
    PERCENTILE_MAX = 99
    PREDICTION_WINDOW = 24
    HISTORY_WINDOW = 24

    def __init__(self, external_prices=None):
        self.original_prices = external_prices
        self.price_index = 0
        self.price_shift = 0
        self.price_history = []

        if self.original_prices is not None:
            min_price = np.min(self.original_prices)
            if min_price < 1:
                self.price_shift = 1 - min_price
                self.external_prices = [price + self.price_shift for price in self.original_prices]
            else:
                self.external_prices = self.original_prices.copy()

            self.MIN_PRICE = np.percentile(self.external_prices, self.PERCENTILE_MIN)
            self.MAX_PRICE = np.percentile(self.external_prices, self.PERCENTILE_MAX)
            self.predicted_prices = np.array(self.external_prices[:self.PREDICTION_WINDOW])
            self.price_index = self.PREDICTION_WINDOW
        else:
            self.external_prices = None
            self.MAX_PRICE = 24
            self.MIN_PRICE = 16
            self.predicted_prices = self.get_initial_prices(self.PREDICTION_WINDOW)

    def get_real_price(self, shifted_price):
        return shifted_price - self.price_shift

    def get_initial_prices(self, num_hours):
        if self.external_prices is not None:
            return np.array(self.external_prices[:num_hours])
        else:
            initial_prices = np.zeros(num_hours, dtype=np.float32)
            initial_prices[0] = self.ELECTRICITY_PRICE_BASE
            for i in range(1, num_hours):
                initial_prices[i] = self.ELECTRICITY_PRICE_BASE * (1 + 0.2 * np.sin(i / 24 * 2 * np.pi))
                initial_prices[i] = max(1.0, initial_prices[i])
            return initial_prices

    def get_next_price(self):
        if self.external_prices is not None:
            new_price = self.external_prices[self.price_index % len(self.external_prices)]
            self.price_index += 1
        else:
            new_price = self.ELECTRICITY_PRICE_BASE * (1 + 0.2 * np.sin((self.price_index % 24) / 24 * 2 * np.pi))
            self.price_index += 1

        # Update price history
        self.price_history.append(new_price)
        if len(self.price_history) > self.HISTORY_WINDOW:
            self.price_history.pop(0)  # Remove oldest price

        return new_price

    def get_price_context(self):
        history_avg = np.mean(self.price_history) if self.price_history else None
        future_avg = np.mean(self.predicted_prices)
        return history_avg, future_avg

    def get_predicted_prices(self):
        # Update the prediction window by shifting and adding a new price
        new_price = self.get_next_price()
        self.predicted_prices = np.roll(self.predicted_prices, -1)
        self.predicted_prices[-1] = new_price
        return self.predicted_prices.copy()

    def plot_price_histogram(self, num_bins=50, save_path=None, use_original=False):
        if use_original and self.original_prices is not None:
            prices = self.original_prices
            price_type = "Original"
        elif self.external_prices is not None:
            prices = self.external_prices
            price_type = "Shifted" if self.price_shift else "Original"
        else:
            prices = [self.get_next_price() for _ in range(24 * 7 * 52)]  # Generate a year's worth of prices
            price_type = "Generated"

        plt.figure(figsize=(10, 6))
        plt.hist(prices, bins=num_bins, edgecolor='black')
        plt.title(f'Distribution of Electricity Prices ({price_type})')
        plt.xlabel(f'Price ($/MWh)')
        plt.ylabel('Frequency')
        plt.axvline(np.percentile(prices, self.PERCENTILE_MIN), color='r', linestyle='dashed', linewidth=2, label=f'{self.PERCENTILE_MIN}th Percentile')
        plt.axvline(np.percentile(prices, self.PERCENTILE_MAX), color='g', linestyle='dashed', linewidth=2, label=f'{self.PERCENTILE_MAX}th Percentile')
        plt.axvline(np.mean(prices), color='b', linestyle='dashed', linewidth=2, label='Mean Price')
        plt.legend()

        if save_path:
            plt.savefig(save_path)
            print(f"Histogram saved to {save_path}")
        else:
            plt.show()
        plt.close()

    def get_price_stats(self, use_original=False):
        if use_original and self.original_prices is not None:
            prices = self.original_prices
        elif self.external_prices is not None:
            prices = self.external_prices
        else:
            prices = [self.get_next_price() for _ in range(24 * 7 * 52)]  # Generate a year's worth of prices

        return {
            'min': np.min(prices),
            'max': np.max(prices),
            'mean': np.mean(prices),
            'median': np.median(prices),
            'std': np.std(prices),
            f'{self.PERCENTILE_MIN}th_percentile': np.percentile(prices, self.PERCENTILE_MIN),
            f'{self.PERCENTILE_MAX}th_percentile': np.percentile(prices, self.PERCENTILE_MAX),
            'price_shift': self.price_shift
        }
