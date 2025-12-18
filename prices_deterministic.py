import numpy as np
import matplotlib.pyplot as plt
from collections import deque

class Prices:
    ELECTRICITY_PRICE_BASE = 20
    PERCENTILE_MIN = 1
    PERCENTILE_MAX = 99
    PREDICTION_WINDOW = 24
    HISTORY_WINDOW = 24

    def __init__(self, external_prices=None):
        self.original_prices = external_prices
        self.external_prices = None

        self.price_shift = 0
        self.price_index = 0
        self.price_history = []
        self.predicted_prices = None

        if self.original_prices is not None:
            prices = np.asarray(self.original_prices, dtype=np.float32)
            if prices.size == 0:
                raise ValueError("external_prices must be a non-empty sequence")
 
            min_price = float(np.min(prices))
            if min_price < 1:
                self.price_shift = 1 - min_price
                prices = prices + self.price_shift

            self.external_prices = prices
            self.MIN_PRICE = float(np.percentile(self.external_prices, self.PERCENTILE_MIN))
            self.MAX_PRICE = float(np.percentile(self.external_prices, self.PERCENTILE_MAX))
        else:
            self.external_prices = None
            # Keep your defaults for normalization bounds
            self.MAX_PRICE = 24
            self.MIN_PRICE = 16

        # IMPORTANT: initialize state in one place
        self.reset(start_index=0)

    # ---------- core price model (pure) ----------
    def _synthetic_price_at(self, t: int) -> float:
        # deterministic daily sinusoid
        base = self.ELECTRICITY_PRICE_BASE
        return float(max(1.0, base * (1 + 0.2 * np.sin((t % 24) / 24 * 2 * np.pi))))

    def get_real_price(self, shifted_price):
        return shifted_price - self.price_shift

    def reset(self, start_index: int = 0):
        """Reset internal timeline/state to episode start.

        start_index is the index in external_prices for the *first* element
        of the 24h prediction window.
        """
        self.price_history = []

        if self.external_prices is not None:
            n = len(self.external_prices)
            start_index = start_index % n

            # 24-hour prediction window starting at start_index, with wrap
            idxs = (np.arange(self.PREDICTION_WINDOW, dtype=np.int64) + start_index) % n
            self.predicted_prices = self.external_prices[idxs].astype(np.float32, copy=True)

            # next unseen price *after* the window
            self.price_index = (start_index + self.PREDICTION_WINDOW) % n
        else:
            # synthetic mode
            self.price_index = self.PREDICTION_WINDOW
            self.predicted_prices = np.array(
                [self._synthetic_price_at(i) for i in range(self.PREDICTION_WINDOW)],
                dtype=np.float32,
            )

    # ---------- *stateful* stepping (used by env.step) ----------
    def get_next_price(self):
        if self.external_prices is not None:
            new_price = float(self.external_prices[self.price_index % len(self.external_prices)])
        else:
            new_price = self._synthetic_price_at(self.price_index)

        self.price_index += 1

        self.price_history.append(new_price)
        # deque automatically removes oldest when maxlen exceeded | Keep the old line for now. Remove during next PR.

        return new_price

    def get_price_context(self):
        history_avg = float(np.mean(self.price_history)) if self.price_history else None
        future_avg = float(np.mean(self.predicted_prices))
        return history_avg, future_avg

    def advance_and_get_predicted_prices(self): # Changed name for readability
        new_price = self.get_next_price()
        self.predicted_prices = np.roll(self.predicted_prices, -1)
        self.predicted_prices[-1] = new_price
        return self.predicted_prices.copy()

    # ---------- NON-MUTATING utilities ----------
    def _generated_prices_for_stats(self, n: int):
        # Do NOT touch price_index/history here.
        return np.array([self._synthetic_price_at(i) for i in range(n)], dtype=np.float32)

    def get_price_stats(self, use_original=False):
        if use_original and self.original_prices is not None:
            prices = np.asarray(self.original_prices, dtype=np.float32)
        elif self.external_prices is not None:
            prices = np.asarray(self.external_prices, dtype=np.float32)
        else:
            prices = self._generated_prices_for_stats(24 * 7 * 52)

        return {
            'min': float(np.min(prices)),
            'max': float(np.max(prices)),
            'mean': float(np.mean(prices)),
            'median': float(np.median(prices)),
            'std': float(np.std(prices)),
            f'{self.PERCENTILE_MIN}th_percentile': float(np.percentile(prices, self.PERCENTILE_MIN)),
            f'{self.PERCENTILE_MAX}th_percentile': float(np.percentile(prices, self.PERCENTILE_MAX)),
            'price_shift': float(self.price_shift),
        }

    def plot_price_histogram(self, num_bins=50, save_path=None, use_original=False):
        if use_original and self.original_prices is not None:
            prices = np.asarray(self.original_prices, dtype=np.float32)
            price_type = "Original"
        elif self.external_prices is not None:
            prices = np.asarray(self.external_prices, dtype=np.float32)
            price_type = "Shifted" if self.price_shift else "Original"
        else:
            prices = self._generated_prices_for_stats(24 * 7 * 52)
            price_type = "Generated"

        plt.figure(figsize=(10, 6))
        plt.hist(prices, bins=num_bins, edgecolor='black')
        plt.title(f'Distribution of Electricity Prices ({price_type})')
        plt.xlabel('Price ($/MWh)')
        plt.ylabel('Frequency')
        plt.axvline(np.percentile(prices, self.PERCENTILE_MIN), color='r', linestyle='dashed', linewidth=2, label=f'{self.PERCENTILE_MIN}th Percentile')
        plt.axvline(np.percentile(prices, self.PERCENTILE_MAX), color='g', linestyle='dashed', linewidth=2, label=f'{self.PERCENTILE_MAX}th Percentile')
        plt.axvline(np.mean(prices), color='b', linestyle='dashed', linewidth=2, label='Mean Price')
        plt.legend()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()
