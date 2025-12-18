from dataclasses import dataclass
import numpy as np

@dataclass
class Weights:
    efficiency_weight: float
    price_weight: float
    idle_weight: float
    job_age_weight: float
    drop_weight: float

    def __str__(self):
        return f"Weights: efficiency={self.efficiency_weight}, price={self.price_weight}, idle={self.idle_weight}, job_age={self.job_age_weight}, drop={self.drop_weight}. sum={self.sum():.2f}"

    def sum(self):
        return round(np.sum([self.efficiency_weight, self.price_weight, self.idle_weight, self.job_age_weight, self.drop_weight]), 2)
