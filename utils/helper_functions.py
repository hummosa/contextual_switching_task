#@title A few helper functions
import numpy as np

def loss_stopped(losses, threshold=1e-6, num_epochs=10):
    """
    Check if the loss has stopped changing by a certain threshold
    over a specified number of epochs.

    Args:
        losses (list): List of losses over epochs.
        threshold (float): The threshold for change in loss.
        num_epochs (int): Number of recent epochs to consider for checking.

    Returns:
        bool: True if the loss has stopped changing, False otherwise.
    """
    if len(losses) < num_epochs:
        return False
    recent_losses = losses[-num_epochs:]
    max_change = max(recent_losses) - min(recent_losses)
    if max_change <= threshold:
        return True
    return False

def find_nearest(array, value):
    """
    Find the nearest value in an arry to some to query value.
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def moving_average(x, winsize):
    """
    Compute a simple moving average with a sliding windo of length `winsize`.
    """
    return np.convolve(x, np.ones(winsize), 'same') / winsize
#@title Helper class for computing online mean and variance
class OnlineStatsEMA:
    def __init__(self, alpha=0.02):
        self.alpha = alpha  # Smoothing factor
        self.ema_mean = None  # Initialize EMA mean as None to handle the first observation
        self.ema_variance = None  # Initialize EMA variance as None to handle the first observation

    def update(self, x):
        if self.ema_mean is None:
            self.ema_mean = x  # For the first observation, set EMA mean = x
            self.ema_variance = 0.0  # Initialize variance as 0 for the first observation
        else:
            # Update the EMA mean using EMA formula
            self.ema_mean = (1 - self.alpha) * self.ema_mean + self.alpha * x
            # Update the EMA variance using EMA formula
            variance = (x - self.ema_mean)**2
            if self.ema_variance is None:
                self.ema_variance = variance
            else:
                self.ema_variance = (1 - self.alpha) * self.ema_variance + self.alpha * variance

    def get_variance(self):
        return self.ema_variance

    def get_mean(self):
        return self.ema_mean

# # Example usage
# # adjust alpha to control the degree of "forgetting"
# stats = OnlineStatsEMA() # lower alpha gives more weight to older observations
# data_points = [2, 4, 4, 4, 5, 5, 7, 9]

# print("OnlineStatsEMA", end="\n\n")
# for x in data_points:
#     stats.update(x)
#     print(f"EMA Mean: {stats.get_mean():.3f}")
#     print(f"EMA Variance: {stats.get_variance():.3f}", end=f"\n{40*'~'}\n")
