import numpy as np
from typing import List


class Solution:
    def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:
        # Implement RMS Normalization (similar to LayerNorm but without mean centering or beta)
        # Normalize x, then scale by gamma
        # Return result rounded to 4 decimal places as a list
        def rms(x):
            return np.sqrt((1/len(x)) * sum(np.square(x)) + eps)
        
        x_hat = x / rms(x)
        y = gamma * x_hat

        return np.round(y, 4)
