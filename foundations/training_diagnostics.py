import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.
        stats = []
        with torch.no_grad():
            out = x
            for layer in model:
                
                out = layer(out)
                if isinstance(layer, nn.Linear):
                    stats.append({
                        'mean': round(out.mean().item(), 4),
                        'std': round(out.std().item(), 4),
                        'dead_fraction': round((out <= 0).all(dim=0).float().mean().item(), 4),
                    })
        return stats
        

    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.
        model.zero_grad()
        loss = nn.MSELoss()(model(x), y)
        loss.backward()

        stats = []
        for layer in model:
            if isinstance(layer, nn.Linear):
                grad = layer.weight.grad
                stats.append({
                    'mean': round(grad.mean().item(), 4),
                    'std': round(grad.std().item(), 4),
                    'norm': round(grad.norm().item(), 4),
                })
        return stats

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        # Check in priority order (see problem description for thresholds)
        if any(i['dead_fraction'] > 0.5 for i in activation_stats):
            return 'dead_neurons'
        if any(i['norm'] > 1000 for i in gradient_stats):
            return 'exploding_gradients'
        last = gradient_stats[-1]
        if last['norm'] < 1e-5:
            return 'vanishing_gradients'
        if any(i['std'] < 0.1 for i in activation_stats):
            return 'vanishing_gradients'
        if any(i['std'] > 10.0 for i in activation_stats):
            return 'exploding_gradients'   
        return 'healthy'
