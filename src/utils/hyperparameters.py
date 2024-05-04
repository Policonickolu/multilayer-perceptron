from dataclasses import dataclass, field
from typing import List
import numpy as np

@dataclass
class Hyperparameters:
    layers: List[int]
    activations: List[str]
    num_classes: int
    shuffle: bool = True
    epochs: int = 100
    learning_rate: float = 0.01
    iteration_type: str = 'batch'
    batch_size: int = 8
    optimizer: str = 'gradient_descent'
    initializer: str = 'he'
    momentum: float = 0.9
    patience: int = 10
    decay: List[float] = field(default_factory=lambda: [0.9, 0.99])
