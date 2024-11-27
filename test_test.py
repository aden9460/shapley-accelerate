import unittest
import torch
from torch.utils.data import DataLoader, TensorDataset

import torch.nn as nn
from test import Shapley_accelereated  # Replace with the actual module name

class TestShapleyAccelerated(unittest.TestCase):
    def setUp(self):
        # Create a simple model
        self.model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 2)
        )

        # Create a simple dataset
        x = torch.randn(100, 10)
        y = torch.randint(0, 2, (100,))
        dataset = TensorDataset(x, y)
        self.data_loader = DataLoader(dataset, batch_size=10)

        # Initialize Shapley_accelereated
        self.shapley_accel = Shapley_accelereated(self.model, self.data_loader)

    def test_get_second_order_grad(self):
        # Test get_second_order_grad with the first layer
        layer = self.model[0]
        grads = self.shapley_accel.get_second_order_grad(layer)

        # Check if the returned gradients have the expected shape
        self.assertEqual(grads.shape, layer.weight.shape)

        # Check if the gradients are non-negative
        self.assertTrue(torch.all(grads >= 0))

if __name__ == '__main__':
    unittest.main()