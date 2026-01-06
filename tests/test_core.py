import unittest
import torch
import torch.nn as nn
import os
from ebf.core import EBFAdapter, ModelWrapper

class MockLlamaMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # Architecture: Up(Gate(x)) -> SiLU -> Down(x)
        self.gate_proj = nn.Linear(32, 128, bias=False) 
        self.down_proj = nn.Linear(128, 32, bias=False) # Target

class MockLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = MockLlamaMLP()

class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([MockLayer() for _ in range(3)])

class Wrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = MockModel()

class TestEBF(unittest.TestCase):
    def setUp(self):
        self.model = Wrapper()
        self.wrapper = ModelWrapper(self.model)
        self.adapter = EBFAdapter(self.wrapper)

    def test_srs_calibration(self):
        # Synthetic data: Layer 1 has high variance signal
        task_acts = {1: torch.randn(10, 32) + 10.0} 
        base_acts = {1: torch.randn(10, 32)}
        self.adapter.calibrate(task_acts, base_acts, threshold=0.1)
        self.assertIn(1, self.adapter.adapters)

    def test_fold_output_projection(self):
        # We target down_proj bias.
        vector = torch.ones(32)
        self.adapter.adapters[1] = vector
        
        layer = self.model.model.layers[1].mlp.down_proj
        self.assertIsNone(layer.bias)
        
        # Fold
        scale = 2.0
        self.adapter.fold_all(scale=scale)
        
        # Expect bias = vector * scale = 2.0
        self.assertIsNotNone(layer.bias)
        self.assertTrue(torch.allclose(layer.bias, torch.ones(32) * 2.0))

    def test_persistence(self):
        self.adapter.adapters[0] = torch.randn(32)
        self.adapter.save("temp_test.ebf")
        
        # New adapter instance
        new_adapter = EBFAdapter(self.wrapper)
        new_adapter.load("temp_test.ebf")
        
        self.assertTrue(torch.allclose(new_adapter.adapters[0], self.adapter.adapters[0]))
        os.remove("temp_test.ebf")

if __name__ == '__main__':
    unittest.main()