import torch
from dimabsa.model import SimpleClassifier

def test_forward_shape():
    model = SimpleClassifier(16, 32, 3)
    out = model(torch.randn(2, 16))
    assert out.shape == (2, 3)
