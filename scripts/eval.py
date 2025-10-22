import argparse
from pathlib import Path
import torch
from dimabsa.model import SimpleClassifier

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=Path("experiments/checkpoints/simple.pt"))
    ap.add_argument("--input-dim", type=int, default=16)
    ap.add_argument("--hidden-dim", type=int, default=32)
    ap.add_argument("--num-classes", type=int, default=3)
    args = ap.parse_args()

    model = SimpleClassifier(args.input_dim, args.hidden_dim, args.num_classes)
    # Safe weights-only load (PyTorch 2.5+)
    state = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        x = torch.randn(4, args.input_dim)
        out = model(x)

    print("logits shape:", tuple(out.shape))

if __name__ == "__main__":
    main()

