import argparse
from pathlib import Path
from dimabsa.train_loop import train_one_run

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--input-dim", type=int, default=16)
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--num-classes", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out-dir", type=Path, default=Path("experiments/checkpoints"))
    return p.parse_args()

def main():
    a = parse_args()
    train_one_run(
        epochs=a.epochs,
        input_dim=a.input_dim,
        hidden_dim=a.hidden_dim,
        num_classes=a.num_classes,
        lr=a.lr,
        out_dir=a.out_dir,
    )

if __name__ == "__main__":
    main()
