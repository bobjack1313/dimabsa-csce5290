# import argparse
# from pathlib import Path
# import torch
# from dimabsa.model import SimpleClassifier

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--ckpt", type=Path, default=Path("experiments/checkpoints/simple.pt"))
#     ap.add_argument("--input-dim", type=int, default=16)
#     ap.add_argument("--hidden-dim", type=int, default=32)
#     ap.add_argument("--num-classes", type=int, default=3)
#     args = ap.parse_args()

#     model = SimpleClassifier(args.input_dim, args.hidden_dim, args.num_classes)
#     # Safe weights-only load (PyTorch 2.5+)
#     state = torch.load(args.ckpt, map_location="cpu", weights_only=True)
#     model.load_state_dict(state)
#     model.eval()

#     with torch.no_grad():
#         x = torch.randn(4, args.input_dim)
#         out = model(x)

#     print("logits shape:", tuple(out.shape))

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
"""
Evaluate a fine-tuned BERT model on Task 1 (DimASR).
Loads model checkpoint from experiments/checkpoints/task1/bert_final
and evaluates against data/processed/task1/valid.jsonl
"""

# import argparse
# from pathlib import Path
# from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
# from datasets import load_dataset

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--model-dir", type=Path, default=Path("experiments/checkpoints/task1/bert_final"))
#     ap.add_argument("--data-dir", type=Path, default=Path("data/processed/task1"))
#     ap.add_argument("--batch-size", type=int, default=8)
#     args = ap.parse_args()

#     # Load tokenizer & model
#     tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
#     model = BertForSequenceClassification.from_pretrained(args.model_dir)

#     # Load validation data
#     data_files = {"valid": str(args.data_dir / "valid.jsonl")}
#     ds = load_dataset("json", data_files=data_files)

#     def preprocess(batch):
#         return tokenizer(batch["Text"], padding="max_length", truncation=True, max_length=128)
#     ds = ds.map(preprocess, batched=True)

#     # Setup evaluation
#     eval_args = TrainingArguments(
#         output_dir="eval_output",
#         per_device_eval_batch_size=args.batch_size,
#         report_to="none"
#     )

#     trainer = Trainer(model=model, args=eval_args, eval_dataset=ds["valid"], tokenizer=tokenizer)

#     results = trainer.evaluate()
#     print("\nEvaluation Results:")
#     for k, v in results.items():
#         print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

# if __name__ == "__main__":
#     main()




    #!/usr/bin/env python3
"""
Evaluate a fine-tuned BERT model on DimABSA Task 1 (DimASR).
Computes RMSE and Pearson correlation (PCC) for valence and arousal scores.
"""

import argparse
import numpy as np
from pathlib import Path
from transformers import BertForSequenceClassification, BertTokenizer
from datasets import load_dataset
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, default=Path("experiments/checkpoints/task1/bert_final"))
    ap.add_argument("--data-dir", type=Path, default=Path("data/processed/task1"))
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    # Load model and tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained(args.model_dir)
    model.eval()

    # Load dataset
    data_files = {"valid": str(args.data_dir / "valid.jsonl")}
    ds = load_dataset("json", data_files=data_files)["valid"]

    # Parse VA values (split "6.75#6.38" -> floats)
    va_data = []
    for sample in ds:
        if "Quadruplet" in sample:
            for q in sample["Quadruplet"]:
                try:
                    v, a = map(float, q["VA"].split("#"))
                    va_data.append((sample["Text"], v, a))
                except Exception:
                    continue
        elif "VA" in sample:
            try:
                v, a = map(float, sample["VA"].split("#"))
                va_data.append((sample["Text"], v, a))
            except Exception:
                continue
        else:
            # No gold VA, skip
            continue

    if not va_data:
        print("No valid VA data found in validation file.")
        return

    texts, v_gold, a_gold = zip(*va_data)

    # Tokenize
    inputs = tokenizer(list(texts), padding=True, truncation=True, return_tensors="pt", max_length=128)

    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits.squeeze().cpu().numpy()

    # If model output is 2D (batch, 2) treat columns as valence and arousal
    if logits.ndim == 2 and logits.shape[1] >= 2:
        v_pred, a_pred = logits[:, 0], logits[:, 1]
    else:
        # fallback (single-dim model)
        v_pred = a_pred = logits if logits.ndim == 1 else logits[:, 0]

    # Compute metrics
    v_rmse = mean_squared_error(v_gold, v_pred, squared=False)
    a_rmse = mean_squared_error(a_gold, a_pred, squared=False)
    v_pcc, _ = pearsonr(v_gold, v_pred)
    a_pcc, _ = pearsonr(a_gold, a_pred)

    print("\nEvaluation Metrics:")
    print(f"Valence RMSE: {v_rmse:.4f}")
    print(f"Arousal RMSE: {a_rmse:.4f}")
    print(f"Valence PCC:  {v_pcc:.4f}")
    print(f"Arousal PCC:  {a_pcc:.4f}")
    print(f"Samples evaluated: {len(v_gold)}")

if __name__ == "__main__":
    main()

