# dimabsa-csce5290
# DimABSA Course Project
Course project for Natural Language Processing for the implementation of Dimensional Aspect-Based Sentiment Analysis (DimABSA) subtasks DimASR and DimASTE using PyTorch and Transformers.
This repository contains the course implementation of **Dimensional Aspect-Based Sentiment Analysis (DimABSA)**, developed as part of the Natural Language Processing term project by **Bob Jack** and **Amrit Adhikari**.

The project focuses on two SemEval 2026 subtasks:
1. **DimASR (Aspect Sentiment Regression):** Predicting continuous *valence–arousal (VA)* scores for given aspects within text.
2. **DimASTE (Aspect Sentiment Triplet Extraction):** Extracting *(Aspect, Opinion, VA)* triplets from text.

We use **PyTorch** and **Hugging Face Transformers** to train and evaluate models on the official **English datasets** provided by the [DimABSA2026](https://github.com/DimABSA/DimABSA2026) task organizers.  

---

### **Project Overview**
- **Framework:** PyTorch  
- **Language:** English (Restaurant domain primary)  
- **Tasks:** DimASR, DimASTE  
- **Evaluation:** RMSE (DimASR), Continuous F1 (DimASTE)  
- **Data Format:** JSON Lines (`.jsonl`)  

---

### **Repository Structure**

```
dimabsa-course/
│
├── data/ # Datasets (ignored in git)
├── src/ # Core model and training scripts
├── notebooks/ # Colab notebooks for experimentation
├── experiments/ # Saved configs, model notes
├── outputs/ # Predictions, logs, metrics (ignored in git)
│
├── LICENSE
├── .gitignore
└── README.md
```

---

### **Team**
- **Bob Jack** - Model development, environment setup, evaluation  
- **Amrit Adhikari** - Data exploration, preprocessing, baseline modeling  

---

### **Acknowledgment**
This work is based on the **SemEval 2026 DimABSA task**, organized by Liang-Chih Yu et al. All datasets and evaluation criteria follow the official competition guidelines.

### **Environment Setup**
This project was done working along side a notebook. Most of the code was done outside of the notebook. Here is the setup for a Mac. We did not develop on Windows machines.

Clone the project
```
git clone git@github.com:bobjack1313/dimabsa-csce5290.git
cd dimabsa-csce5290
```
Use homebrew for setup (Mac), make sure it's up to date
Apple Silicon (arm64)
```eval "$(/opt/homebrew/bin/brew shellenv)"```
Intel (x86_64)
```eval "$(/usr/local/bin/brew shellenv)" ```

Install Miniforge (Conda distribution for macOS)
```brew install --cask miniforge```

Initialize Conda 
Apple Silicon
```/opt/homebrew/bin/conda init bash```
Intel
```/usr/local/bin/conda init bash```

```exec $SHELL -l```

Gets rid of (base) when not in ENV -- Shut off autoactivate
```conda config --set auto_activate_base false```

It is best to close the terminal and start new shell

Create Project environment
```
conda create -y -n nlp5290 python=3.11
conda activate nlp5290
conda info --envs
python -V
```
This allows existing notebook to run If we need to use this
```
pip install jupyter ipykernel
python -m ipykernel install --user --name nlp5290 --display-name "Python (nlp5290)"
```

Will run notebooks (Only when you need to run it)
```jupyter lab```

Install Pytorch
```conda install -y pytorch torchvision torchaudio -c pytorch```

Verify
```
python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("MPS available:", torch.backends.mps.is_available())
PY
```

Install project editable and dev dependancies
```pip install -e ".[dev]"```

Run tests
```
pytest -q                      # should pass
python scripts/train.py --epochs 3
python scripts/eval.py --ckpt experiments/checkpoints/simple.pt
```
If train.py prints a few epochs and saves experiments/checkpoints/simple.pt, you’re good.

Team Workflow
```
conda activate nlp5290
```
 - Develop in modules
 - Put real code in src/dimabsa/ (e.g., dataset.py, model.py, train_loop.py).
 - Keep scripts/ for CLIs that call into the modules (e.g., scripts/train.py, scripts/eval.py).

Run training/eval from CLI
```
python scripts/train.py --epochs 3
python scripts/eval.py --ckpt experiments/checkpoints/simple.pt
```
Once the real dataset is wired in, this will change to:
```
python scripts/train.py --train data/train.jsonl --valid data/valid.jsonl --epochs 5 --batch-size 32
```

For continual testing
```pytest```

Additional project setup Added to toml, so I dont know if this will be needed manually
```pip install transformers datasets scikit-learn matplotlib```

Downloads and tests data
```
python scripts/download_data.py --hf yelp_polarity --train-split train --valid-split test --skip-test

```
Direct download option
```
python scripts/download_data.py --url https://example.com/dataset.zip
```

To get forked dataset for this project
```
git submodule update --init --recursive
```
