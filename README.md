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


---

### **Team**
- **Bob Jack** - Model development, environment setup, evaluation  
- **Amrit Adhikari** - Data exploration, preprocessing, baseline modeling  

---

### **Acknowledgment**
This work is based on the **SemEval 2026 DimABSA task**, organized by Liang-Chih Yu et al. All datasets and evaluation criteria follow the official competition guidelines.

