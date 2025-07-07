# [NeurIPS 2024] Training for Stable Explanation for Free

Official implementation of **R2ET**, introduced in the NeurIPS 2024 paper _“Training for Stable Explanation for Free”_ by Chen _et al._  [Link](https://proceedings.neurips.cc/paper_files/paper/2024/hash/0626822954674a06ccd9c234e3f0d572-Abstract-Conference.html).

## 📘 Overview

R2ET empowers machine learning models with **stable and faithful explanations**, mitigating malicious manipulation of feature saliency without incurring heavy adversarial training costs. Key innovations include:

- A novel **ranking-based stability metric** ("explanation thickness") measuring top‑k salient feature consistency. 
- A **regularization-driven training algorithm** with theoretical guarantees that explanations remain stable across perturbations. 
- Strong **provable links** to multi‑objective optimization and certified robustness—all while avoiding expensive adversarial loops. 

Extensive experiments across diverse modalities and architectures highlight R2ET’s effectiveness against stealthy attacks and its generalization across explanation methods. 



## ⏳ What is included

- 🧪 Preprocessed datasets and notebooks showcasing usage
  
  pretrain.py (optional)
  
- 📊 Benchmarking against baseline robustness methods
  
  batch_retrain.py
  
- 📂 Codes for training and evaluation
  
  eval_retrain.py



## 🎯 Why R2ET Matters

1. **Trustworthy AI deployments** – ensures model explanations cannot be trivially disrupted.  
2. **Efficient training** – no need for adversarial sample generation loops.  
3. **Theoretically grounded** – certified stability with mathematical guarantees.



## 🧩 Get Involved

- ⭐ Star the repo to show support!
- 🐞 Submit issues if you spot bugs or need help.
- 🤝 Contributions welcome—especially with code, docs, or CI pipelines.



## 📄 Citation

If you find R2ET useful, please cite:

```bibtex
@inproceedings{chen2024training,
  author    = {Chen, Chao and Guo, Chenghua and Chen, Rufeng and Ma, Guixiang and Zeng, Ming and Liao, Xiangwen and Zhang, Xi and Xie, Sihong},
  title     = {Training for Stable Explanation for Free},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  volume    = {37},
  year      = {2024},
  pages     = {3421--3457},
}
