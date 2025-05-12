# WSMC: Wake-Sleep Memory Consolidation for Continual Relation Extraction

Continual Relation Extraction (CRE) model has gained significant attention due to its ability to adapt to new relations without frequent retraining. However, existing methods still face challenges such as overfitting and representation bias. Inspired by the “wake-sleep” memory consolidation process of the human brain, we propose a \textbf{W}ake-\textbf{S}leep \textbf{M}emory \textbf{C}onsolidation (WSMC) framework to systematically address these issues. During the wake phase, the model simulates the brain's information processing mechanism, quickly encoding new relations and storing them in short-term memory. We also introduce the Experience Iterative Learning (EIL) approach. This approach dynamically adjusts the distribution of relation samples to correct the model's representation bias and enhances memory stability through experience replay. During the sleep phase, the model consolidates existing knowledge through long-term memory replay and relation prototype contrastive network. Moreover, the framework generates diverse dream data from existing memory sets, thereby increasing the diversity of the training data and improving the model's generalization capability. Experimental results show that WSMC significantly outperforms other CRE baseline methods on FewRel and TACRED datasets, demonstrating its superior performance compared to baseline methods.

---

## Environment Setup

Our implementation is based on **Python 3.9.19** and **PyTorch 2.3.0** (CUDA 12.x).
To upgrade to **PyTorch 2.3.0**, please follow the official installation guide from [PyTorch](https://pytorch.org/), or execute the following command:

```bash
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia
```

Subsequently, install the remaining dependencies via:

```bash
pip install -r requirements.txt
```

---

## Datasets

We evaluate WSMC on two widely-used benchmarks: **FewRel** and **TACRED**.
Pre-processed splits and task order files are available in the corresponding subdirectories under the `data/` folder.

---

## Reproducing Experiments

To reproduce the main experimental results, please run:

```bash
bash FewRel.sh
bash tacred.sh
```

All experiments are conducted on a single **NVIDIA RTX 4090D GPU** with **24 GB** of memory.
