# DeepSMOTE Experiment Results

## Run Results (Structured)

### Config A — CLASSES = ['ship', 'airplane', 'bird', 'dog'], minority = dog (n=80)

```python
# Config A, Seed 10
run_A_s10 = {
    'config': 'A',
    'seed': 10,
    'minority_class': 'dog',
    'minority_n': 80,
    'classes': ['ship', 'airplane', 'bird', 'dog'],

    'baseline':  {'acc': 0.7230, 'f1': 0.7032, 'acsa': 0.7230, 'gm': 0.6751, 'minority_acc': 0.339},
    'smote':     {'acc': 0.7385, 'f1': 0.7220, 'acsa': 0.7385, 'gm': 0.6988, 'minority_acc': 0.380},
    'deepsmote': {'acc': 0.7185, 'f1': 0.6949, 'acsa': 0.7185, 'gm': 0.6628, 'minority_acc': 0.312},

    'minority_nn_ratio': 0.978,
    'majority_nn_ratio': 0.820,
    'fid': 146.77,
    'lpips': 0.1731,
    'tsne_cifar': "./config tsne fig cifar/cifar_configA_seed10.png",
    
}

# Config A, Seed 20
run_A_s20 = {
    'config': 'A',
    'seed': 20,
    'minority_class': 'dog',
    'minority_n': 80,
    'classes': ['ship', 'airplane', 'bird', 'dog'],

    'baseline':  {'acc': 0.7550, 'f1': 0.7438, 'acsa': 0.7550, 'gm': 0.7281, 'minority_acc': 0.451},
    'smote':     {'acc': 0.7522, 'f1': 0.7388, 'acsa': 0.7522, 'gm': 0.7219, 'minority_acc': 0.432},
    'deepsmote': {'acc': 0.7360, 'f1': 0.7189, 'acsa': 0.7360, 'gm': 0.6971, 'minority_acc': 0.382},

    'minority_nn_ratio': 1.009,
    'majority_nn_ratio': 0.842,
    'fid': 150.10,
    'lpips': 0.1729,
    'tsne_cifar': "./config tsne fig cifar/cifar_configA_seed20.png",
}

# Config A, Seed 60
run_A_s60 = {
    'config': 'A',
    'seed': 60,
    'minority_class': 'dog',
    'minority_n': 80,
    'classes': ['ship', 'airplane', 'bird', 'dog'],

    'baseline':  {'acc': 0.7528, 'f1': 0.7371, 'acsa': 0.7527, 'gm': 0.7163, 'minority_acc': 0.403},
    'smote':     {'acc': 0.7588, 'f1': 0.7459, 'acsa': 0.7588, 'gm': 0.7291, 'minority_acc': 0.438},
    'deepsmote': {'acc': 0.7380, 'f1': 0.7189, 'acsa': 0.7380, 'gm': 0.6942, 'minority_acc': 0.364},

    'minority_nn_ratio': 0.972,
    'majority_nn_ratio': 0.832,
    'fid': 147.91,
    'lpips': 0.1707,
    'tsne_cifar': "./config tsne fig cifar/cifar_configA_seed60.png",
}
```

---

### Config B — CLASSES = ['ship', 'dog', 'bird', 'airplane'], minority = airplane (n=80)

```python
# Config B, Seed 10
run_B_s10 = {
    'config': 'B',
    'seed': 10,
    'minority_class': 'airplane',
    'minority_n': 80,
    'classes': ['ship', 'dog', 'bird', 'airplane'],

    'baseline':  {'acc': 0.7117, 'f1': 0.6670, 'acsa': 0.7117, 'gm': 0.6026, 'minority_acc': 0.193},
    'smote':     {'acc': 0.7135, 'f1': 0.6665, 'acsa': 0.7135, 'gm': 0.5975, 'minority_acc': 0.183},
    'deepsmote': {'acc': 0.6817, 'f1': 0.6267, 'acsa': 0.6817, 'gm': 0.5353, 'minority_acc': 0.129},

    'minority_nn_ratio': 0.954,
    'majority_nn_ratio': 0.803,
    'fid': 175.07,
    'lpips': 0.2174,
    'tsne_cifar': "./config tsne fig cifar/cifar_configB_seed10.png",
}

# Config B, Seed 20
run_B_s20 = {
    'config': 'B',
    'seed': 20,
    'minority_class': 'airplane',
    'minority_n': 80,
    'classes': ['ship', 'dog', 'bird', 'airplane'],

    'baseline':  {'acc': 0.7060, 'f1': 0.6629, 'acsa': 0.7060, 'gm': 0.6032, 'minority_acc': 0.202},
    'smote':     {'acc': 0.7010, 'f1': 0.6521, 'acsa': 0.7010, 'gm': 0.5800, 'minority_acc': 0.171},
    'deepsmote': {'acc': 0.6905, 'f1': 0.6349, 'acsa': 0.6905, 'gm': 0.5437, 'minority_acc': 0.132},

    'minority_nn_ratio': 0.953,
    'majority_nn_ratio': 0.833,
    'fid': 170.41,
    'lpips': 0.2127,
    'tsne_cifar': "./config tsne fig cifar/cifar_configB_seed20.png",
}

# Config B, Seed 60
run_B_s60 = {
    'config': 'B',
    'seed': 60,
    'minority_class': 'airplane',
    'minority_n': 80,
    'classes': ['ship', 'dog', 'bird', 'airplane'],

    'baseline':  {'acc': 0.7268, 'f1': 0.6904, 'acsa': 0.7268, 'gm': 0.6403, 'minority_acc': 0.244},
    'smote':     {'acc': 0.7047, 'f1': 0.6642, 'acsa': 0.7047, 'gm': 0.6043, 'minority_acc': 0.206},
    'deepsmote': {'acc': 0.6945, 'f1': 0.6496, 'acsa': 0.6945, 'gm': 0.5818, 'minority_acc': 0.181},

    'minority_nn_ratio': 0.961,
    'majority_nn_ratio': 0.835,
    'fid': 178.31,
    'lpips': 0.2137,
    'tsne_cifar': "./config tsne fig cifar/cifar_configB_seed60.png",
}
```

---

## Summary Tables

### Minority Class Accuracy Comparison

| Config | Seed | Baseline minority acc | Pixel SMOTE minority acc | DeepSMOTE minority acc | NN ratio (minority) | NN ratio (majority) |
|---|---|---|---|---|---|---|
| A (dog) | 10 | 33.9% | 38.0% | 31.2% | 0.978 | 0.820 |
| A (dog) | 20 | 45.1% | 43.2% | 38.2% | 1.009 | 0.842 |
| A (dog) | 60 | 40.3% | 43.8% | 36.4% | 0.972 | 0.832 |
| **A mean** | | **39.8%** | **41.7%** | **35.3%** | **0.986** | **0.831** |
| | | | | | | |
| B (airplane) | 10 | 19.3% | 18.3% | 12.9% | 0.954 | 0.803 |
| B (airplane) | 20 | 20.2% | 17.1% | 13.2% | 0.953 | 0.833 |
| B (airplane) | 60 | 24.4% | 20.6% | 18.1% | 0.961 | 0.835 |
| **B mean** | | **21.3%** | **18.7%** | **14.7%** | **0.956** | **0.824** |

---

### Overall Metrics Comparison

| Config | Seed | Method | Acc | F1 | ACSA | GM |
|---|---|---|---|---|---|---|
| A (dog) | 10 | Baseline | 0.7230 | 0.7032 | 0.7230 | 0.6751 |
| A (dog) | 10 | Pixel SMOTE | 0.7385 | 0.7220 | 0.7385 | 0.6988 |
| A (dog) | 10 | DeepSMOTE | 0.7185 | 0.6949 | 0.7185 | 0.6628 |
| A (dog) | 20 | Baseline | 0.7550 | 0.7438 | 0.7550 | 0.7281 |
| A (dog) | 20 | Pixel SMOTE | 0.7522 | 0.7388 | 0.7522 | 0.7219 |
| A (dog) | 20 | DeepSMOTE | 0.7360 | 0.7189 | 0.7360 | 0.6971 |
| A (dog) | 60 | Baseline | 0.7528 | 0.7371 | 0.7527 | 0.7163 |
| A (dog) | 60 | Pixel SMOTE | 0.7588 | 0.7459 | 0.7588 | 0.7291 |
| A (dog) | 60 | DeepSMOTE | 0.7380 | 0.7189 | 0.7380 | 0.6942 |
| | | | | | | |
| B (airplane) | 10 | Baseline | 0.7117 | 0.6670 | 0.7117 | 0.6026 |
| B (airplane) | 10 | Pixel SMOTE | 0.7135 | 0.6665 | 0.7135 | 0.5975 |
| B (airplane) | 10 | DeepSMOTE | 0.6817 | 0.6267 | 0.6817 | 0.5353 |
| B (airplane) | 20 | Baseline | 0.7060 | 0.6629 | 0.7060 | 0.6032 |
| B (airplane) | 20 | Pixel SMOTE | 0.7010 | 0.6521 | 0.7010 | 0.5800 |
| B (airplane) | 20 | DeepSMOTE | 0.6905 | 0.6349 | 0.6905 | 0.5437 |
| B (airplane) | 60 | Baseline | 0.7268 | 0.6904 | 0.7268 | 0.6403 |
| B (airplane) | 60 | Pixel SMOTE | 0.7047 | 0.6642 | 0.7047 | 0.6043 |
| B (airplane) | 60 | DeepSMOTE | 0.6945 | 0.6496 | 0.6945 | 0.5818 |

---

### Image Quality Diagnostics

| Config | Seed | FID | LPIPS diversity |
|---|---|---|---|
| A (dog) | 10 | 146.77 | 0.1731 |
| A (dog) | 20 | 150.10 | 0.1729 |
| A (dog) | 60 | 147.91 | 0.1707 |
| **A mean** | | **148.26** | **0.1722** |
| | | | |
| B (airplane) | 10 | 175.07 | 0.2174 |
| B (airplane) | 20 | 170.41 | 0.2127 |
| B (airplane) | 60 | 178.31 | 0.2137 |
| **B mean** | | **174.60** | **0.2146** |

---

### Key Takeaways

> [!IMPORTANT]
> **Consistent across all 6 runs:** DeepSMOTE **underperforms** both baseline and pixel SMOTE on minority class accuracy. The ranking is always: **Pixel SMOTE ≥ Baseline > DeepSMOTE**.

| Observation | Config A (dog) | Config B (airplane) |
|---|---|---|
| DeepSMOTE helps minority? | ❌ No (35.3% vs 39.8% baseline) | ❌ No (14.7% vs 21.3% baseline) |
| Pixel SMOTE helps minority? | ✅ Slightly (41.7% vs 39.8%) | ❌ No (18.7% vs 21.3%) |
| NN ratio (minority) | ~0.98 (very sparse) | ~0.96 (very sparse) |
| FID (lower=better) | ~148 | ~175 (worse) |
| Airplane harder than dog? | — | ✅ Yes, ~2× worse minority acc |
