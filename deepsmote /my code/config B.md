```python
class Config:
    ROOT_DIR = '/kaggle/input/datasets/ayush1220/cifar10/cifar10'
    
    CLASSES = ['ship', 'dog', 'bird', 'airplane']

    TRAIN_SAMPLES = [2000, 1000, 600, 80]
    
    LATENT_DIM = 600   # ← fix this back
    DIM_H = 64
    IMG_SIZE = 32
    
    AE_EPOCHS = 100
    CLF_EPOCHS = 50
    BATCH_SIZE = 128
    LR = 0.0002
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = Config()
```

---

# seed value 10
PyTorch: 2.10.0+cu128 | CUDA: True
Train samples: 3680 | Test samples: 4000
==================================================
PHASE 1: BASELINE (Imbalanced Data)
==================================================
100%
 50/50 [02:06<00:00,  2.48s/it]

--- Baseline (Imbalanced) Results ---
Accuracy: 0.7117 | F1: 0.6670 | ACSA: 0.7117 | GM: 0.6026
Per-class accuracy:
  ship: 0.9800 (n=1000)
  dog: 0.8940 (n=1000)
  bird: 0.7800 (n=1000)
  airplane: 0.1930 (n=1000)
Confusion matrix (rows=true, cols=pred): ['ship', 'dog', 'bird', 'airplane']
[[980   6  12   2]
 [ 24 894  81   1]
 [ 56 155 780   9]
 [513  58 236 193]]

==================================================
PHASE 1.5: TRADITIONAL SMOTE (Balanced Data)
==================================================
Applying traditional SMOTE on pixel space...
  Class 1: Generating 1000 samples via SMOTE...
  Class 2: Generating 1400 samples via SMOTE...
  Class 3: Generating 1920 samples via SMOTE...
Traditional SMOTE - Total samples: 8000
100%
 50/50 [01:19<00:00,  1.61s/it]

--- Traditional SMOTE (Balanced) Results ---
Accuracy: 0.7135 | F1: 0.6665 | ACSA: 0.7135 | GM: 0.5975
Per-class accuracy:
  ship: 0.9810 (n=1000)
  dog: 0.9100 (n=1000)
  bird: 0.7800 (n=1000)
  airplane: 0.1830 (n=1000)
Confusion matrix (rows=true, cols=pred): ['ship', 'dog', 'bird', 'airplane']
[[981   6  10   3]
 [ 23 910  66   1]
 [ 56 154 780  10]
 [513  56 248 183]]

==================================================
PHASE 2: DEEPSMOTE (Balancing Data)
==================================================
Preloading images for penalty loss sampling...
AE Epoch [10/100] Loss: 0.1027
AE Epoch [20/100] Loss: 0.0682
AE Epoch [30/100] Loss: 0.0503
AE Epoch [40/100] Loss: 0.0409
AE Epoch [50/100] Loss: 0.0344
AE Epoch [60/100] Loss: 0.0313
AE Epoch [70/100] Loss: 0.0270
AE Epoch [80/100] Loss: 0.0255
AE Epoch [90/100] Loss: 0.0215
AE Epoch [100/100] Loss: 0.0196
Generating synthetic images for ALL underrepresented classes...
Class 0: Has 2000 samples. Target is 2000. No generation needed.
Class 1: Generating 1000 synthetic samples...
Class 2: Generating 1400 synthetic samples...
Class 3: Generating 1920 synthetic samples...
Airplane (n=80):
  mean nearest-neighbor distance: 18.631
  mean latent vector norm: 19.536
  ratio: 0.954
Ship (n=2000):
  mean nearest-neighbor distance: 14.380
  mean latent vector norm: 17.918
  ratio: 0.803
Setting up [LPIPS] perceptual loss: trunk [alex], v[0.1], spatial [off]
/usr/local/lib/python3.12/dist-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
/usr/local/lib/python3.12/dist-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=AlexNet_Weights.IMAGENET1K_V1`. You can also use `weights=AlexNet_Weights.DEFAULT` to get the most up-to-date weights.
  warnings.warn(msg)
Loading model from: /usr/local/lib/python3.12/dist-packages/lpips/weights/v0.1/alex.pth
FID (real vs synthetic airplane): 175.07
LPIPS diversity among synthetic airplanes: 0.2174
Total synthetic images generated: 4320
Original dataset size: 3680
New balanced dataset size: 8000
==================================================
PHASE 3: DEEPSMOTE CLASSIFIER (Balanced Data)
==================================================
100%
 50/50 [01:22<00:00,  1.67s/it]

--- DeepSMOTE (Balanced) Results ---
Accuracy: 0.6817 | F1: 0.6267 | ACSA: 0.6817 | GM: 0.5353
Per-class accuracy:
  ship: 0.9790 (n=1000)
  dog: 0.8820 (n=1000)
  bird: 0.7370 (n=1000)
  airplane: 0.1290 (n=1000)
Confusion matrix (rows=true, cols=pred): ['ship', 'dog', 'bird', 'airplane']
[[979  10   8   3]
 [ 43 882  75   0]
 [ 81 180 737   2]
 [615  48 208 129]]

✅ COMPLETE! Compare Phase 1 and Phase 3 metrics.


---

# seed value: 20
PyTorch: 2.10.0+cu128 | CUDA: True
Train samples: 3680 | Test samples: 4000
==================================================
PHASE 1: BASELINE (Imbalanced Data)
==================================================
100%
 50/50 [02:06<00:00,  2.55s/it]

--- Baseline (Imbalanced) Results ---
Accuracy: 0.7060 | F1: 0.6629 | ACSA: 0.7060 | GM: 0.6032
Per-class accuracy:
  ship: 0.9770 (n=1000)
  dog: 0.8990 (n=1000)
  bird: 0.7460 (n=1000)
  airplane: 0.2020 (n=1000)
Confusion matrix (rows=true, cols=pred): ['ship', 'dog', 'bird', 'airplane']
[[977  11  10   2]
 [ 35 899  66   0]
 [ 57 190 746   7]
 [514  66 218 202]]

==================================================
PHASE 1.5: TRADITIONAL SMOTE (Balanced Data)
==================================================
Applying traditional SMOTE on pixel space...
  Class 1: Generating 1000 samples via SMOTE...
  Class 2: Generating 1400 samples via SMOTE...
  Class 3: Generating 1920 samples via SMOTE...
Traditional SMOTE - Total samples: 8000
100%
 50/50 [01:20<00:00,  1.64s/it]

--- Traditional SMOTE (Balanced) Results ---
Accuracy: 0.7010 | F1: 0.6521 | ACSA: 0.7010 | GM: 0.5800
Per-class accuracy:
  ship: 0.9800 (n=1000)
  dog: 0.9140 (n=1000)
  bird: 0.7390 (n=1000)
  airplane: 0.1710 (n=1000)
Confusion matrix (rows=true, cols=pred): ['ship', 'dog', 'bird', 'airplane']
[[980   9   9   2]
 [ 31 914  55   0]
 [ 67 188 739   6]
 [542  53 234 171]]

==================================================
PHASE 2: DEEPSMOTE (Balancing Data)
==================================================
Preloading images for penalty loss sampling...
AE Epoch [10/100] Loss: 0.1068
AE Epoch [20/100] Loss: 0.0662
AE Epoch [30/100] Loss: 0.0482
AE Epoch [40/100] Loss: 0.0401
AE Epoch [50/100] Loss: 0.0310
AE Epoch [60/100] Loss: 0.0287
AE Epoch [70/100] Loss: 0.0238
AE Epoch [80/100] Loss: 0.0229
AE Epoch [90/100] Loss: 0.0220
AE Epoch [100/100] Loss: 0.0194
Generating synthetic images for ALL underrepresented classes...
Class 0: Has 2000 samples. Target is 2000. No generation needed.
Class 1: Generating 1000 synthetic samples...
Class 2: Generating 1400 synthetic samples...
Class 3: Generating 1920 synthetic samples...
Airplane (n=80):
  mean nearest-neighbor distance: 18.375
  mean latent vector norm: 19.275
  ratio: 0.953
Ship (n=2000):
  mean nearest-neighbor distance: 14.966
  mean latent vector norm: 17.965
  ratio: 0.833
Setting up [LPIPS] perceptual loss: trunk [alex], v[0.1], spatial [off]
/usr/local/lib/python3.12/dist-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
/usr/local/lib/python3.12/dist-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=AlexNet_Weights.IMAGENET1K_V1`. You can also use `weights=AlexNet_Weights.DEFAULT` to get the most up-to-date weights.
  warnings.warn(msg)
Loading model from: /usr/local/lib/python3.12/dist-packages/lpips/weights/v0.1/alex.pth
FID (real vs synthetic airplane): 170.41
LPIPS diversity among synthetic airplanes: 0.2127
Total synthetic images generated: 4320
Original dataset size: 3680
New balanced dataset size: 8000
==================================================
PHASE 3: DEEPSMOTE CLASSIFIER (Balanced Data)
==================================================
100%
 50/50 [01:26<00:00,  1.73s/it]

--- DeepSMOTE (Balanced) Results ---
Accuracy: 0.6905 | F1: 0.6349 | ACSA: 0.6905 | GM: 0.5437
Per-class accuracy:
  ship: 0.9760 (n=1000)
  dog: 0.9020 (n=1000)
  bird: 0.7520 (n=1000)
  airplane: 0.1320 (n=1000)
Confusion matrix (rows=true, cols=pred): ['ship', 'dog', 'bird', 'airplane']
[[976   8  14   2]
 [ 40 902  58   0]
 [ 62 181 752   5]
 [597  49 222 132]]

✅ COMPLETE! Compare Phase 1 and Phase 3 metrics.

---


# seed value: 60
PyTorch: 2.10.0+cu128 | CUDA: True
Train samples: 3680 | Test samples: 4000
==================================================
PHASE 1: BASELINE (Imbalanced Data)
==================================================
100%
 50/50 [02:13<00:00,  2.83s/it]

--- Baseline (Imbalanced) Results ---
Accuracy: 0.7268 | F1: 0.6904 | ACSA: 0.7268 | GM: 0.6403
Per-class accuracy:
  ship: 0.9820 (n=1000)
  dog: 0.9110 (n=1000)
  bird: 0.7700 (n=1000)
  airplane: 0.2440 (n=1000)
Confusion matrix (rows=true, cols=pred): ['ship', 'dog', 'bird', 'airplane']
[[982  11   7   0]
 [ 24 911  64   1]
 [ 58 159 770  13]
 [507  56 193 244]]

==================================================
PHASE 1.5: TRADITIONAL SMOTE (Balanced Data)
==================================================
Applying traditional SMOTE on pixel space...
  Class 1: Generating 1000 samples via SMOTE...
  Class 2: Generating 1400 samples via SMOTE...
  Class 3: Generating 1920 samples via SMOTE...
Traditional SMOTE - Total samples: 8000
100%
 50/50 [01:21<00:00,  1.64s/it]

--- Traditional SMOTE (Balanced) Results ---
Accuracy: 0.7047 | F1: 0.6642 | ACSA: 0.7047 | GM: 0.6043
Per-class accuracy:
  ship: 0.9750 (n=1000)
  dog: 0.9020 (n=1000)
  bird: 0.7360 (n=1000)
  airplane: 0.2060 (n=1000)
Confusion matrix (rows=true, cols=pred): ['ship', 'dog', 'bird', 'airplane']
[[975  10   7   8]
 [ 29 902  68   1]
 [ 80 171 736  13]
 [565  54 175 206]]

==================================================
PHASE 2: DEEPSMOTE (Balancing Data)
==================================================
Preloading images for penalty loss sampling...
AE Epoch [10/100] Loss: 0.1041
AE Epoch [20/100] Loss: 0.0684
AE Epoch [30/100] Loss: 0.0482
AE Epoch [40/100] Loss: 0.0390
AE Epoch [50/100] Loss: 0.0317
AE Epoch [60/100] Loss: 0.0268
AE Epoch [70/100] Loss: 0.0261
AE Epoch [80/100] Loss: 0.0222
AE Epoch [90/100] Loss: 0.0192
AE Epoch [100/100] Loss: 0.0195
Generating synthetic images for ALL underrepresented classes...
Class 0: Has 2000 samples. Target is 2000. No generation needed.
Class 1: Generating 1000 synthetic samples...
Class 2: Generating 1400 synthetic samples...
Class 3: Generating 1920 synthetic samples...
Airplane (n=80):
  mean nearest-neighbor distance: 18.723
  mean latent vector norm: 19.474
  ratio: 0.961
Ship (n=2000):
  mean nearest-neighbor distance: 14.952
  mean latent vector norm: 17.916
  ratio: 0.835
Setting up [LPIPS] perceptual loss: trunk [alex], v[0.1], spatial [off]
/usr/local/lib/python3.12/dist-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
/usr/local/lib/python3.12/dist-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=AlexNet_Weights.IMAGENET1K_V1`. You can also use `weights=AlexNet_Weights.DEFAULT` to get the most up-to-date weights.
  warnings.warn(msg)
Loading model from: /usr/local/lib/python3.12/dist-packages/lpips/weights/v0.1/alex.pth
FID (real vs synthetic airplane): 178.31
LPIPS diversity among synthetic airplanes: 0.2137
Total synthetic images generated: 4320
Original dataset size: 3680
New balanced dataset size: 8000
==================================================
PHASE 3: DEEPSMOTE CLASSIFIER (Balanced Data)
==================================================
100%
 50/50 [01:23<00:00,  1.67s/it]

--- DeepSMOTE (Balanced) Results ---
Accuracy: 0.6945 | F1: 0.6496 | ACSA: 0.6945 | GM: 0.5818
Per-class accuracy:
  ship: 0.9770 (n=1000)
  dog: 0.9000 (n=1000)
  bird: 0.7200 (n=1000)
  airplane: 0.1810 (n=1000)
Confusion matrix (rows=true, cols=pred): ['ship', 'dog', 'bird', 'airplane']
[[977  10  11   2]
 [ 34 900  66   0]
 [ 72 190 720  18]
 [589  51 179 181]]

✅ COMPLETE! Compare Phase 1 and Phase 3 metrics.
---