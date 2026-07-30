
```python
class Config:
    ROOT_DIR = '/kaggle/input/datasets/ayush1220/cifar10/cifar10'
    
    CLASSES = ['ship', 'airplane', 'bird', 'dog']
    
    TRAIN_SAMPLES = [2000, 1000, 600, 80]
    
    LATENT_DIM = 600   
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
 50/50 [01:58<00:00,  2.26s/it]

--- Baseline (Imbalanced) Results ---
Accuracy: 0.7230 | F1: 0.7032 | ACSA: 0.7230 | GM: 0.6751
Per-class accuracy:
  ship: 0.9280 (n=1000)
  airplane: 0.8070 (n=1000)
  bird: 0.8180 (n=1000)
  dog: 0.3390 (n=1000)
Confusion matrix (rows=true, cols=pred): ['ship', 'airplane', 'bird', 'dog']
[[928  54  17   1]
 [133 807  60   0]
 [ 44 119 818  19]
 [ 88  91 482 339]]

==================================================
PHASE 1.5: TRADITIONAL SMOTE (Balanced Data)
==================================================
Applying traditional SMOTE on pixel space...
  Class 1: Generating 1000 samples via SMOTE...
  Class 2: Generating 1400 samples via SMOTE...
  Class 3: Generating 1920 samples via SMOTE...
Traditional SMOTE - Total samples: 8000
100%
 50/50 [01:17<00:00,  1.58s/it]

--- Traditional SMOTE (Balanced) Results ---
Accuracy: 0.7385 | F1: 0.7220 | ACSA: 0.7385 | GM: 0.6988
Per-class accuracy:
  ship: 0.9390 (n=1000)
  airplane: 0.8280 (n=1000)
  bird: 0.8070 (n=1000)
  dog: 0.3800 (n=1000)
Confusion matrix (rows=true, cols=pred): ['ship', 'airplane', 'bird', 'dog']
[[939  44  14   3]
 [115 828  56   1]
 [ 46 123 807  24]
 [ 86  94 440 380]]

==================================================
PHASE 2: DEEPSMOTE (Balancing Data)
==================================================
Preloading images for penalty loss sampling...
AE Epoch [10/100] Loss: 0.1009
AE Epoch [20/100] Loss: 0.0646
AE Epoch [30/100] Loss: 0.0468
AE Epoch [40/100] Loss: 0.0386
AE Epoch [50/100] Loss: 0.0334
AE Epoch [60/100] Loss: 0.0290
AE Epoch [70/100] Loss: 0.0256
AE Epoch [80/100] Loss: 0.0245
AE Epoch [90/100] Loss: 0.0212
AE Epoch [100/100] Loss: 0.0202


Generating synthetic images for ALL underrepresented classes...
Class 0: Has 2000 samples. Target is 2000. No generation needed.
Class 1: Generating 1000 synthetic samples...
Class 2: Generating 1400 synthetic samples...
Class 3: Generating 1920 synthetic samples...
dog (n=80):
  mean nearest-neighbor distance: 21.221
  mean latent vector norm: 21.695
  ratio: 0.978
Ship (n=2000):
  mean nearest-neighbor distance: 15.008
  mean latent vector norm: 18.308
  ratio: 0.820
Downloading: "https://github.com/toshas/torch-fidelity/releases/download/v0.2.0/weights-inception-2015-12-05-6726825d.pth" to /root/.cache/torch/hub/checkpoints/weights-inception-2015-12-05-6726825d.pth
100%|██████████| 91.2M/91.2M [00:00<00:00, 255MB/s]
Setting up [LPIPS] perceptual loss: trunk [alex], v[0.1], spatial [off]
/usr/local/lib/python3.12/dist-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
/usr/local/lib/python3.12/dist-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=AlexNet_Weights.IMAGENET1K_V1`. You can also use `weights=AlexNet_Weights.DEFAULT` to get the most up-to-date weights.
  warnings.warn(msg)
Downloading: "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth" to /root/.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth
100%|██████████| 233M/233M [00:01<00:00, 186MB/s]  
Loading model from: /usr/local/lib/python3.12/dist-packages/lpips/weights/v0.1/alex.pth
FID (real vs synthetic dog): 146.77
LPIPS diversity among synthetic dog: 0.1731
Total synthetic images generated: 4320
Original dataset size: 3680
New balanced dataset size: 8000

==================================================
PHASE 3: DEEPSMOTE CLASSIFIER (Balanced Data)
==================================================
100%
 50/50 [01:22<00:00,  1.66s/it]

--- DeepSMOTE (Balanced) Results ---
Accuracy: 0.7185 | F1: 0.6949 | ACSA: 0.7185 | GM: 0.6628
Per-class accuracy:
  ship: 0.9380 (n=1000)
  airplane: 0.8110 (n=1000)
  bird: 0.8130 (n=1000)
  dog: 0.3120 (n=1000)
Confusion matrix (rows=true, cols=pred): ['ship', 'airplane', 'bird', 'dog']
[[938  46  16   0]
 [136 811  53   0]
 [ 59 107 813  21]
 [105 101 482 312]]

✅ COMPLETE! Compare Phase 1 and Phase 3 metrics.

---

# seed value 20 
PyTorch: 2.10.0+cu128 | CUDA: True
Train samples: 3680 | Test samples: 4000
==================================================
PHASE 1: BASELINE (Imbalanced Data)
==================================================
100%
 50/50 [02:10<00:00,  2.59s/it]

--- Baseline (Imbalanced) Results ---
Accuracy: 0.7550 | F1: 0.7438 | ACSA: 0.7550 | GM: 0.7281
Per-class accuracy:
  ship: 0.9430 (n=1000)
  airplane: 0.8050 (n=1000)
  bird: 0.8210 (n=1000)
  dog: 0.4510 (n=1000)
Confusion matrix (rows=true, cols=pred): ['ship', 'airplane', 'bird', 'dog']
[[943  42  14   1]
 [136 805  56   3]
 [ 45 110 821  24]
 [ 80  95 374 451]]

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
Accuracy: 0.7522 | F1: 0.7388 | ACSA: 0.7522 | GM: 0.7219
Per-class accuracy:
  ship: 0.9480 (n=1000)
  airplane: 0.8280 (n=1000)
  bird: 0.8010 (n=1000)
  dog: 0.4320 (n=1000)
Confusion matrix (rows=true, cols=pred): ['ship', 'airplane', 'bird', 'dog']
[[948  43   8   1]
 [125 828  47   0]
 [ 52 117 801  30]
 [111 110 347 432]]

==================================================
PHASE 2: DEEPSMOTE (Balancing Data)
==================================================
Preloading images for penalty loss sampling...
AE Epoch [10/100] Loss: 0.1043
AE Epoch [20/100] Loss: 0.0638
AE Epoch [30/100] Loss: 0.0485
AE Epoch [40/100] Loss: 0.0388
AE Epoch [50/100] Loss: 0.0302
AE Epoch [60/100] Loss: 0.0271
AE Epoch [70/100] Loss: 0.0233
AE Epoch [80/100] Loss: 0.0223
AE Epoch [90/100] Loss: 0.0203
AE Epoch [100/100] Loss: 0.0189
Generating synthetic images for ALL underrepresented classes...
Class 0: Has 2000 samples. Target is 2000. No generation needed.
Class 1: Generating 1000 synthetic samples...
Class 2: Generating 1400 synthetic samples...
Class 3: Generating 1920 synthetic samples...
dog (n=80):
  mean nearest-neighbor distance: 20.600
  mean latent vector norm: 20.424
  ratio: 1.009
Ship (n=2000):
  mean nearest-neighbor distance: 14.981
  mean latent vector norm: 17.792
  ratio: 0.842
Setting up [LPIPS] perceptual loss: trunk [alex], v[0.1], spatial [off]
/usr/local/lib/python3.12/dist-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
/usr/local/lib/python3.12/dist-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=AlexNet_Weights.IMAGENET1K_V1`. You can also use `weights=AlexNet_Weights.DEFAULT` to get the most up-to-date weights.
  warnings.warn(msg)
Loading model from: /usr/local/lib/python3.12/dist-packages/lpips/weights/v0.1/alex.pth
FID (real vs synthetic dog): 150.10
LPIPS diversity among synthetic dog: 0.1729
Total synthetic images generated: 4320
Original dataset size: 3680
New balanced dataset size: 8000
==================================================
PHASE 3: DEEPSMOTE CLASSIFIER (Balanced Data)
==================================================
100%
 50/50 [01:22<00:00,  1.67s/it]

--- DeepSMOTE (Balanced) Results ---
Accuracy: 0.7360 | F1: 0.7189 | ACSA: 0.7360 | GM: 0.6971
Per-class accuracy:
  ship: 0.9390 (n=1000)
  airplane: 0.8020 (n=1000)
  bird: 0.8210 (n=1000)
  dog: 0.3820 (n=1000)
Confusion matrix (rows=true, cols=pred): ['ship', 'airplane', 'bird', 'dog']
[[939  44  16   1]
 [149 802  47   2]
 [ 49 111 821  19]
 [ 99 122 397 382]]

✅ COMPLETE! Compare Phase 1 and Phase 3 metrics.
----

# seed value 60
PyTorch: 2.10.0+cu128 | CUDA: True
Train samples: 3680 | Test samples: 4000
==================================================
PHASE 1: BASELINE (Imbalanced Data)
==================================================
100%
 50/50 [02:14<00:00,  2.69s/it]

--- Baseline (Imbalanced) Results ---
Accuracy: 0.7528 | F1: 0.7371 | ACSA: 0.7527 | GM: 0.7163
Per-class accuracy:
  ship: 0.9460 (n=1000)
  airplane: 0.8170 (n=1000)
  bird: 0.8450 (n=1000)
  dog: 0.4030 (n=1000)
Confusion matrix (rows=true, cols=pred): ['ship', 'airplane', 'bird', 'dog']
[[946  36  17   1]
 [135 817  44   4]
 [ 40  93 845  22]
 [ 96  97 404 403]]

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
Accuracy: 0.7588 | F1: 0.7459 | ACSA: 0.7588 | GM: 0.7291
Per-class accuracy:
  ship: 0.9400 (n=1000)
  airplane: 0.8420 (n=1000)
  bird: 0.8150 (n=1000)
  dog: 0.4380 (n=1000)
Confusion matrix (rows=true, cols=pred): ['ship', 'airplane', 'bird', 'dog']
[[940  48  11   1]
 [114 842  43   1]
 [ 47 112 815  26]
 [ 93 108 361 438]]

==================================================
PHASE 2: DEEPSMOTE (Balancing Data)
==================================================
Preloading images for penalty loss sampling...
AE Epoch [10/100] Loss: 0.0980
AE Epoch [20/100] Loss: 0.0673
AE Epoch [30/100] Loss: 0.0454
AE Epoch [40/100] Loss: 0.0370
AE Epoch [50/100] Loss: 0.0315
AE Epoch [60/100] Loss: 0.0259
AE Epoch [70/100] Loss: 0.0252
AE Epoch [80/100] Loss: 0.0217
AE Epoch [90/100] Loss: 0.0185
AE Epoch [100/100] Loss: 0.0183
Generating synthetic images for ALL underrepresented classes...
Class 0: Has 2000 samples. Target is 2000. No generation needed.
Class 1: Generating 1000 synthetic samples...
Class 2: Generating 1400 synthetic samples...
Class 3: Generating 1920 synthetic samples...
dog (n=80):
  mean nearest-neighbor distance: 20.416
  mean latent vector norm: 20.995
  ratio: 0.972
Ship (n=2000):
  mean nearest-neighbor distance: 15.114
  mean latent vector norm: 18.176
  ratio: 0.832
Setting up [LPIPS] perceptual loss: trunk [alex], v[0.1], spatial [off]
/usr/local/lib/python3.12/dist-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
/usr/local/lib/python3.12/dist-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=AlexNet_Weights.IMAGENET1K_V1`. You can also use `weights=AlexNet_Weights.DEFAULT` to get the most up-to-date weights.
  warnings.warn(msg)
Loading model from: /usr/local/lib/python3.12/dist-packages/lpips/weights/v0.1/alex.pth
FID (real vs synthetic dog): 147.91
LPIPS diversity among synthetic dog: 0.1707
Total synthetic images generated: 4320
Original dataset size: 3680
New balanced dataset size: 8000
==================================================
PHASE 3: DEEPSMOTE CLASSIFIER (Balanced Data)
==================================================
100%
 50/50 [01:22<00:00,  1.66s/it]

--- DeepSMOTE (Balanced) Results ---
Accuracy: 0.7380 | F1: 0.7189 | ACSA: 0.7380 | GM: 0.6942
Per-class accuracy:
  ship: 0.9430 (n=1000)
  airplane: 0.8230 (n=1000)
  bird: 0.8220 (n=1000)
  dog: 0.3640 (n=1000)
Confusion matrix (rows=true, cols=pred): ['ship', 'airplane', 'bird', 'dog']
[[943  47   8   2]
 [133 823  44   0]
 [ 49 116 822  13]
 [ 97 111 428 364]]

✅ COMPLETE! Compare Phase 1 and Phase 3 metrics.
---