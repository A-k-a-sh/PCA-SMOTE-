# seed 10
==================================================
PHASE 1: BASELINE (Imbalanced Data)
==================================================
100%
 50/50 [01:22<00:00,  1.64s/it]

--- Baseline (Imbalanced) Results ---
Accuracy: 0.9666 | F1: 0.9662 | ACSA: 0.9664 | GM: 0.9655
Per-class accuracy:
  0: 1.0000 (n=980)
  1: 0.9991 (n=1135)
  2: 0.9942 (n=1032)
  3: 0.9941 (n=1010)
  4: 0.9969 (n=982)
  5: 0.9865 (n=892)
  6: 0.9572 (n=958)
  7: 0.9329 (n=1028)
  8: 0.9333 (n=974)
  9: 0.8702 (n=1009)
Confusion matrix (rows=true, cols=pred): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
[[ 980    0    0    0    0    0    0    0    0    0]
 [   0 1134    1    0    0    0    0    0    0    0]
 [   2    0 1026    1    1    0    0    2    0    0]
 [   0    1    0 1004    0    4    0    1    0    0]
 [   1    1    0    0  979    0    0    0    0    1]
 [   1    0    0    9    0  880    1    1    0    0]
 [  19    8    3    0    1   10  917    0    0    0]
 [   1   27   35    2    4    0    0  959    0    0]
 [  26    1    8    4    5   17    0    0  909    4]
 [  27    6    6    3   62   18    0    6    3  878]]

==================================================
PHASE 1.5: TRADITIONAL SMOTE (Balanced Data)
==================================================
Applying traditional SMOTE on pixel space...
  Class 1: Generating 2000 samples via SMOTE...
  Class 2: Generating 3000 samples via SMOTE...
  Class 3: Generating 3250 samples via SMOTE...
  Class 4: Generating 3500 samples via SMOTE...
  Class 5: Generating 3650 samples via SMOTE...
  Class 6: Generating 3800 samples via SMOTE...
  Class 7: Generating 3900 samples via SMOTE...
  Class 8: Generating 3940 samples via SMOTE...
  Class 9: Generating 3960 samples via SMOTE...
Traditional SMOTE - Total samples: 40000
100%
 50/50 [05:48<00:00,  6.94s/it]

--- Traditional SMOTE (Balanced) Results ---
Accuracy: 0.9628 | F1: 0.9621 | ACSA: 0.9627 | GM: 0.9610
Per-class accuracy:
  0: 1.0000 (n=980)
  1: 0.9982 (n=1135)
  2: 0.9990 (n=1032)
  3: 0.9980 (n=1010)
  4: 0.9969 (n=982)
  5: 0.9865 (n=892)
  6: 0.9624 (n=958)
  7: 0.9368 (n=1028)
  8: 0.9333 (n=974)
  9: 0.8157 (n=1009)
Confusion matrix (rows=true, cols=pred): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
[[ 980    0    0    0    0    0    0    0    0    0]
 [   0 1133    1    1    0    0    0    0    0    0]
 [   1    0 1031    0    0    0    0    0    0    0]
 [   0    0    1 1008    0    1    0    0    0    0]
 [   1    1    0    0  979    0    0    0    0    1]
 [   1    0    0    9    1  880    1    0    0    0]
 [  18    6    4    0    3    5  922    0    0    0]
 [   0   28   29    4    3    0    0  963    0    1]
 [  28    2    4   13    4   10    1    2  909    1]
 [  34    8    9    7   90   24    0   10    4  823]]

==================================================
PHASE 2: DEEPSMOTE (Balancing Data)
==================================================
Preloading images for penalty loss sampling...
AE Epoch [10/200] Loss: 0.0267
AE Epoch [20/200] Loss: 0.0156
AE Epoch [30/200] Loss: 0.0107
AE Epoch [40/200] Loss: 0.0087
AE Epoch [50/200] Loss: 0.0078
AE Epoch [60/200] Loss: 0.0067
AE Epoch [70/200] Loss: 0.0061
AE Epoch [80/200] Loss: 0.0057
AE Epoch [90/200] Loss: 0.0049
AE Epoch [100/200] Loss: 0.0047
AE Epoch [110/200] Loss: 0.0044
AE Epoch [120/200] Loss: 0.0037
AE Epoch [130/200] Loss: 0.0035
AE Epoch [140/200] Loss: 0.0031
AE Epoch [150/200] Loss: 0.0032
AE Epoch [160/200] Loss: 0.0029
AE Epoch [170/200] Loss: 0.0027
AE Epoch [180/200] Loss: 0.0024
AE Epoch [190/200] Loss: 0.0025
AE Epoch [200/200] Loss: 0.0025



Generating synthetic images for ALL underrepresented classes...
Class 0: Has 4000 samples. Target is 4000. No generation needed.
Class 1: Generating 2000 synthetic samples...
Class 2: Generating 3000 synthetic samples...
Class 3: Generating 3250 synthetic samples...
Class 4: Generating 3500 synthetic samples...
Class 5: Generating 3650 synthetic samples...
Class 6: Generating 3800 synthetic samples...
Class 7: Generating 3900 synthetic samples...
Class 8: Generating 3940 synthetic samples...
Class 9: Generating 3960 synthetic samples...
9 (n=40):
  mean nearest-neighbor distance: 11.689
  mean latent vector norm: 11.672
  ratio: 1.001
0 (n=4000):
  mean nearest-neighbor distance: 9.908
  mean latent vector norm: 11.728
  ratio: 0.845
Downloading: "https://github.com/toshas/torch-fidelity/releases/download/v0.2.0/weights-inception-2015-12-05-6726825d.pth" to /root/.cache/torch/hub/checkpoints/weights-inception-2015-12-05-6726825d.pth
100%|██████████| 91.2M/91.2M [00:00<00:00, 277MB/s]
Setting up [LPIPS] perceptual loss: trunk [alex], v[0.1], spatial [off]
/usr/local/lib/python3.12/dist-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
/usr/local/lib/python3.12/dist-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=AlexNet_Weights.IMAGENET1K_V1`. You can also use `weights=AlexNet_Weights.DEFAULT` to get the most up-to-date weights.
  warnings.warn(msg)
Downloading: "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth" to /root/.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth
100%|██████████| 233M/233M [00:01<00:00, 202MB/s] 
Loading model from: /usr/local/lib/python3.12/dist-packages/lpips/weights/v0.1/alex.pth
FID (real vs synthetic 9): 20.20
LPIPS diversity among synthetic 9: 0.1627
Total synthetic images generated: 31000
Original dataset size: 9000
New balanced dataset size: 40000


==================================================
PHASE 3: DEEPSMOTE CLASSIFIER (Balanced Data)
==================================================
100%
 50/50 [05:50<00:00,  6.98s/it]

--- DeepSMOTE (Balanced) Results ---
Accuracy: 0.9594 | F1: 0.9588 | ACSA: 0.9592 | GM: 0.9578
Per-class accuracy:
  0: 1.0000 (n=980)
  1: 0.9974 (n=1135)
  2: 0.9961 (n=1032)
  3: 0.9990 (n=1010)
  4: 0.9969 (n=982)
  5: 0.9843 (n=892)
  6: 0.9509 (n=958)
  7: 0.9134 (n=1028)
  8: 0.9127 (n=974)
  9: 0.8414 (n=1009)
Confusion matrix (rows=true, cols=pred): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
[[ 980    0    0    0    0    0    0    0    0    0]
 [   0 1132    2    1    0    0    0    0    0    0]
 [   2    0 1028    0    0    0    0    2    0    0]
 [   0    0    0 1009    0    1    0    0    0    0]
 [   1    0    1    0  979    0    0    0    0    1]
 [   2    0    1    8    1  878    2    0    0    0]
 [  24    7    6    0    3    7  911    0    0    0]
 [   2   33   42    5    5    0    0  939    0    2]
 [  36    2    8   20    6   10    0    2  889    1]
 [  28    8    5    2   79   28    0    7    3  849]]

✅ COMPLETE! Compare Phase 1 and Phase 3 metrics.
----

# seed 20

==================================================
PHASE 1: BASELINE (Imbalanced Data)
==================================================
100%
 50/50 [01:19<00:00,  1.60s/it]

--- Baseline (Imbalanced) Results ---
Accuracy: 0.9672 | F1: 0.9667 | ACSA: 0.9670 | GM: 0.9661
Per-class accuracy:
  0: 1.0000 (n=980)
  1: 0.9982 (n=1135)
  2: 0.9952 (n=1032)
  3: 0.9970 (n=1010)
  4: 0.9980 (n=982)
  5: 0.9865 (n=892)
  6: 0.9520 (n=958)
  7: 0.9455 (n=1028)
  8: 0.9251 (n=974)
  9: 0.8722 (n=1009)
Confusion matrix (rows=true, cols=pred): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
[[ 980    0    0    0    0    0    0    0    0    0]
 [   0 1133    2    0    0    0    0    0    0    0]
 [   2    1 1027    0    1    0    0    1    0    0]
 [   0    0    0 1007    0    2    0    1    0    0]
 [   0    0    1    0  980    0    0    0    0    1]
 [   1    2    0    7    0  880    1    1    0    0]
 [  30    8    1    0    2    5  912    0    0    0]
 [   1   21   24    3    7    0    0  972    0    0]
 [  24    2    4   11    4   16    2    2  901    8]
 [  27    8    3    3   60   18    0    8    2  880]]

==================================================
PHASE 1.5: TRADITIONAL SMOTE (Balanced Data)
==================================================
Applying traditional SMOTE on pixel space...
  Class 1: Generating 2000 samples via SMOTE...
  Class 2: Generating 3000 samples via SMOTE...
  Class 3: Generating 3250 samples via SMOTE...
  Class 4: Generating 3500 samples via SMOTE...
  Class 5: Generating 3650 samples via SMOTE...
  Class 6: Generating 3800 samples via SMOTE...
  Class 7: Generating 3900 samples via SMOTE...
  Class 8: Generating 3940 samples via SMOTE...
  Class 9: Generating 3960 samples via SMOTE...
Traditional SMOTE - Total samples: 40000
100%
 50/50 [05:44<00:00,  6.86s/it]

--- Traditional SMOTE (Balanced) Results ---
Accuracy: 0.9575 | F1: 0.9565 | ACSA: 0.9572 | GM: 0.9551
Per-class accuracy:
  0: 1.0000 (n=980)
  1: 0.9991 (n=1135)
  2: 0.9971 (n=1032)
  3: 0.9941 (n=1010)
  4: 0.9949 (n=982)
  5: 0.9854 (n=892)
  6: 0.9499 (n=958)
  7: 0.9465 (n=1028)
  8: 0.9097 (n=974)
  9: 0.7958 (n=1009)
Confusion matrix (rows=true, cols=pred): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
[[ 980    0    0    0    0    0    0    0    0    0]
 [   0 1134    1    0    0    0    0    0    0    0]
 [   3    0 1029    0    0    0    0    0    0    0]
 [   1    1    2 1004    0    2    0    0    0    0]
 [   1    0    3    0  977    0    1    0    0    0]
 [   4    2    0    6    0  879    1    0    0    0]
 [  29    8    5    0    2    4  910    0    0    0]
 [   2   21   20    7    5    0    0  973    0    0]
 [  33    2    8   15    9   14    2    4  886    1]
 [  40   10    7    6   92   26    0   24    1  803]]

==================================================
PHASE 2: DEEPSMOTE (Balancing Data)
==================================================
Preloading images for penalty loss sampling...
AE Epoch [10/200] Loss: 0.0253
AE Epoch [20/200] Loss: 0.0144
AE Epoch [30/200] Loss: 0.0112
AE Epoch [40/200] Loss: 0.0084
AE Epoch [50/200] Loss: 0.0076
AE Epoch [60/200] Loss: 0.0070
AE Epoch [70/200] Loss: 0.0061
AE Epoch [80/200] Loss: 0.0056
AE Epoch [90/200] Loss: 0.0049
AE Epoch [100/200] Loss: 0.0046
AE Epoch [110/200] Loss: 0.0043
AE Epoch [120/200] Loss: 0.0041
AE Epoch [130/200] Loss: 0.0037
AE Epoch [140/200] Loss: 0.0033
AE Epoch [150/200] Loss: 0.0031
AE Epoch [160/200] Loss: 0.0031
AE Epoch [170/200] Loss: 0.0030
AE Epoch [180/200] Loss: 0.0027
AE Epoch [190/200] Loss: 0.0026
AE Epoch [200/200] Loss: 0.0025



Generating synthetic images for ALL underrepresented classes...
Class 0: Has 4000 samples. Target is 4000. No generation needed.
Class 1: Generating 2000 synthetic samples...
Class 2: Generating 3000 synthetic samples...
Class 3: Generating 3250 synthetic samples...
Class 4: Generating 3500 synthetic samples...
Class 5: Generating 3650 synthetic samples...
Class 6: Generating 3800 synthetic samples...
Class 7: Generating 3900 synthetic samples...
Class 8: Generating 3940 synthetic samples...
Class 9: Generating 3960 synthetic samples...
9 (n=40):
  mean nearest-neighbor distance: 12.347
  mean latent vector norm: 11.615
  ratio: 1.063
0 (n=4000):
  mean nearest-neighbor distance: 9.880
  mean latent vector norm: 11.758
  ratio: 0.840
Downloading: "https://github.com/toshas/torch-fidelity/releases/download/v0.2.0/weights-inception-2015-12-05-6726825d.pth" to /root/.cache/torch/hub/checkpoints/weights-inception-2015-12-05-6726825d.pth
100%|██████████| 91.2M/91.2M [00:00<00:00, 243MB/s]
Setting up [LPIPS] perceptual loss: trunk [alex], v[0.1], spatial [off]
/usr/local/lib/python3.12/dist-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
/usr/local/lib/python3.12/dist-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=AlexNet_Weights.IMAGENET1K_V1`. You can also use `weights=AlexNet_Weights.DEFAULT` to get the most up-to-date weights.
  warnings.warn(msg)
Downloading: "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth" to /root/.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth
100%|██████████| 233M/233M [00:01<00:00, 199MB/s] 
Loading model from: /usr/local/lib/python3.12/dist-packages/lpips/weights/v0.1/alex.pth
FID (real vs synthetic 9): 20.08
LPIPS diversity among synthetic 9: 0.1773
Total synthetic images generated: 31000
Original dataset size: 9000
New balanced dataset size: 40000


==================================================
PHASE 3: DEEPSMOTE CLASSIFIER (Balanced Data)
==================================================
100%
 50/50 [05:47<00:00,  6.92s/it]

--- DeepSMOTE (Balanced) Results ---
Accuracy: 0.9617 | F1: 0.9610 | ACSA: 0.9615 | GM: 0.9599
Per-class accuracy:
  0: 1.0000 (n=980)
  1: 0.9974 (n=1135)
  2: 0.9961 (n=1032)
  3: 0.9950 (n=1010)
  4: 0.9959 (n=982)
  5: 0.9832 (n=892)
  6: 0.9572 (n=958)
  7: 0.9484 (n=1028)
  8: 0.9168 (n=974)
  9: 0.8246 (n=1009)
Confusion matrix (rows=true, cols=pred): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
[[ 980    0    0    0    0    0    0    0    0    0]
 [   0 1132    2    1    0    0    0    0    0    0]
 [   3    0 1028    0    0    0    0    1    0    0]
 [   0    1    1 1005    0    1    0    2    0    0]
 [   1    0    2    0  978    0    1    0    0    0]
 [   1    2    0    9    1  877    1    1    0    0]
 [  23    7    5    0    0    6  917    0    0    0]
 [   3   20   18    6    6    0    0  975    0    0]
 [  25    4    9   17    8   14    0    4  893    0]
 [  29   12    9    9   71   20    0   25    2  832]]

✅ COMPLETE! Compare Phase 1 and Phase 3 metrics.
---

# seed 60
==================================================
PHASE 1: BASELINE (Imbalanced Data)
==================================================
100%
 50/50 [01:27<00:00,  1.72s/it]

--- Baseline (Imbalanced) Results ---
Accuracy: 0.9702 | F1: 0.9699 | ACSA: 0.9701 | GM: 0.9694
Per-class accuracy:
  0: 1.0000 (n=980)
  1: 0.9982 (n=1135)
  2: 0.9942 (n=1032)
  3: 0.9960 (n=1010)
  4: 0.9990 (n=982)
  5: 0.9888 (n=892)
  6: 0.9582 (n=958)
  7: 0.9387 (n=1028)
  8: 0.9425 (n=974)
  9: 0.8850 (n=1009)
Confusion matrix (rows=true, cols=pred): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
[[ 980    0    0    0    0    0    0    0    0    0]
 [   0 1133    2    0    0    0    0    0    0    0]
 [   3    1 1026    0    0    0    0    2    0    0]
 [   0    0    0 1006    0    2    0    2    0    0]
 [   0    1    0    0  981    0    0    0    0    0]
 [   2    1    0    4    1  882    1    1    0    0]
 [  23    9    2    0    2    4  918    0    0    0]
 [   0   24   28    3    8    0    0  965    0    0]
 [  19    1    8    6    5    8    4    0  918    5]
 [  19    9    2    2   32   19    0   30    3  893]]

==================================================
PHASE 1.5: TRADITIONAL SMOTE (Balanced Data)
==================================================
Applying traditional SMOTE on pixel space...
  Class 1: Generating 2000 samples via SMOTE...
  Class 2: Generating 3000 samples via SMOTE...
  Class 3: Generating 3250 samples via SMOTE...
  Class 4: Generating 3500 samples via SMOTE...
  Class 5: Generating 3650 samples via SMOTE...
  Class 6: Generating 3800 samples via SMOTE...
  Class 7: Generating 3900 samples via SMOTE...
  Class 8: Generating 3940 samples via SMOTE...
  Class 9: Generating 3960 samples via SMOTE...
Traditional SMOTE - Total samples: 40000
100%
 50/50 [05:49<00:00,  6.92s/it]

--- Traditional SMOTE (Balanced) Results ---
Accuracy: 0.9592 | F1: 0.9584 | ACSA: 0.9591 | GM: 0.9570
Per-class accuracy:
  0: 1.0000 (n=980)
  1: 0.9982 (n=1135)
  2: 0.9952 (n=1032)
  3: 0.9931 (n=1010)
  4: 0.9980 (n=982)
  5: 0.9877 (n=892)
  6: 0.9426 (n=958)
  7: 0.9377 (n=1028)
  8: 0.9435 (n=974)
  9: 0.7948 (n=1009)
Confusion matrix (rows=true, cols=pred): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
[[ 980    0    0    0    0    0    0    0    0    0]
 [   0 1133    2    0    0    0    0    0    0    0]
 [   4    0 1027    0    0    0    0    1    0    0]
 [   0    2    1 1003    0    2    0    2    0    0]
 [   1    1    0    0  980    0    0    0    0    0]
 [   3    1    0    5    1  881    1    0    0    0]
 [  34    8    7    0    1    5  903    0    0    0]
 [   1   25   23    7    8    0    0  964    0    0]
 [  24    0    7    8    6    5    1    2  919    2]
 [  36    9    2    8   74   27    0   46    5  802]]

==================================================
PHASE 2: DEEPSMOTE (Balancing Data)
==================================================
Preloading images for penalty loss sampling...
AE Epoch [10/200] Loss: 0.0245
AE Epoch [20/200] Loss: 0.0134
AE Epoch [30/200] Loss: 0.0100
AE Epoch [40/200] Loss: 0.0091
AE Epoch [50/200] Loss: 0.0080
AE Epoch [60/200] Loss: 0.0073
AE Epoch [70/200] Loss: 0.0060
AE Epoch [80/200] Loss: 0.0053
AE Epoch [90/200] Loss: 0.0050
AE Epoch [100/200] Loss: 0.0044
AE Epoch [110/200] Loss: 0.0044
AE Epoch [120/200] Loss: 0.0036
AE Epoch [130/200] Loss: 0.0039
AE Epoch [140/200] Loss: 0.0033
AE Epoch [150/200] Loss: 0.0031
AE Epoch [160/200] Loss: 0.0031
AE Epoch [170/200] Loss: 0.0027
AE Epoch [180/200] Loss: 0.0029
AE Epoch [190/200] Loss: 0.0025
AE Epoch [200/200] Loss: 0.0024



Generating synthetic images for ALL underrepresented classes...
Class 0: Has 4000 samples. Target is 4000. No generation needed.
Class 1: Generating 2000 synthetic samples...
Class 2: Generating 3000 synthetic samples...
Class 3: Generating 3250 synthetic samples...
Class 4: Generating 3500 synthetic samples...
Class 5: Generating 3650 synthetic samples...
Class 6: Generating 3800 synthetic samples...
Class 7: Generating 3900 synthetic samples...
Class 8: Generating 3940 synthetic samples...
Class 9: Generating 3960 synthetic samples...
9 (n=40):
  mean nearest-neighbor distance: 12.086
  mean latent vector norm: 11.865
  ratio: 1.019
0 (n=4000):
  mean nearest-neighbor distance: 10.030
  mean latent vector norm: 11.816
  ratio: 0.849
Setting up [LPIPS] perceptual loss: trunk [alex], v[0.1], spatial [off]
/usr/local/lib/python3.12/dist-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
/usr/local/lib/python3.12/dist-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=AlexNet_Weights.IMAGENET1K_V1`. You can also use `weights=AlexNet_Weights.DEFAULT` to get the most up-to-date weights.
  warnings.warn(msg)
Loading model from: /usr/local/lib/python3.12/dist-packages/lpips/weights/v0.1/alex.pth
FID (real vs synthetic 9): 20.12
LPIPS diversity among synthetic 9: 0.1457
Total synthetic images generated: 31000
Original dataset size: 9000
New balanced dataset size: 40000


==================================================
PHASE 3: DEEPSMOTE CLASSIFIER (Balanced Data)
==================================================
100%
 50/50 [05:51<00:00,  7.01s/it]

--- DeepSMOTE (Balanced) Results ---
Accuracy: 0.9639 | F1: 0.9632 | ACSA: 0.9638 | GM: 0.9624
Per-class accuracy:
  0: 1.0000 (n=980)
  1: 0.9974 (n=1135)
  2: 0.9961 (n=1032)
  3: 0.9941 (n=1010)
  4: 0.9939 (n=982)
  5: 0.9865 (n=892)
  6: 0.9499 (n=958)
  7: 0.9426 (n=1028)
  8: 0.9476 (n=974)
  9: 0.8295 (n=1009)
Confusion matrix (rows=true, cols=pred): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
[[ 980    0    0    0    0    0    0    0    0    0]
 [   0 1132    3    0    0    0    0    0    0    0]
 [   2    0 1028    0    1    0    0    1    0    0]
 [   0    0    2 1004    0    3    0    1    0    0]
 [   2    1    0    0  976    0    3    0    0    0]
 [   2    0    0    8    0  880    1    1    0    0]
 [  28    6    3    0    2    9  910    0    0    0]
 [   0   20   23    9    6    0    0  969    0    1]
 [  12    2    9    9    8    8    1    0  923    2]
 [  26    9    2    8   48   29    0   45    5  837]]

✅ COMPLETE! Compare Phase 1 and Phase 3 metrics.