Train samples: 9000 | Test samples: 10000
==================================================
PHASE 1: BASELINE (Imbalanced Data)
==================================================
100%
 50/50 [01:18<00:00,  1.56s/it]

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
 50/50 [05:44<00:00,  6.93s/it]

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
FID (real vs synthetic 9): 20.20
LPIPS diversity among synthetic 9: 0.1627
Total synthetic images generated: 31000
Original dataset size: 9000
New balanced dataset size: 40000
==================================================
PHASE 3: DEEPSMOTE CLASSIFIER (Balanced Data)
==================================================
100%
 50/50 [05:49<00:00,  6.95s/it]

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

✅ COMPLETE! Compare Phase 1, Phase 3, and Phase 4 metrics.

==================================================
PHASE 4: PCA-SMOTE (Proposed Method)
==================================================
Reusing trained encoder/decoder from Phase 2.
PCA reduces latent space to intrinsic dimensionality before SMOTE.

PCA-SMOTE: Intrinsic dimensionality analysis
--------------------------------------------------
Class 0: Has 4000 samples. No generation needed.
Class 1 (n=2000): PCA 20d | 90% var in 21 dims | 95% var in 21 dims | Generating 2000 samples...
Class 2 (n=1000): PCA 20d | 90% var in 21 dims | 95% var in 21 dims | Generating 3000 samples...
Class 3 (n=750): PCA 20d | 90% var in 21 dims | 95% var in 21 dims | Generating 3250 samples...
Class 4 (n=500): PCA 20d | 90% var in 21 dims | 95% var in 21 dims | Generating 3500 samples...
Class 5 (n=350): PCA 20d | 90% var in 21 dims | 95% var in 21 dims | Generating 3650 samples...
Class 6 (n=200): PCA 20d | 90% var in 21 dims | 95% var in 21 dims | Generating 3800 samples...
Class 7 (n=100): PCA 20d | 90% var in 21 dims | 95% var in 21 dims | Generating 3900 samples...
Class 8 (n=60): PCA 20d | 90% var in 21 dims | 95% var in 21 dims | Generating 3940 samples...
Class 9 (n=40): PCA 20d | 90% var in 21 dims | 95% var in 21 dims | Generating 3960 samples...
--------------------------------------------------
PCA-SMOTE FID (real vs synthetic 9): 30.99
PCA-SMOTE LPIPS diversity among synthetic 9: 0.1487
PCA-SMOTE balanced dataset size: 40000
100%
 50/50 [05:50<00:00,  6.97s/it]

--- PCA-SMOTE (Proposed) Results ---
Accuracy: 0.9615 | F1: 0.9610 | ACSA: 0.9613 | GM: 0.9601
Per-class accuracy:
  0: 1.0000 (n=980)
  1: 0.9982 (n=1135)
  2: 0.9922 (n=1032)
  3: 0.9980 (n=1010)
  4: 0.9969 (n=982)
  5: 0.9843 (n=892)
  6: 0.9572 (n=958)
  7: 0.9212 (n=1028)
  8: 0.9168 (n=974)
  9: 0.8484 (n=1009)
Confusion matrix (rows=true, cols=pred): ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
[[ 980    0    0    0    0    0    0    0    0    0]
 [   0 1133    2    0    0    0    0    0    0    0]
 [   4    0 1024    1    0    0    0    2    1    0]
 [   0    0    0 1008    0    1    0    1    0    0]
 [   1    1    0    0  979    0    0    0    0    1]
 [   1    1    0   10    0  878    1    1    0    0]
 [  22    5    4    0    4    5  917    0    1    0]
 [   6   23   34    5    9    0    0  947    0    4]
 [  38    2    3   21    5    8    1    2  893    1]
 [  35    7    4    6   75   16    0    6    4  856]]
