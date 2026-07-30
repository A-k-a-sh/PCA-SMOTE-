`i didn't include confusion matrix in any seed, if need, tell me`
config: ['ship', 'dog', 'bird', 'airplane']
# seed 10
PHASE 1: BASELINE (Imbalanced Data)
--- Baseline (Imbalanced) Results ---
Accuracy: 0.7282 | F1: 0.6905 | ACSA: 0.7282 | GM: 0.6374
Per-class accuracy:
  ship: 0.9830 (n=1000)
  dog: 0.9090 (n=1000)
  bird: 0.7860 (n=1000)
  airplane: 0.2350 (n=1000)
PHASE 1.5: TRADITIONAL SMOTE (Balanced Data)
--- Traditional SMOTE (Balanced) Results ---
Accuracy: 0.7140 | F1: 0.6723 | ACSA: 0.7140 | GM: 0.6130
Per-class accuracy:
  ship: 0.9810 (n=1000)
  dog: 0.9040 (n=1000)
  bird: 0.7620 (n=1000)
  airplane: 0.2090 (n=1000)
PHASE 2: DEEPSMOTE (Balancing Data)
Preloading images for penalty loss sampling...
AE Epoch [10/100] Loss: 0.1093
...
Airplane (n=80):
  mean nearest-neighbor distance: 12.142
  mean latent vector norm: 13.616
  ratio: 0.892
Ship (n=2000):
  mean nearest-neighbor distance: 9.233
  mean latent vector norm: 12.258
  ratio: 0.753
FID (real vs synthetic airplane): 236.74
LPIPS diversity among synthetic airplanes: 0.1957
Total synthetic images generated: 4320
Original dataset size: 3680
New balanced dataset size: 8000
PHASE 3: DEEPSMOTE CLASSIFIER (Balanced Data)
--- DeepSMOTE (Balanced) Results ---
Accuracy: 0.6985 | F1: 0.6566 | ACSA: 0.6985 | GM: 0.5963
Per-class accuracy:
  ship: 0.9740 (n=1000)
  dog: 0.8790 (n=1000)
  bird: 0.7420 (n=1000)
  airplane: 0.1990 (n=1000)
✅ COMPLETE! Compare Phase 1 and Phase 3 metrics.
PHASE 4: PCA-SMOTE (Proposed Method)
PCA-SMOTE: Intrinsic dimensionality analysis
--------------------------------------------------
Class 0: Has 2000 samples. No generation needed.
Class 1 (n=1000): PCA 20d | 90% var in 21 dims | 95% var in 21 dims | Generating 1000 samples...
Class 2 (n=600): PCA 20d | 90% var in 21 dims | 95% var in 21 dims | Generating 1400 samples...
Class 3 (n=80): PCA 20d | 90% var in 21 dims | 95% var in 21 dims | Generating 1920 samples...
PCA-SMOTE FID (real vs synthetic Airplane): 303.43
PCA-SMOTE LPIPS diversity among synthetic Airplane: 0.1885
PCA-SMOTE balanced dataset size: 8000
--- PCA-SMOTE (Proposed) Results ---
Accuracy: 0.7017 | F1: 0.6636 | ACSA: 0.7017 | GM: 0.6113
Per-class accuracy:
  ship: 0.9690 (n=1000)
  dog: 0.8690 (n=1000)
  bird: 0.7470 (n=1000)
  airplane: 0.2220 (n=1000)
PHASE 3W: DEEPSMOTE + WEIGHTED SYNTHETIC LOSS

Same synthetic images as Phase 3 (DeepSMOTE, target=2000).
Difference: synthetic samples downweighted in classifier loss.
synth_weight = 0.3  (real=1.0, synthetic=0.3)
  [Weighted training: real=1.0, synthetic=0.3]
--- DeepSMOTE+Weighted (w=0.3) Results ---
Accuracy: 0.7033 | F1: 0.6593 | ACSA: 0.7033 | GM: 0.5958
Per-class accuracy:
  ship: 0.9780 (n=1000)
  dog: 0.8840 (n=1000)
  bird: 0.7590 (n=1000)
  airplane: 0.1920 (n=1000)
PHASE 4W: PCA-SMOTE + WEIGHTED SYNTHETIC LOSS
--- PCA-SMOTE+Weighted (w=0.3) Results ---
Accuracy: 0.7107 | F1: 0.6722 | ACSA: 0.7107 | GM: 0.6185
Per-class accuracy:
  ship: 0.9780 (n=1000)
  dog: 0.8770 (n=1000)
  bird: 0.7650 (n=1000)
  airplane: 0.2230 (n=1000)

---

# seed 20

PHASE 1: BASELINE (Imbalanced Data)
--- Baseline (Imbalanced) Results ---
Accuracy: 0.7055 | F1: 0.6587 | ACSA: 0.7055 | GM: 0.5918
Per-class accuracy:
  ship: 0.9740 (n=1000)
  dog: 0.9010 (n=1000)
  bird: 0.7640 (n=1000)
  airplane: 0.1830 (n=1000)
PHASE 1.5: TRADITIONAL SMOTE (Balanced Data)
--- Traditional SMOTE (Balanced) Results ---
Accuracy: 0.7015 | F1: 0.6520 | ACSA: 0.7015 | GM: 0.5771
Per-class accuracy:
  ship: 0.9770 (n=1000)
  dog: 0.9180 (n=1000)
  bird: 0.7450 (n=1000)
  airplane: 0.1660 (n=1000)
PHASE 2: DEEPSMOTE (Balancing Data)
Airplane (n=80):
  mean nearest-neighbor distance: 11.934
  mean latent vector norm: 13.098
  ratio: 0.911
Ship (n=2000):
  mean nearest-neighbor distance: 9.410
  mean latent vector norm: 12.174
  ratio: 0.773
LPIPS diversity among synthetic airplanes: 0.1808
Total synthetic images generated: 4320
Original dataset size: 3680
New balanced dataset size: 8000
PHASE 3: DEEPSMOTE CLASSIFIER (Balanced Data)
--- DeepSMOTE (Balanced) Results ---
Accuracy: 0.6903 | F1: 0.6383 | ACSA: 0.6903 | GM: 0.5568
Per-class accuracy:
  ship: 0.9780 (n=1000)
  dog: 0.8830 (n=1000)
  bird: 0.7520 (n=1000)
  airplane: 0.1480 (n=1000)
✅ COMPLETE! Compare Phase 1 and Phase 3 metrics.
PHASE 4: PCA-SMOTE (Proposed Method)
PCA-SMOTE FID (real vs synthetic Airplane): 279.21
PCA-SMOTE LPIPS diversity among synthetic Airplane: 0.1577
PCA-SMOTE balanced dataset size: 8000
--- PCA-SMOTE (Proposed) Results ---
Accuracy: 0.6973 | F1: 0.6492 | ACSA: 0.6973 | GM: 0.5785
Per-class accuracy:
  ship: 0.9680 (n=1000)
  dog: 0.8900 (n=1000)
  bird: 0.7600 (n=1000)
  airplane: 0.1710 (n=1000)
PHASE 3W: DEEPSMOTE + WEIGHTED SYNTHETIC LOSS
--- DeepSMOTE+Weighted (w=0.3) Results ---
Accuracy: 0.7033 | F1: 0.6535 | ACSA: 0.7032 | GM: 0.5767
Per-class accuracy:
  ship: 0.9870 (n=1000)
  dog: 0.8930 (n=1000)
  bird: 0.7700 (n=1000)
  airplane: 0.1630 (n=1000)
PHASE 4W: PCA-SMOTE + WEIGHTED SYNTHETIC LOSS
--- PCA-SMOTE+Weighted (w=0.3) Results ---
Accuracy: 0.7040 | F1: 0.6581 | ACSA: 0.7040 | GM: 0.5892
Per-class accuracy:
  ship: 0.9780 (n=1000)
  dog: 0.9040 (n=1000)
  bird: 0.7530 (n=1000)
  airplane: 0.1810 (n=1000)


---

# seed 60
PHASE 1: BASELINE (Imbalanced Data)
--- Baseline (Imbalanced) Results ---
Accuracy: 0.7278 | F1: 0.6929 | ACSA: 0.7278 | GM: 0.6437
Per-class accuracy:
  ship: 0.9830 (n=1000)
  dog: 0.8970 (n=1000)
  bird: 0.7820 (n=1000)
  airplane: 0.2490 (n=1000)
PHASE 1.5: TRADITIONAL SMOTE (Balanced Data)
--- Traditional SMOTE (Balanced) Results ---
Accuracy: 0.7140 | F1: 0.6717 | ACSA: 0.7140 | GM: 0.6108
Per-class accuracy:
  ship: 0.9710 (n=1000)
  dog: 0.9210 (n=1000)
  bird: 0.7590 (n=1000)
  airplane: 0.2050 (n=1000)
PHASE 2: DEEPSMOTE (Balancing Data)
Airplane (n=80):
  mean nearest-neighbor distance: 12.141
  mean latent vector norm: 13.372
  ratio: 0.908
Ship (n=2000):
  mean nearest-neighbor distance: 9.508
  mean latent vector norm: 12.139
  ratio: 0.783
FID (real vs synthetic airplane): 213.97
LPIPS diversity among synthetic airplanes: 0.2068
Total synthetic images generated: 4320
Original dataset size: 3680
New balanced dataset size: 8000
PHASE 3: DEEPSMOTE CLASSIFIER (Balanced Data)
--- DeepSMOTE (Balanced) Results ---
Accuracy: 0.6903 | F1: 0.6380 | ACSA: 0.6902 | GM: 0.5517
Per-class accuracy:
  ship: 0.9800 (n=1000)
  dog: 0.8970 (n=1000)
  bird: 0.7420 (n=1000)
  airplane: 0.1420 (n=1000)
✅ COMPLETE! Compare Phase 1 and Phase 3 metrics.
PHASE 4: PCA-SMOTE (Proposed Method)
PCA-SMOTE FID (real vs synthetic Airplane): 294.86
PCA-SMOTE LPIPS diversity among synthetic Airplane: 0.1834
PCA-SMOTE balanced dataset size: 8000
--- PCA-SMOTE (Proposed) Results ---
Accuracy: 0.6995 | F1: 0.6584 | ACSA: 0.6995 | GM: 0.5997
Per-class accuracy:
  ship: 0.9750 (n=1000)
  dog: 0.8790 (n=1000)
  bird: 0.7400 (n=1000)
  airplane: 0.2040 (n=1000)
PHASE 3W: DEEPSMOTE + WEIGHTED SYNTHETIC LOSS
--- DeepSMOTE+Weighted (w=0.3) Results ---
Accuracy: 0.6975 | F1: 0.6496 | ACSA: 0.6975 | GM: 0.5721
Per-class accuracy:
  ship: 0.9810 (n=1000)
  dog: 0.8870 (n=1000)
  bird: 0.7600 (n=1000)
  airplane: 0.1620 (n=1000)
PHASE 4W: PCA-SMOTE + WEIGHTED SYNTHETIC LOSS
--- PCA-SMOTE+Weighted (w=0.3) Results ---
Accuracy: 0.7115 | F1: 0.6676 | ACSA: 0.7115 | GM: 0.6024
Per-class accuracy:
  ship: 0.9760 (n=1000)
  dog: 0.9020 (n=1000)
  bird: 0.7750 (n=1000)
  airplane: 0.1930 (n=1000)

---

PHASE 5: TARGET SWEEP ANALYSIS (DeepSMOTE vs PCA-SMOTE)
 SWEEPING TARGET = 100 
========================================

--- DeepSMOTE target=100 Results ---
Accuracy: 0.7210 | F1: 0.6829 | ACSA: 0.7210 | GM: 0.6288
Per-class accuracy:
  ship: 0.9830 (n=1000)
  dog: 0.9010 (n=1000)
  bird: 0.7710 (n=1000)
  airplane: 0.2290 (n=1000)

--- PCA-SMOTE target=100 Results ---
Accuracy: 0.7165 | F1: 0.6743 | ACSA: 0.7165 | GM: 0.6143
Per-class accuracy:
  ship: 0.9700 (n=1000)
  dog: 0.9260 (n=1000)
  bird: 0.7620 (n=1000)
  airplane: 0.2080 (n=1000)
========================================
 SWEEPING TARGET = 200 
========================================

--- DeepSMOTE target=200 Results ---
Accuracy: 0.7265 | F1: 0.6889 | ACSA: 0.7265 | GM: 0.6361
Per-class accuracy:
  ship: 0.9810 (n=1000)
  dog: 0.9070 (n=1000)
  bird: 0.7830 (n=1000)
  airplane: 0.2350 (n=1000)


--- PCA-SMOTE target=200 Results ---
Accuracy: 0.7305 | F1: 0.6956 | ACSA: 0.7305 | GM: 0.6463
Per-class accuracy:
  ship: 0.9770 (n=1000)
  dog: 0.9240 (n=1000)
  bird: 0.7700 (n=1000)
  airplane: 0.2510 (n=1000)

========================================
 SWEEPING TARGET = 500 
========================================

--- DeepSMOTE target=500 Results ---
Accuracy: 0.7212 | F1: 0.6839 | ACSA: 0.7212 | GM: 0.6331
Per-class accuracy:
  ship: 0.9610 (n=1000)
  dog: 0.9060 (n=1000)
  bird: 0.7820 (n=1000)
  airplane: 0.2360 (n=1000)


--- PCA-SMOTE target=500 Results ---
Accuracy: 0.7248 | F1: 0.6914 | ACSA: 0.7248 | GM: 0.6501
Per-class accuracy:
  ship: 0.9490 (n=1000)
  dog: 0.9230 (n=1000)
  bird: 0.7580 (n=1000)
  airplane: 0.2690 (n=1000)
========================================
 SWEEPING TARGET = 1000 
========================================

--- DeepSMOTE target=1000 Results ---
Accuracy: 0.7105 | F1: 0.6696 | ACSA: 0.7105 | GM: 0.6069
Per-class accuracy:
  ship: 0.9840 (n=1000)
  dog: 0.8840 (n=1000)
  bird: 0.7720 (n=1000)
  airplane: 0.2020 (n=1000)


--- PCA-SMOTE target=1000 Results ---
Accuracy: 0.7167 | F1: 0.6748 | ACSA: 0.7168 | GM: 0.6130
Per-class accuracy:
  ship: 0.9750 (n=1000)
  dog: 0.8950 (n=1000)
  bird: 0.7930 (n=1000)
  airplane: 0.2040 (n=1000)

========================================
 SWEEPING TARGET = 2000 
========================================
--- DeepSMOTE target=2000 Results ---
Accuracy: 0.6967 | F1: 0.6529 | ACSA: 0.6967 | GM: 0.5865
Per-class accuracy:
  ship: 0.9810 (n=1000)
  dog: 0.8590 (n=1000)
  bird: 0.7630 (n=1000)
  airplane: 0.1840 (n=1000)


--- PCA-SMOTE target=2000 Results ---
Accuracy: 0.7007 | F1: 0.6541 | ACSA: 0.7007 | GM: 0.5847
Per-class accuracy:
  ship: 0.9690 (n=1000)
  dog: 0.8980 (n=1000)
  bird: 0.7590 (n=1000)
  airplane: 0.1770 (n=1000)

