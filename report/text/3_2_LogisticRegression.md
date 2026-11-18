**A1. Suitability.** Logistic regression is a natural baseline for a binary "is this the reference arc?" problem: it outputs a calibrated probability directly, which the HMM needs, and its coefficients are interpretable. Several of our strongest features (distance, relative distance, heading difference) relate roughly monotonically to the target, which suits a linear model.

**A2. Strengths and limitations.** It is extremely fast (0.4 s to train on 423,868 rows, 0.0004 ms per observation) and well calibrated (student-testing log loss 0.0885). Its limitation is the linear decision boundary: it cannot learn interactions such as "a small heading difference only matters when two candidates are equally close", which the tree ensembles capture.

**B5. Hyperparameters.** StandardScaler on all features (required because the features have very different scales), L2 regularisation with C = 1.0, lbfgs solver, max_iter = 2000.

**B6. Objective.** Regularised binary cross-entropy (log loss).

**C7-C8. Student-testing results.** Top-1 accuracy 0.8887, macro F1 0.9301 (macro precision 0.9324, recall 0.9278), log loss 0.0885, route continuity 0.9332. CV macro F1 was 0.9338 (std 0.0133).

**C9. Fit.** The model slightly underfits rather than overfits: training macro F1 (0.9385) is close to CV (0.9338) and student-testing (0.9301), and it is about 1.4 percentage points below the best model on Top-1. The small train-CV gap (0.005) shows it has little capacity to memorise; its errors come from the missing feature interactions.
