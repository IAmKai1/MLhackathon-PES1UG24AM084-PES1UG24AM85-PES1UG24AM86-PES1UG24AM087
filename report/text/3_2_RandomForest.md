**A1. Suitability.** A random forest averages many decision trees, each trained on a bootstrap sample with random feature subsets. Bagging reduces the variance of individual trees while keeping their ability to model interactions, which suits tabular features like ours.

**A2. Strengths and limitations.** It is robust, needs little tuning, and produced the best cross-validation macro F1 (0.9393) and good probabilities (log loss 0.0724). The costs are size and speed: 150 deep trees make a 27 MB model, and inference (0.007 ms per observation) is slower than boosting. Its trees grow deep, so it can memorise training rows.

**B5. Hyperparameters.** n_estimators = 150, min_samples_leaf = 5, max_features = sqrt (random feature subset at each split), no depth limit.

**B6. Objective.** Each tree greedily minimises Gini impurity; the forest averages the trees' class probabilities.

**C7-C8. Student-testing results.** Top-1 accuracy 0.8914, macro F1 0.9376 (precision 0.9419, recall 0.9334), log loss 0.0724, route continuity 0.9375. CV macro F1 was 0.9393 (std 0.0139).

**C9. Fit.** It overfits the most of all models: training macro F1 is 0.9766 against 0.9393 in CV, a gap of 0.037, and training Top-1 is 0.968 against 0.892 on student-testing. Averaging still generalises well, so its CV score is high, but the strong memorisation and the lower student-testing Top-1 are why we preferred HistGradientBoosting.
