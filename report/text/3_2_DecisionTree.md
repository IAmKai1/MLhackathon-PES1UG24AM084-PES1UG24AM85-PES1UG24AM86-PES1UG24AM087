**A1. Suitability.** A decision tree learns if-then rules on the features, which matches how a human would reason about map-matching ("if the nearest candidate is much closer than the second and the heading agrees, choose it"). It handles feature interactions and differently scaled features without preprocessing.

**A2. Strengths and limitations.** It is fast (4.2 s to train, 0.0004 ms per observation), needs no scaling, and is interpretable. A single tree is unstable, however: small changes in the data can change the splits. Its probabilities are piecewise constant (one value per leaf), so they are less smooth than a boosted ensemble's.

**B5. Hyperparameters.** max_depth = 12 and min_samples_leaf = 20, both set to limit overfitting; Gini impurity criterion.

**B6. Objective.** Greedy minimisation of Gini impurity at each split.

**C7-C8. Student-testing results.** Top-1 accuracy 0.8972, macro F1 0.9345 (precision 0.9329, recall 0.9362), log loss 0.0998, route continuity 0.9308. CV macro F1 was 0.9319 (std 0.0141).

**C9. Fit.** There is moderate overfitting: training macro F1 0.9552 vs CV 0.9319 (gap 0.023). The depth and leaf-size limits keep this in check, since an unrestricted tree would memorise the training rows. Its Top-1 accuracy is surprisingly competitive, but its log loss is worse than the ensembles', because a single tree's leaf probabilities are coarse.
