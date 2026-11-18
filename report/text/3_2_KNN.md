**A1. Suitability.** k-nearest neighbours classifies a candidate by the labels of the most similar training candidates. It is non-parametric, so it can represent irregular decision regions without assuming a functional form, which makes it a useful contrast to the linear model.

**A2. Strengths and limitations.** It needs no training beyond storing the data and adapts to local structure in feature space. Its limitations are serious for deployment: prediction compares every query with the stored training rows, so inference is slow (0.31 ms per observation, roughly 80 times slower than HistGradientBoosting), memory grows with the training set, and it depends entirely on feature scaling. To keep prediction time manageable we trained it on a 150,000-row random sample.

**B5. Hyperparameters.** StandardScaler, k = 25 neighbours, distance weighting (closer neighbours count more), Euclidean distance, 150,000-row training sample.

**B6. Objective.** None: KNN has no training loss. Predictions are distance-weighted neighbour votes.

**C7-C8. Student-testing results.** Top-1 accuracy 0.8825, macro F1 0.9270 (precision 0.9245, recall 0.9295), log loss 0.1189, route continuity 0.9323. CV macro F1 was 0.9305 (std 0.0139).

**C9. Fit.** KNN shows clear overfitting: training macro F1 is 0.9643, but CV is 0.9305 and student-testing 0.9270, a train-CV gap of 0.034. Training points sit close to themselves in feature space, so their training predictions are optimistic. Its probabilities are also coarse (log loss 0.119) because they come from a vote over 25 neighbours.
