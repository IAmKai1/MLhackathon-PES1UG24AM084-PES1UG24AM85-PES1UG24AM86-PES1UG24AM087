**A1. Suitability.** Gaussian Naive Bayes is a fast probabilistic classifier from the Bayesian family covered in the course. We included it to test whether a simple generative model is enough when each feature on its own is informative.

**A2. Strengths and limitations.** It trains almost instantly (0.1 s) and needs no tuning. However, it assumes features are conditionally independent given the class and normally distributed. Both assumptions fail badly here: our distance features (dist, dist_minus_min, dist_prev, dist_win_mean and others) are strongly correlated, so their evidence is counted several times, and distances are skewed rather than Gaussian.

**B5. Hyperparameters.** Default var_smoothing = 1e-9; no scaling is needed.

**B6. Objective.** Maximum-likelihood estimation of class priors and per-feature Gaussian means and variances.

**C7-C8. Student-testing results.** Top-1 accuracy 0.8527, macro F1 0.8766 (precision 0.8393, recall 0.9326), log loss 0.5081, route continuity 0.9200. CV macro F1 was 0.8824 (std 0.0188).

**C9. Fit.** It underfits: training, CV and student-testing macro F1 are all low and close together (0.8847, 0.8824, 0.8766). The very high log loss, about seven times that of the other models, shows that double-counting correlated evidence makes its probabilities overconfident. The high recall but low precision means it labels too many candidates as the reference arc. It was the weakest model, and its probabilities would be unsuitable as HMM emissions.
