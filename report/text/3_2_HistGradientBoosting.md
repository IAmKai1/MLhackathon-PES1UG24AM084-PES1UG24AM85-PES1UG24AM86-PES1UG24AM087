**A1. Suitability.** Gradient boosting builds trees sequentially, each one correcting the errors of the ensemble so far. The histogram-based variant bins features first, which makes training fast on our 424,000-row training set. Boosting is consistently strong on tabular features, captures interactions, and optimises log loss directly, so its probabilities are well suited to serve as HMM emissions.

**A2. Strengths and limitations.** It gave the best student-testing Top-1 accuracy and log loss, trains in about a second with default settings, and is compact (3 MB tuned). Its limitations are more hyperparameters to tune than the simpler models, and sequential boosting can overfit the probabilities if run for too many iterations, which our loss curves show (see below).

**B5. Hyperparameters (initial values).** learning_rate = 0.1, max_iter = 200, max_leaf_nodes = 31, min_samples_leaf = 20, l2_regularization ~ 0 (1e-6), all features per split. Section 4 tunes these.

**B6. Objective.** Binary log loss, minimised by gradient boosting.

**C7-C8. Student-testing results (untuned).** Top-1 accuracy 0.9026, macro F1 0.9386 (precision 0.9389, recall 0.9383), log loss 0.0715, route continuity 0.9382. CV macro F1 0.9382 (std 0.0139). After tuning: Top-1 0.9076, macro F1 0.9405, log loss 0.0697.

**C9. Fit.** The untuned model generalises well: training macro F1 0.9577 vs CV 0.9382 vs student-testing 0.9386, a moderate train-CV gap of 0.020 with no drop on unseen trips. The training/validation loss curves of the tuned configuration show validation log loss reaching its minimum (0.087) around iteration 115 and then rising slightly to 0.094 by the final iteration, while training loss keeps falling to 0.024. So with the tuned settings the probabilities overfit mildly after about 115 trees, even though macro F1 and Top-1 still improved over the untuned model. Fewer iterations or a stronger early-stopping rule are the clear next improvement.
