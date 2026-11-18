**A1. Suitability.** A multi-layer perceptron learns non-linear combinations of the features through hidden layers, so it can model interactions without hand-crafted rules. It represents the neural-network family and produces smooth probabilities.

**A2. Strengths and limitations.** It achieved the highest route continuity of the classifiers (0.9413) and good calibration (log loss 0.0752). It is the slowest model to train (11.7 s), sensitive to feature scaling and learning rate, and less interpretable. On tabular data of this kind it did not beat the tree ensembles on Top-1 accuracy.

**B5. Hyperparameters.** StandardScaler; two hidden layers of 64 and 32 ReLU units; Adam optimiser with learning rate 0.001; batch size 1,024; L2 penalty alpha = 1e-4; early stopping on a 10% validation split with patience 5 (training stopped after 21 epochs).

**B6. Objective.** Binary cross-entropy (log loss) with L2 regularisation, minimised by Adam.

**C7-C8. Student-testing results.** Top-1 accuracy 0.8951, macro F1 0.9338 (precision 0.9381, recall 0.9296), log loss 0.0752, route continuity 0.9413. CV macro F1 was 0.9364 (std 0.0144).

**C9. Fit.** It fits well with a small train-CV gap (training macro F1 0.9471 vs CV 0.9364, gap 0.011). The epoch-wise curves on a held-out CV fold show validation loss reaching its minimum (0.094) at epoch 9 and creeping up to 0.097 by epoch 30, while training loss keeps falling, which is the onset of overfitting. Early stopping, which halted the comparison run at epoch 21, limits this.
