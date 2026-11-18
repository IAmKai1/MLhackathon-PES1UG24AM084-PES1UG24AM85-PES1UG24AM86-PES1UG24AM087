# Report outline (follow Deliverables.pdf sections 2-7)

Numbers come from `report_data.md`. Write the prose yourselves: the guidelines check report
text for AI-generated content, and every member must be able to defend each claim.

## 2.2 Dataset understanding & EDA  (state the partition for every analysis)
- Preprocessing: drop non-finite GPS rows (none found), sort by timestamp (already sorted),
  project lon/lat to metres with a local equirectangular projection per trip (why: Euclidean
  distance in degrees is wrong; trips span a few km, so projection error << GPS noise).
- Observations per trip, 1 Hz sampling (~94% of steps exactly 1 s in train), gaps (45 train trips
  have a gap > 5 s; trip 90 has a 42 h pause), network size (median ~19.5k arcs per trip), no
  isolated nodes, several weakly connected components per trip.
- Reference-route limitations: trips 33 and 54 — GPS mostly 120-140 m away from the supplied
  route (route covers only part of the drive). Labels for points > 30 m from their aligned arc
  are treated as untrusted (88% of train observations trusted, 99% of student-test).
- Label construction: the .route is an arc sequence, not per-point labels; explain the monotone
  alignment (DP that walks forward along the route minimising point-to-arc distance).
- Features and why: distance + rank + relative distance (closest road usually right), heading
  difference (separates two-way carriageways / parallel roads), context distances of +/-5 points
  (smooths noise), projection position / endpoint flag (near junctions), node degrees
  (junction complexity), speed / time gap.
- Figures: trip_distributions, all_trajectories, sampling_intervals, candidate_miss_analysis,
  a Folium map screenshot (results/eda/map_trip_00000054.html shows the bad-route case well).

## 3. Model experimentation (7 models, one per family)
- Table 3.1 from report_data.md. Explain metric definitions: CV macro F1 (binary
  candidate-is-reference, threshold 0.5), observation Top-1 accuracy, log loss, route continuity.
- Why GroupKFold by trip: points from one trip are highly correlated; splitting a trip across
  folds would leak.
- Per model (A/B/C questions): suitability, strengths/limits, hyperparameters, loss
  (log loss for LR/MLP/boosting; Gini for trees; likelihood for NB; none for KNN), student-test
  results, over/underfitting evidence (compare CV vs student-test; fold std; MLP curves).
  Observations worth discussing: GaussianNB worst (independence + Gaussian assumptions break on
  correlated distance features, very poor log loss 0.51); KNN slow at inference; trees/boosting
  best; HistGradientBoosting best student-test Top-1, F1 and log loss.
- Selection of HistGradientBoosting: best student-test Top-1/F1/log loss, calibrated
  probabilities (needed as HMM emissions), fast inference; RandomForest CV F1 within one std.

## 4. Bayesian optimisation
- Optuna TPE, 25 trials, 3-fold GroupKFold on training trips, objective CV macro F1; trial 0 =
  defaults. Search space + best values from report_data.md.
- Tuned vs untuned table (gain is small but consistent across all metrics; cost: ~8x training
  time, ~5x inference). Figures: optimisation_history, param_importance.

## 5. HMM & Viterbi
- Hidden states = candidate arcs per observation; emission = classifier log-probability;
  transition = -|network distance - GPS distance| / (beta * sqrt(dt)); bounded Dijkstra for
  network distance; unreachable = -inf; chain restart when no transition is possible
  (chain breaks column); points without candidates inherit the previous arc.
- Decoder tuned on training trips with out-of-fold probabilities (no test leakage); only
  beta x emission_weight matters (scale invariance of argmax); best product 3.
- Results: classifier-only vs HMM table, per-trip figure, route_comparison_00000077.png.
  Discuss where HMM helped most (trip 77: 0.79 -> 0.90) and least (trip 65: 0.86 -> 0.88;
  trip 07 already ~0.99).

## 6. Folium maps
- Screenshot results/maps/student_test_00000077.html (layers: GPS, network, reference,
  predicted; legend; layer control). Commit the HTML (already in results/maps/).

## 7. API, UI, deployment
- Screenshots on the evaluation day: UI on Laptop 2 with a trip result, and the API terminal on
  Laptop 1 showing the "POST /predict trip=..." log lines.

## Viva checklist (every member)
- What each raw file is and how arcs/nodes/route relate.
- Why metres, not degrees. How candidates are found (KD-tree over points sampled every 20 m along arcs).
- How per-point labels were derived and why some are untrusted.
- Why group CV. Why HistGradientBoosting. What TPE does vs grid/random search.
- HMM: states, emissions, transitions, Viterbi recurrence, chain breaks, gaps.
- Why ground truth never leaves the server.
