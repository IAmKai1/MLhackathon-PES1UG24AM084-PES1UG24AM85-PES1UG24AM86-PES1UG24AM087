**Candidate generation.** Scoring all of a trip's arcs for every observation would be prohibitively slow, and almost all of them are obviously wrong. For each observation we keep the K nearest arcs within radius R. To find nearby arcs quickly, a KD-tree indexes points sampled every 20 m along every arc, so long arcs are found even when the GPS point lies near their middle. If no arc lies within R, the radius is widened to 120, 250 and then 500 m, so every observation has candidates. The miss-rate table shows that with K = 10 the reference arc is absent from the candidate set for only 0.27% of trusted training observations, while K = 5 would miss 1.5%. We chose R = 60 m and K = 10; the radius leaves headroom above the 30 m trusted-label cutoff for noisy points. (Because the analysis only uses trusted labels, radii of 30 m and above behave identically in the table.)

Each (observation, candidate) pair becomes one row, labelled 1 if the candidate is the observation's reference arc. This gives 1.40 million training rows, 13.4% of them positive.

**Features (19 per candidate).**

**Distance group (dist, dist_rank, dist_minus_min, dist_ratio):** the perpendicular distance to the arc, its rank among the candidates, and how much farther it is than the nearest candidate. The closest road is usually correct, so these are the strongest signals; the relative versions make them comparable across noisy and clean stretches.

**Direction group (heading_diff, heading_diff_win):** the angle between the vehicle's travel direction, estimated from points two seconds either side, and the arc's bearing, for the observation and averaged over a window. Opposite carriageways of a two-way street have the same geometry and differ only in direction, so distance alone cannot separate them. Heading also separates parallel roads.

**Context group (dist_prev, dist_next, dist_win_mean, dist_win_max):** the distances from the neighbouring observations, up to five seconds either side, to the same arc. A single GPS fix can be badly displaced, but if its neighbours all lie close to one arc, that arc is very likely correct. This smooths noise before decoding.

**Geometry group (arc_len, proj_t, at_endpoint, tail_out_deg, head_in_deg):** the arc's length, where the observation projects along it, whether the projection falls on an endpoint, and the degrees of the arc's end nodes. Errors concentrate near junctions, where several arcs meet, and these features let the model recognise those situations.

**Motion and quality group (speed, moving, n_cand, time_gap):** speed, whether the vehicle is moving, how many candidates exist, and the time since the previous fix. They indicate when GPS fixes are less reliable, for example when stationary or after a gap.
