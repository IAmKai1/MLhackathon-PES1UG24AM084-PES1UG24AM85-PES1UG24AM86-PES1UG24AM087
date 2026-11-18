"""HMM route reconstruction with Viterbi decoding.

Hidden states  : the candidate road arcs of each GPS observation (up to K per step).
Emissions      : log P(candidate is the reference arc | features) from the classifier,
                 scaled by `emission_weight`.
Transitions    : log p(s_t | s_{t-1}) = -|d_route - d_gps| / beta_t   (Newson & Krumm style),
                 where d_route is the driving distance through the road network between the
                 two projected positions and d_gps the straight-line GPS displacement.
                 beta_t = beta * sqrt(max(dt, 1)) widens the model across GPS gaps.
Invalid moves  : unreachable pairs (no path within the search cutoff) get -inf. If every
                 transition into a step is -inf, the chain is broken and decoding restarts
                 from that step using emissions only, so one bad step cannot derail the route.
"""
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class HMMConfig:
    beta: float = 10.0              # metres; tolerance for route-vs-GPS distance mismatch
    emission_weight: float = 1.0    # weight on classifier log-probability
    cutoff_factor: float = 3.0      # path search limit = cutoff_factor * d_gps + cutoff_base
    cutoff_base: float = 150.0
    min_prob: float = 1e-6

    def to_dict(self):
        return asdict(self)


def viterbi_decode(obs_ids, arcs_per_obs, t_per_obs, p_per_obs, xy, times, net, cfg: HMMConfig):
    """Decode the most likely arc sequence.

    obs_ids        : (T,) observation indices with at least one candidate, increasing
    arcs_per_obs   : list of T arrays of candidate arc ids
    t_per_obs      : list of T arrays of projection fractions on each candidate arc
    p_per_obs      : list of T arrays of classifier probabilities
    Returns (best_arcs (T,), n_breaks).
    """
    T = len(obs_ids)
    if T == 0:
        return np.empty(0, dtype=np.int64), 0
    emis = [cfg.emission_weight * np.log(np.clip(p, cfg.min_prob, 1.0)) for p in p_per_obs]
    score = emis[0].copy()
    scores = [score]
    back = [np.full(len(arcs_per_obs[0]), -1, dtype=np.int64)]
    n_breaks = 0
    for i in range(1, T):
        a_prev, t_prev = arcs_per_obs[i - 1], t_per_obs[i - 1]
        a_cur, t_cur = arcs_per_obs[i], t_per_obs[i]
        p0, p1 = xy[obs_ids[i - 1]], xy[obs_ids[i]]
        d_gps = float(np.hypot(*(p1 - p0)))
        dt = float(times[obs_ids[i]] - times[obs_ids[i - 1]])
        beta = cfg.beta * np.sqrt(max(dt, 1.0))
        cutoff = cfg.cutoff_factor * d_gps + cfg.cutoff_base
        trans = np.full((len(a_prev), len(a_cur)), -np.inf)
        for s in range(len(a_prev)):
            if not np.isfinite(score[s]):
                continue
            for r in range(len(a_cur)):
                d_route = net.route_distance(int(a_prev[s]), float(t_prev[s]),
                                             int(a_cur[r]), float(t_cur[r]), cutoff)
                if np.isfinite(d_route):
                    trans[s, r] = -abs(d_route - d_gps) / beta
        total = score[:, None] + trans
        best_prev = np.argmax(total, axis=0)
        best = total[best_prev, np.arange(len(a_cur))]
        if not np.isfinite(best).any():          # disconnected step: restart the chain here
            n_breaks += 1
            best_prev = np.full(len(a_cur), -1, dtype=np.int64)
            best = np.zeros(len(a_cur))
        score = best + emis[i]
        scores.append(score)
        back.append(best_prev)
    # Backtrack. A -1 pointer marks a chain restart: the earlier segment is then
    # decoded independently, ending in its own best state.
    path = np.empty(T, dtype=np.int64)
    s = int(np.argmax(scores[-1]))
    for i in range(T - 1, -1, -1):
        path[i] = arcs_per_obs[i][s]
        prev = back[i][s]
        if i > 0:
            s = int(np.argmax(scores[i - 1])) if prev < 0 else int(prev)
    return path, n_breaks
