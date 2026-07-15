# k-means: Lloyd's algorithm and the k-means++ initialisation

Implementation: `KMeans` in [`cluster.py`](../../src/scratchlearn/cluster.py).

## Objective

Partition $n$ points into $K$ clusters with centres $\mu_1, \dots, \mu_K$ minimising the
inertia

$$J = \sum_{i=1}^{n} \min_k \lVert x_i - \mu_k \rVert^2 .$$

Minimising $J$ exactly is NP-hard, so Lloyd's algorithm alternates two steps that each cannot
increase $J$:

1. **Assignment** — with centres fixed, each point joins its nearest centre. Optimal by
   definition of the $\min$.
2. **Update** — with assignments fixed, each centre moves to the mean of its members, since
   $\sum_{i \in S} \lVert x_i - \mu \rVert^2$ is minimised at $\mu = \bar{x}_S$ (set its
   gradient $-2\sum_{i \in S}(x_i - \mu)$ to zero).

$J$ is bounded below and decreases monotonically, so the algorithm converges — but only to a
local minimum that depends on the starting centres.

## k-means++

Bad initialisations (two starting centres in the same true cluster) give bad local minima.
k-means++ (Arthur & Vassilvitskii, 2007) picks the first centre uniformly at random, then each
subsequent centre with probability proportional to $D(x)^2$, the squared distance from $x$ to
its nearest already-chosen centre. Spreading centres out this way bounds the expected inertia
within $O(\log K)$ of optimal, before Lloyd's algorithm even runs.

The implementation is the direct translation: keep a running vector of squared distances to
the closest chosen centre, sample an index from `closest / closest.sum()`, update with
`np.minimum`.

## Restarts

Because every run still lands in some local minimum, `n_init` independent runs are performed
and the one with the lowest inertia kept — the same policy as scikit-learn, and the reason
the parity test only asserts our inertia is within 5% of theirs rather than identical.
