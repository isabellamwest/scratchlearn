# Gaussian mixtures: EM as coordinate ascent on the ELBO

Implementation: `GaussianMixture` in [`cluster.py`](../../src/scratchlearn/cluster.py).

## The model

A $K$-component Gaussian mixture assigns each point a latent component $z_i \in \{1..K\}$:

$$p(z_i = k) = \pi_k, \qquad p(x_i \mid z_i = k) = \mathcal{N}(x_i \mid \mu_k, \Sigma_k),$$

so the marginal likelihood of one point is
$p(x_i \mid \theta) = \sum_k \pi_k\, \mathcal{N}(x_i \mid \mu_k, \Sigma_k)$ with parameters
$\theta = \{\pi_k, \mu_k, \Sigma_k\}$. We want to maximise
$\log p(X \mid \theta) = \sum_i \log \sum_k \pi_k \mathcal{N}(x_i \mid \mu_k, \Sigma_k)$.
The log of a sum has no closed-form maximiser — this is where EM comes in.

## The ELBO decomposition

Let $q(z)$ be any distribution over the latent assignments. For a single point (the sum over
$i$ carries through), multiply and divide inside the log and apply Jensen's inequality —
$\log \mathbb{E}[u] \ge \mathbb{E}[\log u]$ since $\log$ is concave:

$$\log p(x \mid \theta)
 = \log \sum_z q(z) \frac{p(x, z \mid \theta)}{q(z)}
 \ge \sum_z q(z) \log \frac{p(x, z \mid \theta)}{q(z)}
 \equiv \mathcal{L}(q, \theta).$$

$\mathcal{L}$ is the **evidence lower bound** (ELBO). The gap is exactly a KL divergence,
which one can verify by expanding $p(x, z) = p(z \mid x)\, p(x)$:

$$\log p(x \mid \theta) = \mathcal{L}(q, \theta)
  + \mathrm{KL}\big(q(z)\, \Vert\, p(z \mid x, \theta)\big),
  \qquad \mathrm{KL}(q \Vert p) = \sum_z q(z) \log \frac{q(z)}{p(z \mid x)} \ge 0.$$

EM is coordinate ascent on $\mathcal{L}$:

- **E-step** — maximise over $q$ with $\theta$ fixed. Since $\log p(x \mid \theta)$ does not
  depend on $q$, maximising $\mathcal{L}$ means minimising the KL term, and KL is zero iff
  $q(z) = p(z \mid x, \theta)$: the posterior. The bound becomes *tight*.
- **M-step** — maximise over $\theta$ with $q$ fixed, which is now a tractable expected
  complete-data log-likelihood.

**Why the log-likelihood never decreases.** After the E-step,
$\mathcal{L} = \log p(x \mid \theta_{\text{old}})$ (KL is zero). The M-step can only raise
$\mathcal{L}$, and the new log-likelihood is at least the new ELBO. Chaining:
$\log p(x \mid \theta_{\text{new}}) \ge \mathcal{L}(q, \theta_{\text{new}}) \ge
\mathcal{L}(q, \theta_{\text{old}}) = \log p(x \mid \theta_{\text{old}})$.
The test suite asserts this monotonicity on the recorded history. It is a strong correctness
check, because nearly any bug in either step breaks it.

## E-step in formulas

The posterior responsibility of component $k$ for point $i$ is, by Bayes' rule,

$$\gamma_{ik} = p(z_i = k \mid x_i, \theta)
 = \frac{\pi_k\, \mathcal{N}(x_i \mid \mu_k, \Sigma_k)}
        {\sum_j \pi_j\, \mathcal{N}(x_i \mid \mu_j, \Sigma_j)}.$$

## M-step in formulas

With $q$ fixed at $\gamma$, maximise
$\sum_i \sum_k \gamma_{ik} \big[\log \pi_k + \log \mathcal{N}(x_i \mid \mu_k, \Sigma_k)\big]$.
Setting derivatives to zero (with a Lagrange multiplier enforcing $\sum_k \pi_k = 1$) gives,
with $N_k = \sum_i \gamma_{ik}$:

$$\pi_k = \frac{N_k}{n}, \qquad
  \mu_k = \frac{1}{N_k} \sum_i \gamma_{ik}\, x_i, \qquad
  \Sigma_k = \frac{1}{N_k} \sum_i \gamma_{ik}\, (x_i - \mu_k)(x_i - \mu_k)^\top.$$

Each is the ordinary Gaussian MLE with points weighted by responsibility. The code adds
`reg_covar` to every covariance diagonal so a component that collapses onto few points cannot
produce a singular matrix.

## The log-sum-exp trick

Responsibilities are computed in log space. A Gaussian log-density in a few dimensions easily
reaches $-800$, and `np.exp(-800)` underflows to zero — the naive ratio above becomes $0/0$.
But for any constant $m$,

$$\log \sum_k e^{a_k} = m + \log \sum_k e^{a_k - m},$$

and choosing $m = \max_k a_k$ makes every exponent $\le 0$: no overflow, and at least one term
equals $1$ so the sum never vanishes. This is `_log_sum_exp` in the code; the mean of the
per-point normalisers is precisely the mean log-likelihood recorded in
`log_likelihood_history_`.

## The information-theoretic view

The E-step sets the KL divergence to zero. More broadly, EM shows that fitting a
latent-variable model is a variational problem: maximise a free-energy-style functional
$\mathcal{L}(q, \theta)$ that trades data fit against the information gap between the chosen
$q$ and the true posterior. Variational inference generalises exactly this picture by
restricting $q$ to a tractable family instead of using the exact posterior.
