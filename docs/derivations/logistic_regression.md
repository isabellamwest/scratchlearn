# Logistic regression: gradient of the log-likelihood

Implementation: `LogisticRegression` in [`linear_model.py`](../../src/scratchlearn/linear_model.py).

## Model

For binary targets $y_i \in \{0, 1\}$ the model says

$$P(y_i = 1 \mid x_i) = \sigma(z_i), \qquad z_i = w^\top x_i + b, \qquad
  \sigma(z) = \frac{1}{1 + e^{-z}} .$$

## From likelihood to loss

The likelihood of the data is $\prod_i \sigma(z_i)^{y_i} (1 - \sigma(z_i))^{1 - y_i}$.
Taking the negative mean log gives the binary cross-entropy

$$L(w, b) = -\frac{1}{n} \sum_{i=1}^{n} \Big[ y_i \log \sigma(z_i) + (1 - y_i) \log\big(1 - \sigma(z_i)\big) \Big].$$

## The gradient, step by step

The key fact is the sigmoid's derivative:

$$\sigma'(z) = \frac{e^{-z}}{(1 + e^{-z})^2} = \sigma(z)\,\big(1 - \sigma(z)\big).$$

Write $p_i = \sigma(z_i)$ and differentiate one term of $L$ with respect to $z_i$ using the
chain rule:

$$-\frac{\partial}{\partial z_i}\Big[y_i \log p_i + (1 - y_i)\log(1 - p_i)\Big]
  = -\left[\frac{y_i}{p_i} - \frac{1 - y_i}{1 - p_i}\right] p_i (1 - p_i)
  = p_i - y_i .$$

Everything collapses to the residual $p_i - y_i$ — the same shape as in linear regression,
which is no coincidence (both are generalised linear models with canonical links). Chaining
through $z_i = w^\top x_i + b$:

$$\nabla_w L = \frac{1}{n} X^\top (\sigma(Xw + b) - y), \qquad
  \frac{\partial L}{\partial b} = \frac{1}{n} \sum_i \big(\sigma(z_i) - y_i\big).$$

With an L2 penalty $\frac{\alpha}{2}\lVert w \rVert^2$ (intercept excluded), the weight
gradient gains an $\alpha w$ term. Gradient descent then iterates
$w \leftarrow w - \eta\,(\nabla_w L + \alpha w)$.

## Numerical care in the code

- Probabilities use a sigmoid with inputs clipped to $\pm 500$, since `np.exp(710)` overflows.
- The loss is computed as $\frac{1}{n}\sum_i \big[\log(1 + e^{z_i}) - y_i z_i\big]$ via
  `np.logaddexp`, an algebraically identical form that never overflows and never takes
  $\log 0$. (Check: for $y_i = 1$ it equals $-\log \sigma(z_i)$; for $y_i = 0$,
  $-\log(1 - \sigma(z_i))$.)

There is no closed form for $w^*$ — the normal-equation trick does not carry over — which is
why an iterative method is required.
