# Backpropagation for an MLP, derived from the chain rule

Implementation: `MLPClassifier` in [`neural.py`](../../src/scratchlearn/neural.py).

## Notation and forward pass

The network has layers $\ell = 1, \dots, L$ with weights $W^{(\ell)}$ (shape
$\text{fan-in} \times \text{fan-out}$) and biases $b^{(\ell)}$. For a batch of $m$ rows, with
$a^{(0)} = X$:

$$z^{(\ell)} = a^{(\ell-1)} W^{(\ell)} + b^{(\ell)}, \qquad
  a^{(\ell)} = \begin{cases}
    \mathrm{ReLU}(z^{(\ell)}) = \max(z^{(\ell)}, 0) & \ell < L \\
    \mathrm{softmax}(z^{(L)}) & \ell = L .
  \end{cases}$$

The loss is the mean cross-entropy against one-hot targets $Y$:

$$L = -\frac{1}{m} \sum_{i=1}^{m} \sum_{k} Y_{ik} \log P_{ik},
  \qquad P_{ik} = \frac{e^{z^{(L)}_{ik}}}{\sum_j e^{z^{(L)}_{ij}}}.$$

The forward pass caches every $z^{(\ell)}$ and $a^{(\ell)}$; the backward pass consumes them.

## Step 1 — the output delta: softmax + cross-entropy

Define $\delta^{(\ell)} = \partial L / \partial z^{(\ell)}$. Start at the output. For one row,
with true class $t$ (so the loss contribution is $-\log P_t$):

$$\frac{\partial P_t}{\partial z_k} = P_t (\mathbb{1}[k = t] - P_k)$$

(differentiate the softmax quotient; the two cases $k = t$ and $k \ne t$ collapse into this
one formula). Then by the chain rule

$$\frac{\partial (-\log P_t)}{\partial z_k}
  = -\frac{1}{P_t} \cdot P_t (\mathbb{1}[k = t] - P_k) = P_k - Y_k .$$

All the softmax coupling and the logarithm cancel into a plain residual. Averaged over the
batch:

$$\boxed{\;\delta^{(L)} = \frac{1}{m} (P - Y)\;}$$

This cancellation is the reason the code pairs softmax with cross-entropy in a single output
delta rather than differentiating them separately (doing so is also numerically safer — no
division by potentially tiny $P_{ik}$).

## Step 2 — propagating through a hidden layer

Fix a hidden layer $\ell$. Its pre-activation feeds the next layer through
$z^{(\ell+1)} = a^{(\ell)} W^{(\ell+1)} + b^{(\ell+1)}$ with
$a^{(\ell)} = \mathrm{ReLU}(z^{(\ell)})$. Chain rule, one entry at a time:

$$\frac{\partial L}{\partial a^{(\ell)}_{ij}}
  = \sum_k \frac{\partial L}{\partial z^{(\ell+1)}_{ik}} W^{(\ell+1)}_{jk}
  \;\Longrightarrow\;
  \frac{\partial L}{\partial a^{(\ell)}} = \delta^{(\ell+1)}\, {W^{(\ell+1)}}^{\top},$$

and through the ReLU, whose derivative is $\mathbb{1}[z > 0]$ elementwise:

$$\boxed{\;\delta^{(\ell)} = \big(\delta^{(\ell+1)} {W^{(\ell+1)}}^{\top}\big)
  \odot \mathbb{1}\big[z^{(\ell)} > 0\big]\;}$$

## Step 3 — gradients of the parameters

$z^{(\ell)}_{ik} = \sum_j a^{(\ell-1)}_{ij} W^{(\ell)}_{jk} + b^{(\ell)}_k$, so

$$\frac{\partial L}{\partial W^{(\ell)}_{jk}}
  = \sum_i \delta^{(\ell)}_{ik} a^{(\ell-1)}_{ij}
  \;\Longrightarrow\;
  \boxed{\;\frac{\partial L}{\partial W^{(\ell)}} = {a^{(\ell-1)}}^{\top} \delta^{(\ell)}\;}
  \qquad
  \boxed{\;\frac{\partial L}{\partial b^{(\ell)}} = \textstyle\sum_i \delta^{(\ell)}_{i\cdot}\;}$$

(The $1/m$ lives inside $\delta^{(L)}$ and is carried backwards automatically.)

These four boxed equations are the whole of `_gradients`: compute $\delta^{(L)}$, then loop
$\ell = L, \dots, 1$ producing the two parameter gradients and, while $\ell > 1$, the next
delta.

## Gradient checking

Backprop is easy to get subtly wrong (a transpose, an off-by-one in the cached activations),
and a wrong gradient often still *decreases* the loss — training curves will not tell you.
The reliable test is central finite differences: for each scalar parameter $\theta$,

$$\frac{\partial L}{\partial \theta} \approx
  \frac{L(\theta + \varepsilon) - L(\theta - \varepsilon)}{2\varepsilon},
  \qquad \varepsilon = 10^{-5},$$

whose truncation error is $O(\varepsilon^2)$ (the one-sided difference is only
$O(\varepsilon)$ — Taylor-expand both to see the odd terms cancel). The test in
`tests/test_neural.py` runs this over every weight and bias of a tiny network and requires
relative agreement within $10^{-6}$.

## Initialisation and optimisation notes

- **He initialisation** ($\mathcal{N}(0, 2/\text{fan-in})$): a ReLU zeroes half its inputs,
  halving the variance a linear analysis would predict; the factor 2 compensates so that
  activation variance is roughly preserved layer to layer and deep stacks neither vanish nor
  explode at the start of training.
- **Momentum** keeps an exponentially-weighted velocity
  $v \leftarrow \beta v - \eta \nabla L$, $\;\theta \leftarrow \theta + v$, which damps
  oscillation across steep directions of the loss and speeds progress along shallow ones.
- **Softmax stability**: the row maximum is subtracted before exponentiating —
  $\mathrm{softmax}(z) = \mathrm{softmax}(z - c)$ for any $c$, and it keeps every exponent
  $\le 0$ (the same idea as the log-sum-exp trick in the GMM).
