# Linear regression: the normal equations

Implementation: `LinearRegression` in [`linear_model.py`](../../src/scratchlearn/linear_model.py).

## Setup

Given $n$ samples $x_i \in \mathbb{R}^p$ with targets $y_i$, collect them into the design
matrix $X \in \mathbb{R}^{n \times p}$ (an all-ones column is appended for the intercept, so
$w$ below includes it). We minimise the mean squared error

$$L(w) = \frac{1}{n}\,\lVert y - Xw \rVert^2 .$$

## Deriving the normal equations

Expand the loss:

$$L(w) = \frac{1}{n}\left(y^\top y - 2 w^\top X^\top y + w^\top X^\top X w\right).$$

Differentiate with respect to $w$, using $\nabla_w (w^\top a) = a$ and
$\nabla_w (w^\top A w) = 2Aw$ for symmetric $A$:

$$\nabla_w L = \frac{2}{n}\left(X^\top X w - X^\top y\right).$$

Setting the gradient to zero gives the **normal equations**

$$X^\top X\, w = X^\top y .$$

When $X^\top X$ is invertible the unique minimiser is $w^* = (X^\top X)^{-1} X^\top y$.
The loss is convex (its Hessian $\tfrac{2}{n} X^\top X$ is positive semi-definite), so this
stationary point is the global minimum.

## Why the code never forms the inverse

Computing $(X^\top X)^{-1}$ explicitly squares the condition number of the problem and wastes
work. `np.linalg.lstsq` solves the least-squares problem directly via SVD, which also handles
rank-deficient $X$ (collinear features) gracefully. The same reasoning is why `Ridge` uses
`np.linalg.solve` rather than `np.linalg.inv`.

## Gradient descent variant

`solver="gd"` iterates

$$w \leftarrow w - \eta \cdot \frac{2}{n} X^\top (Xw + b - y), \qquad
  b \leftarrow b - \eta \cdot \frac{2}{n} \textstyle\sum_i (x_i^\top w + b - y_i),$$

which is the same gradient as above with the intercept kept separate. Because the loss is
convex, full-batch descent with a small enough step size converges to the same solution as the
normal equations; the test suite checks the two agree to $10^{-3}$.
