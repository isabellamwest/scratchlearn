# PCA: eigendecomposition and SVD give the same answer

Implementation: `PCA` in [`decomposition.py`](../../src/scratchlearn/decomposition.py).

## Variance maximisation

Centre the data ($X$ has column means subtracted, $n$ rows). The first principal component is
the unit vector $u$ maximising the variance of the projection:

$$\max_{\lVert u \rVert = 1} \; \frac{1}{n-1} \lVert X u \rVert^2
  = \max_{\lVert u \rVert = 1} \; u^\top C u,
  \qquad C = \frac{X^\top X}{n-1} .$$

Introducing a Lagrange multiplier $\lambda$ for the constraint and differentiating
$u^\top C u - \lambda (u^\top u - 1)$ gives

$$C u = \lambda u ,$$

so the optimiser is an eigenvector of the covariance matrix, and the variance it captures is
its eigenvalue $\lambda$. Repeating the argument orthogonally to the components already chosen
yields the remaining eigenvectors in decreasing eigenvalue order. That is `method="eigen"`:
`np.linalg.eigh` on $C$, sorted descending.

## The SVD route

Write the thin singular value decomposition $X = U S V^\top$. Then

$$C = \frac{X^\top X}{n-1} = \frac{V S U^\top U S V^\top}{n-1} = V \frac{S^2}{n-1} V^\top,$$

which is exactly the eigendecomposition of $C$: the right singular vectors $V$ are the
principal components and the eigenvalues are $\lambda_j = s_j^2 / (n-1)$. That is
`method="svd"`. It avoids forming $X^\top X$ at all, which matters when $X$ is
ill-conditioned (squaring a matrix squares its condition number).

## Sign ambiguity

If $u$ is an eigenvector so is $-u$, so the two methods (and scikit-learn) can disagree by a
sign per component. The implementation pins the convention — each component's
largest-magnitude entry is made positive — and the parity tests compare absolute values.

## Explained variance ratio

Total variance is $\operatorname{tr}(C) = \sum_j \lambda_j$, so component $j$ explains
$\lambda_j / \sum_k \lambda_k$ of it. Reconstruction from the top $k$ components,
$\hat{X} = X V_k V_k^\top$, has squared error $\sum_{j > k} \lambda_j (n-1)$ — monotonically
decreasing in $k$, which the test suite checks directly.
