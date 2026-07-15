# Decision trees: entropy, information gain and greedy splitting

Implementation: `DecisionTreeClassifier` in [`tree.py`](../../src/scratchlearn/tree.py).

## Impurity measures

For a node whose class proportions are $p_1, \dots, p_K$:

- **Entropy** (in bits): $H(p) = -\sum_k p_k \log_2 p_k$, with $0 \log 0 = 0$. It is the
  expected number of bits needed to encode a label drawn from the node — $0$ for a pure node,
  $\log_2 K$ when all classes are equally likely.
- **Gini impurity**: $G(p) = 1 - \sum_k p_k^2$, the probability that two labels drawn with
  replacement disagree. Cheaper (no logarithm) and in practice almost interchangeable with
  entropy.

Both are implemented as small pure functions (`entropy`, `gini`) at the top of `tree.py`.

## Information gain

Splitting a node with $n$ samples into left/right children with $n_L$, $n_R$ samples changes
the impurity by the **information gain**

$$\mathrm{IG} = H(\text{parent}) - \frac{n_L}{n} H(\text{left}) - \frac{n_R}{n} H(\text{right}),$$

i.e. the mutual information between the split indicator and the label. It is non-negative
(conditioning never increases entropy), and zero exactly when the split tells you nothing
about the class.

## The greedy algorithm

Finding the smallest accurate tree is NP-complete, so CART builds greedily: at each node, try
every feature and every threshold, keep the split with the largest gain, recurse. Candidate
thresholds are midpoints between consecutive *distinct* sorted feature values — any threshold
between the same two points induces the same partition, so nothing else needs checking.

The implementation vectorises the scan per feature: after sorting, one cumulative sum of
one-hot labels gives the left-child class counts for *every* candidate split at once, and the
weighted impurities come from a single array expression.

## Stopping and overfitting

Recursion stops at a pure node, at `max_depth`, or below `min_samples_split`. With no limits
the tree drives training error to zero by memorising the data — the test suite asserts
exactly this, and notebook 03 shows the wiggly decision boundaries it produces. Limiting depth
trades that variance for bias; depth is the main way to regularise the tree.
