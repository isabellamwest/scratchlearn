# Development log

A rough journal of building this library, kept in the shape of the commit history. It was mostly evenings and weekends around other work, so the pace is uneven: the weekends did the heavy lifting, and plenty of the weekday evenings were just reading or a single small fix.

## Week of Mon 11 May 2026

```
commit 72656a7
Author: Isabella <isabellamwest@icloud.com>
Date:   Tue May 12 21:18:07 2026 +0100

    initial commit -- package skeleton, pyproject, MIT licence

commit 14a4637
Author: Isabella <isabellamwest@icloud.com>
Date:   Tue May 12 22:36:29 2026 +0100

    NotFittedError + input validation helpers (check_array/check_X_y)

commit dad9615
Author: Isabella <isabellamwest@icloud.com>
Date:   Thu May 14 20:41:17 2026 +0100

    base: BaseEstimator with get_params via signature, repr + scoring mixins

    get_params reads __init__ from inspect.signature so I never have to keep
    a parameter list in sync. Mirrors the sklearn contract closely enough
    that the notebooks read the same.

commit 304d59d
Author: Isabella <isabellamwest@icloud.com>
Date:   Sat May 16 11:07:18 2026 +0100

    LinearRegression: normal equations (lstsq) and gradient descent

commit 162bd2e
Author: Isabella <isabellamwest@icloud.com>
Date:   Sat May 16 14:22:30 2026 +0100

    metrics: r2_score, accuracy_score, rmse, mae

commit 4ab4818
Author: Isabella <isabellamwest@icloud.com>
Date:   Sat May 16 16:50:46 2026 +0100

    test: LinearRegression parity vs sklearn on synthetic data

commit 3bb5bbe
Author: Isabella <isabellamwest@icloud.com>
Date:   Sat May 16 18:39:33 2026 +0100

    Ridge: closed form via np.linalg.solve, intercept left unpenalised

commit ee2e50a
Author: Isabella <isabellamwest@icloud.com>
Date:   Sun May 17 12:14:31 2026 +0100

    Lasso: cyclic coordinate descent with soft-thresholding

    Soft-thresholding is the bit that actually drives coefficients to
    exactly zero; plain gradient descent only ever gets close.

commit 6411209
Author: Isabella <isabellamwest@icloud.com>
Date:   Sun May 17 15:33:07 2026 +0100

    test: ridge/lasso coefficients vs sklearn

commit 9d538ce
Author: Isabella <isabellamwest@icloud.com>
Date:   Sun May 17 22:51:25 2026 +0100

    lasso: track largest coord change per sweep for the tol check

```

## Week of Mon 18 May 2026

```
commit b3d3c59
Author: Isabella <isabellamwest@icloud.com>
Date:   Tue May 19 21:09:51 2026 +0100

    LogisticRegression: gradient descent with L2, logaddexp for a stable loss

    log(1+e^z) - t*z instead of the naive -y log p - (1-y) log(1-p); the
    latter blows up as soon as a probability rounds to 0 or 1.

commit 977a8f6
Author: Isabella <isabellamwest@icloud.com>
Date:   Thu May 21 19:47:29 2026 +0100

    docs: start logistic_regression derivation (cross-entropy gradient)

commit a833981
Author: Isabella <isabellamwest@icloud.com>
Date:   Sat May 23 13:02:02 2026 +0100

    metrics: precision/recall/F1 + confusion_matrix (binary and macro)

commit 08cae67
Author: Isabella <isabellamwest@icloud.com>
Date:   Sat May 23 15:28:59 2026 +0100

    metrics: roc_curve + roc_auc_score by trapezoid

commit beff5e7
Author: Isabella <isabellamwest@icloud.com>
Date:   Sat May 23 17:11:30 2026 +0100

    test: metrics vs sklearn (P/R/F1, confusion matrix, AUC)

commit 896aa78
Author: Isabella <isabellamwest@icloud.com>
Date:   Sat May 23 23:47:57 2026 +0100

    roc_curve: one point per distinct score, not per sample

commit afea51a
Author: Isabella <isabellamwest@icloud.com>
Date:   Sun May 24 10:19:14 2026 +0100

    fix: roc_curve threshold array was off by one from last night

commit 27127ce
Author: Isabella <isabellamwest@icloud.com>
Date:   Sun May 24 14:36:38 2026 +0100

    docs: linear_regression derivation (OLS, ridge, lasso)

```

## Week of Mon 25 May 2026

```
commit 23ac292
Author: Isabella <isabellamwest@icloud.com>
Date:   Mon May 25 15:12:34 2026 +0100

    model_selection: train_test_split with an optional stratify

commit 700796d
Author: Isabella <isabellamwest@icloud.com>
Date:   Mon May 25 17:44:44 2026 +0100

    model_selection: KFold + cross_val_score, clone via get_params

commit a14b10f
Author: Isabella <isabellamwest@icloud.com>
Date:   Wed May 27 20:55:18 2026 +0100

    StratifiedKFold: keep class proportions per fold

    Assign the members of each class round-robin to folds, so every fold
    sees roughly the same class balance. Remainder goes to the largest
    fractional quotas.

commit 500dd68
Author: Isabella <isabellamwest@icloud.com>
Date:   Sat May 30 16:23:22 2026 +0100

    GridSearchCV: exhaustive dict-of-lists search, cv-scored

```

## Week of Mon 1 Jun 2026

```
commit c3cffc7
Author: Isabella <isabellamwest@icloud.com>
Date:   Thu Jun 4 20:08:27 2026 +0100

    docs: start PCA derivation (variance maximisation <-> eigen/SVD)

commit e1603c2
Author: Isabella <isabellamwest@icloud.com>
Date:   Sat Jun 6 12:31:43 2026 +0100

    PCA: SVD and eigendecomposition paths

    eigh and svd agree only up to the sign of each component, which makes
    the sklearn parity test flap. Fix each component so its largest-
    magnitude entry is positive and it settles down.

commit f1fd40a
Author: Isabella <isabellamwest@icloud.com>
Date:   Sat Jun 6 14:49:53 2026 +0100

    test: PCA vs sklearn (components up to sign, explained variance ratio)

commit d170136
Author: Isabella <isabellamwest@icloud.com>
Date:   Sat Jun 6 16:58:44 2026 +0100

    PCA: inverse_transform + fit_transform

commit 867a836
Author: Isabella <isabellamwest@icloud.com>
Date:   Sun Jun 7 13:20:29 2026 +0100

    docs: finish PCA derivation

commit 215f632
Author: Isabella <isabellamwest@icloud.com>
Date:   Sun Jun 7 20:37:02 2026 +0100

    tidy: share check_array across estimators, drop the copy-paste

```

## Week of Mon 8 Jun 2026

```
commit 0be376f
Author: Isabella <isabellamwest@icloud.com>
Date:   Tue Jun 9 21:26:02 2026 +0100

    docs: kmeans derivation + notes on k-means++ seeding

commit 00eec1e
Author: Isabella <isabellamwest@icloud.com>
Date:   Sat Jun 13 11:15:31 2026 +0100

    KMeans: Lloyd with k-means++ init, n_init restarts, inertia

commit 369f44a
Author: Isabella <isabellamwest@icloud.com>
Date:   Sat Jun 13 14:41:21 2026 +0100

    KMeans: re-seed an emptied cluster on the furthest point

commit 000f1be
Author: Isabella <isabellamwest@icloud.com>
Date:   Sat Jun 13 16:52:14 2026 +0100

    test: KMeans recovers blobs, inertia is non-increasing

commit dd5cd17
Author: Isabella <isabellamwest@icloud.com>
Date:   Sun Jun 14 15:09:52 2026 +0100

    vectorise pairwise squared distances via the ||a-b||^2 expansion

```

## Week of Mon 15 Jun 2026

```
commit ad96508
Author: Isabella <isabellamwest@icloud.com>
Date:   Wed Jun 17 20:44:15 2026 +0100

    docs: GMM/EM derivation (responsibilities, ELBO, log-sum-exp)

commit 4a6be76
Author: Isabella <isabellamwest@icloud.com>
Date:   Fri Jun 19 22:58:04 2026 +0100

    wip: GaussianMixture E/M steps, not converging yet

commit 26e4963
Author: Isabella <isabellamwest@icloud.com>
Date:   Sat Jun 20 11:32:06 2026 +0100

    GaussianMixture: EM with full covariances, k-means init

commit f4d8970
Author: Isabella <isabellamwest@icloud.com>
Date:   Sat Jun 20 13:47:17 2026 +0100

    log-sum-exp in the E-step so the densities stop underflowing

    A 64-d Gaussian log-density is easily below -700, and exp of that is 0.
    logsumexp(a) = m + logsumexp(a - m); subtract the row max first and
    every exponent is <= 0.

commit c91302a
Author: Isabella <isabellamwest@icloud.com>
Date:   Sat Jun 20 15:19:56 2026 +0100

    fix: GMM covariance underflow -- add reg_covar to the diagonal

commit 6faa953
Author: Isabella <isabellamwest@icloud.com>
Date:   Sat Jun 20 17:38:24 2026 +0100

    test: EM mean log-likelihood is monotonically non-decreasing

    The one test I trust most here. If a refactor breaks the maths the log-
    likelihood dips on some iteration and this catches it.

commit 9a5ab69
Author: Isabella <isabellamwest@icloud.com>
Date:   Sun Jun 21 12:26:43 2026 +0100

    test: GMM recovers blob labels (ARI against the truth)

commit 061f736
Author: Isabella <isabellamwest@icloud.com>
Date:   Sun Jun 21 14:53:09 2026 +0100

    docs: finish GMM notes -- log-sum-exp and the reg_covar floor

```

## Week of Mon 22 Jun 2026

```
commit 9364f6c
Author: Isabella <isabellamwest@icloud.com>
Date:   Tue Jun 23 21:17:22 2026 +0100

    docs: decision tree derivation (impurity, information gain)

commit 9823b31
Author: Isabella <isabellamwest@icloud.com>
Date:   Sat Jun 27 11:04:26 2026 +0100

    DecisionTreeClassifier: greedy CART, gini and entropy

commit d329b70
Author: Isabella <isabellamwest@icloud.com>
Date:   Sat Jun 27 14:22:59 2026 +0100

    tree: vectorise the split search over sorted midpoints

    Sort each feature once, take cumulative class counts, and score every
    candidate split at the same time instead of looping. Roughly 20x faster
    on the digits.

commit 0a878b5
Author: Isabella <isabellamwest@icloud.com>
Date:   Sat Jun 27 16:48:01 2026 +0100

    test: unpruned tree overfits, matches sklearn on iris

commit 0184d72
Author: Isabella <isabellamwest@icloud.com>
Date:   Sat Jun 27 22:39:19 2026 +0100

    KNNClassifier: vectorised distances, uniform and distance weights

commit 9ff3dbe
Author: Isabella <isabellamwest@icloud.com>
Date:   Sun Jun 28 13:11:01 2026 +0100

    test: kNN on iris, distance weighting sharpens the boundary

commit a3f48ec
Author: Isabella <isabellamwest@icloud.com>
Date:   Sun Jun 28 15:35:51 2026 +0100

    entropy/gini/information_gain as standalone functions + tests

```

## Week of Mon 29 Jun 2026

```
commit 9f0c524
Author: Isabella <isabellamwest@icloud.com>
Date:   Wed Jul 1 20:29:48 2026 +0100

    docs: start MLP backprop derivation, the four equations by hand

commit b7445b8
Author: Isabella <isabellamwest@icloud.com>
Date:   Fri Jul 3 23:14:38 2026 +0100

    wip: MLPClassifier forward pass, softmax + cross-entropy

commit 12da5ee
Author: Isabella <isabellamwest@icloud.com>
Date:   Sat Jul 4 10:26:32 2026 +0100

    MLPClassifier: hand-written backprop, ReLU hidden + softmax out

commit e43b41a
Author: Isabella <isabellamwest@icloud.com>
Date:   Sat Jul 4 12:48:17 2026 +0100

    MLP: He initialisation and SGD with momentum

commit 0fb37ed
Author: Isabella <isabellamwest@icloud.com>
Date:   Sat Jul 4 14:33:29 2026 +0100

    test: analytic gradients vs central finite differences

    Backprop is easy to get subtly wrong and a bad gradient still decreases
    the loss, so the training curve won't tell you. Central differences over
    every weight and bias, relative error under 1e-6.

commit db8ef9e
Author: Isabella <isabellamwest@icloud.com>
Date:   Sat Jul 4 16:07:08 2026 +0100

    fix: ReLU mask used the wrong layer's pre-activation

    delta was gated with z of the current layer instead of the previous one.
    Passed the gradient check only after fixing the index -- exactly the
    kind of bug the finite-difference test is there to catch.

commit 7403554
Author: Isabella <isabellamwest@icloud.com>
Date:   Sat Jul 4 18:22:38 2026 +0100

    MLP reaches ~96% on the digits after 50 epochs

commit 24792cd
Author: Isabella <isabellamwest@icloud.com>
Date:   Sun Jul 5 13:41:41 2026 +0100

    docs: finish MLP derivation with the gradient-checking section

commit 34662c6
Author: Isabella <isabellamwest@icloud.com>
Date:   Sun Jul 5 20:12:11 2026 +0100

    tidy MLP: split _forward/_gradients, type the weight/bias lists

```

## Week of Mon 6 Jul 2026

```
commit a17d7b3
Author: Isabella <isabellamwest@icloud.com>
Date:   Tue Jul 7 21:33:23 2026 +0100

    examples: 01_linear_models -- sklearn parity and the lasso path

commit 4aca3a9
Author: Isabella <isabellamwest@icloud.com>
Date:   Thu Jul 9 20:47:13 2026 +0100

    examples: 02_pca_and_clustering on the digits

commit 3f9b529
Author: Isabella <isabellamwest@icloud.com>
Date:   Sat Jul 11 12:18:58 2026 +0100

    examples: 03_trees_vs_knn decision boundaries on two moons

commit 38a872f
Author: Isabella <isabellamwest@icloud.com>
Date:   Sat Jul 11 14:44:52 2026 +0100

    examples: 04_mlp_on_digits, confusion of the misread digits

commit 3649b8f
Author: Isabella <isabellamwest@icloud.com>
Date:   Sat Jul 11 16:29:51 2026 +0100

    ci: GitHub Actions -- ruff, mypy, pytest on 3.10-3.13

commit 297dd59
Author: Isabella <isabellamwest@icloud.com>
Date:   Sat Jul 11 18:03:28 2026 +0100

    ci: coverage gate at 85%, coverage badge in the README

commit c9642c9
Author: Isabella <isabellamwest@icloud.com>
Date:   Sun Jul 12 13:26:44 2026 +0100

    fix: mypy on 3.10 -- annotate loss_history_ and the layer lists

commit f24e0dd
Author: Isabella <isabellamwest@icloud.com>
Date:   Sun Jul 12 15:02:07 2026 +0100

    ruff: import order and pyupgrade fixes

commit 263f19a
Author: Isabella <isabellamwest@icloud.com>
Date:   Sun Jul 12 21:19:05 2026 +0100

    readme: rewrite the intro, add the algorithm/derivation/test table

```

## Week of Mon 13 Jul 2026

```
commit a816e75
Author: Isabella <isabellamwest@icloud.com>
Date:   Mon Jul 13 22:41:50 2026 +0100

    test: push coverage to 97% -- unfitted errors, empty-input edge cases

commit d2593f3
Author: Isabella <isabellamwest@icloud.com>
Date:   Tue Jul 14 21:08:00 2026 +0100

    docs: cross-link derivations from the docstrings; limitations section

commit 82c16f8
Author: Isabella <isabellamwest@icloud.com>
Date:   Tue Jul 14 23:26:56 2026 +0100

    readme: fix badge links, tidy the quickstart

commit 10fc3c3
Author: Isabella <isabellamwest@icloud.com>
Date:   Wed Jul 15 16:12:18 2026 +0100

    final pass: docstrings, repr examples, pin version 0.1.0

commit ec8d4e7
Author: Isabella <isabellamwest@icloud.com>
Date:   Wed Jul 15 17:04:31 2026 +0100

    run the full suite one more time (65 green) and clean the tree for upload

```
