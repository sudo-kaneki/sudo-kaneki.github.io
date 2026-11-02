---
title: "Regularization and Validation"
chapter: 9
permalink: "/blog/cracking-the-ml-interview/regularization-and-validation/"
updated: 2026-11-01
description: "Regularization makes a model simpler than the data alone would make it, to trade a little bias for a large drop in variance."
---

Regularization makes a model simpler than the data alone would make it, to trade a little bias for a large drop in variance. Validation tells you whether the trade worked. The two go together: every regularization method has a strength to tune, and cross-validation is how you tune it honestly. Expect questions on L1 versus L2, dropout, and how to validate without fooling yourself.

## 9.1 Intuition

**Why regularize?** A flexible model can fit the training data, noise included. Regularization adds a preference for simpler solutions (small weights, few features, redundant units), which lowers variance at the cost of a little bias. The strength is a hyperparameter, usually written λ or α: zero means no regularization, and very large values give a model too simple to be useful.

**L2 regularization (ridge regression, weight decay)** (reference 19) adds λ times the sum of squared weights to the loss. It shrinks all weights smoothly toward zero but rarely makes any exactly zero. With correlated features, it spreads the weight evenly across them instead of picking one. In neural networks the same penalty is called **weight decay**, because each update multiplies the weights by a factor slightly below 1.

**L1 regularization (the lasso, Least Absolute Shrinkage and Selection Operator)** adds λ times the sum of absolute weights. It drives many weights to **exactly zero**, so it performs **feature selection** and gives a sparse, more interpretable model. With a group of correlated features it tends to keep one and drop the rest, somewhat arbitrarily.

**Elastic net** combines both penalties. It keeps L1's sparsity while L2 makes it share weight within groups of correlated features and stabilizes the selection.

**Why L1 gives zeros and L2 does not: the geometry.** Regularizing is equivalent to minimizing the training error subject to a budget on the weights. The L1 budget is a diamond with corners on the axes; the L2 budget is a circle. The solution is where the error contours first touch the budget region, which for the diamond is very often a corner, where some coefficients are exactly zero.

<figure class="figure-pending"><div class="figure-pending__box">diagram coming soon</div><figcaption>constraint geometry of lasso and ridge · after ISLR, Figure 6.7</figcaption></figure>

**Entropy regularization** (reference 20) adds a bonus for the entropy of the model's predicted distribution, discouraging overconfident outputs.

- In **reinforcement learning** (A3C, PPO, soft actor-critic), an entropy bonus on the policy keeps it exploring instead of collapsing early onto one action.
- In classification, a **confidence penalty** (subtracting β times the output entropy from the loss) works much like label smoothing: it stops the model from becoming overconfident and improves calibration.
- **Entropy minimization**, the opposite sign, is used in semi-supervised learning to push predictions on unlabeled data toward confident decisions.

**Dropout** (reference 22) randomly switches off each unit of a neural network with probability p during training, a fresh random choice for every example. No unit can rely on a specific other unit being present, which prevents **co-adaptation**, where units only work in fixed combinations. Training with dropout is like training an exponentially large ensemble of thinned networks that share weights; at test time all units are kept and outputs are rescaled so their expected value matches training. Typical rates are 0.5 for large fully connected layers and 0.1 to 0.3 elsewhere, including transformers.

**Other regularizers to name:** **early stopping** (stop training when validation loss stops improving), **data augmentation** (flips, crops, noise, paraphrases that create plausible new training examples), **label smoothing** (Chapter 8), **batch normalization** (which adds a mild noise effect), smaller models, and more data, the most reliable regularizer of all.

**Validation: estimating performance honestly.** You cannot judge a model by its training error, so you hold data out. A single train/validation split is simple but noisy when data is small, and wastes data.

**k-fold cross-validation** (reference 21) splits the data into k equal folds. Each fold takes one turn as the validation set while the model trains on the other k − 1; the k scores are averaged, and their spread shows how stable the estimate is. Every example is used for validation exactly once and for training k − 1 times.

<figure class="figure-pending"><div class="figure-pending__box">diagram coming soon</div><figcaption>k-fold cross-validation · k = 5</figcaption></figure>

| Method                  | How                                                      | Strengths                                       | Weaknesses                                               |
| ----------------------- | -------------------------------------------------------- | ----------------------------------------------- | -------------------------------------------------------- |
| Hold-out split          | One train/validation split                               | Fast; fine with lots of data                    | Noisy estimate on small data; wastes data                |
| k-fold CV (k = 5 or 10) | k rotations of k − 1 train folds and 1 validation fold   | Uses all data; gives a spread                   | k times the training cost                                |
| Leave-one-out (LOOCV)   | k = n: each row is its own fold                          | Almost unbiased; deterministic                  | n fits; high-variance estimate; too slow for most models |
| Stratified k-fold       | k-fold that keeps class proportions in each fold         | Stable on imbalanced data                       | Classification only                                      |
| Group k-fold            | All rows of an entity stay in one fold                   | Prevents group leakage                          | Needs group labels; folds may be uneven                  |
| Time-series split       | Train on the past, validate on the next block            | Respects time order                             | Fewer, smaller training sets early on                    |
| Nested CV               | An inner CV tunes hyperparameters inside each outer fold | Unbiased estimate of the whole tuning procedure | k_outer × k_inner × grid size fits                       |

**The tuning trap.** If you try many hyperparameter settings and report the best cross-validation score, that score is optimistic: you selected it partly for its luck. Report performance from data that played no part in the selection, either a held-out test set or **nested cross-validation** (9.3 shows the bias on pure noise).

## 9.2 Core math

**Penalized objectives.** For a linear model with weights β and regularization strength λ:

$$
\text{Ridge: } \min_\beta \|y - X\beta\|^2 + \lambda\|\beta\|_2^2, \qquad \text{Lasso: } \min_\beta \|y - X\beta\|^2 + \lambda\|\beta\|_1, \qquad \text{Elastic net: } + \lambda\big(\alpha\|\beta\|_1 + (1-\alpha)\|\beta\|_2^2\big)
$$

The intercept is not penalized, and features must be **standardized** first, because the penalty treats every coefficient alike and a feature in small units would otherwise need, and be punished for, a large coefficient.

**Ridge has a closed form.** Setting the gradient to zero adds λI to the normal equations (Chapter 2):

$$
\hat\beta_{\text{ridge}} = (X^\top X + \lambda I)^{-1} X^\top y
$$

Adding λI makes the matrix invertible even with perfectly collinear features or more features than rows. In terms of the singular values d_j of X, ridge shrinks the fit along each principal direction by d_j² / (d_j² + λ): directions with little variance in the data are shrunk the most.

**Lasso has no closed form, but each coordinate does.** With standardized features, coordinate descent updates one coefficient at a time by **soft-thresholding** the least-squares value z_j:

$$
S(z, \lambda) = \operatorname{sign}(z)\,\max(|z| - \lambda,\, 0)
$$

Any coefficient whose least-squares value is smaller than the threshold becomes exactly zero; larger ones are pulled toward zero by λ. Compare ridge, which scales every coefficient by the same factor and never reaches zero. In the orthonormal-design case the two rules are, with λ′ the matching threshold:

$$
\hat\beta_j^{\text{ridge}} = \frac{\hat\beta_j^{\text{OLS}}}{1 + \lambda}, \qquad \hat\beta_j^{\text{lasso}} = S\big(\hat\beta_j^{\text{OLS}}, \lambda'\big)
$$

**The Bayesian view.** Regularized least squares is **maximum a posteriori (MAP)** estimation: the penalty is the negative log of a prior on the weights. A Gaussian prior gives ridge; a Laplace prior, sharply peaked at zero, gives the lasso.

$$
-\log p(\beta \mid y) = \underbrace{-\log p(y \mid \beta)}_{\text{squared error}} \underbrace{- \log p(\beta)}_{\text{penalty}} + \text{const}, \qquad \beta_j \sim \mathcal{N}(0, \tau^2) \Rightarrow \text{L2}, \quad \beta_j \sim \text{Laplace}(0, b) \Rightarrow \text{L1}
$$

**Weight decay versus L2 in adaptive optimizers.** With plain SGD, adding λ‖w‖²/2 to the loss and multiplying the weights by (1 − ηλ) each step are the same thing. With Adam they are not: an L2 term's gradient is divided by Adam's per-weight scaling, so weights with large gradients are barely regularized. **AdamW** applies the decay directly to the weights, outside the adaptive step, and is the standard for training transformers (Chapter 10):

$$
w \leftarrow w - \eta\left(\frac{\hat m}{\sqrt{\hat v} + \epsilon} + \lambda\, w\right)
$$

**Entropy regularization.** For a predicted distribution π over actions or classes, subtract β times its entropy from the loss. In policy-gradient RL, with advantage A:

$$
H(\pi) = -\sum_a \pi(a)\log\pi(a), \qquad L = -\log\pi(a_t)\,A_t \;-\; \beta\, H\big(\pi(\cdot \mid s_t)\big)
$$

Entropy is largest (log K for K options) when the distribution is uniform, so the bonus pushes away from collapsing onto one choice. Example: logits (2.0, 0.5, −1.0) give probabilities (0.786, 0.175, 0.039) and entropy 0.622 nats, against a maximum of log 3 = 1.099.

**Inverted dropout.** During training, each activation h_i is kept with probability 1 − p and scaled up by 1/(1 − p); at test time nothing is dropped and nothing is scaled. The expected activation is unchanged:

$$
\tilde h_i = \frac{m_i\, h_i}{1 - p}, \quad m_i \sim \text{Bernoulli}(1 - p), \qquad \mathbb{E}[\tilde h_i] = h_i
$$

The original paper instead scaled weights by (1 − p) at test time; the inverted form moves that work to training so inference stays untouched. For a linear model with squared loss, dropout on the inputs is equivalent in expectation to an L2 penalty scaled by each feature's variance. Keeping dropout on at test time and averaging several stochastic passes, **Monte Carlo dropout**, gives a cheap estimate of prediction uncertainty.

**Cross-validation estimate.** With k folds F_1 … F_k and the model f̂^(−j) trained without fold j:

$$
\text{CV}_{(k)} = \frac{1}{k}\sum_{j=1}^{k} \text{Err}\big(\hat f^{(-j)},\ F_j\big)
$$

**Choosing k is a bias-variance trade-off** for the estimate itself. Small k trains on less data, so the estimate is pessimistic (biased upward). LOOCV (k = n) is nearly unbiased but averages n highly correlated models, so its variance is high. k = 5 or 10 is the usual compromise. For least-squares linear regression, LOOCV even has a shortcut needing one fit, where h_ii is the **leverage** of point i, the i-th diagonal of the hat matrix X(XᵀX)⁻¹Xᵀ:

$$
\text{CV}_{(n)} = \frac{1}{n}\sum_{i=1}^{n} \left(\frac{y_i - \hat y_i}{1 - h_{ii}}\right)^2
$$

**The one-standard-error rule** (from ISLR): among models whose CV error is within one standard error of the best, choose the simplest. It guards against picking a complex model on noise.

## 9.3 Code

**Ridge, lasso and elastic net on two nearly identical features.** y depends on x₁ only (true coefficient 3); x₂ is x₁ plus tiny noise.

```python
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

rng = np.random.default_rng(0)
x1 = rng.normal(size=100)
x2 = x1 + rng.normal(scale=0.01, size=100)                  # almost a copy of x1
y = 3 * x1 + rng.normal(scale=0.5, size=100)
X = np.c_[x1, x2]

for name, m in [("OLS", LinearRegression()), ("Ridge", Ridge(alpha=1.0)),
                ("Lasso", Lasso(alpha=0.1)), ("Elastic net", ElasticNet(alpha=0.1, l1_ratio=0.5))]:
    print(name, m.fit(X, y).coef_.round(3))
# OLS          [1.156 1.835]   unstable split of the effect
# Ridge        [1.486 1.49 ]   shared evenly, sum close to 3
# Lasso        [2.884 0.   ]   one kept, one dropped
# Elastic net  [1.43  1.43 ]   shared, with some shrinkage

# Ridge closed form on centered data matches scikit-learn
Xc, yc = X - X.mean(0), y - y.mean()
print(np.linalg.solve(Xc.T @ Xc + 1.0 * np.eye(2), Xc.T @ yc).round(3))   # [1.486 1.49 ]
```

**Lasso as a feature selector.** 200 rows, 50 standardized features, 10 truly informative:

```python
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler

X, y, true_coef = make_regression(n_samples=200, n_features=50, n_informative=10,
                                  noise=5.0, coef=True, random_state=0)
X = StandardScaler().fit_transform(X)
for alpha in [0.01, 0.1, 1, 5, 20]:
    lasso = Lasso(alpha=alpha, max_iter=10_000).fit(X, y)
    ridge = Ridge(alpha=alpha).fit(X, y)
    print(alpha, (lasso.coef_ != 0).sum(), (ridge.coef_ != 0).sum())
# alpha    lasso non-zero   ridge non-zero
# 0.01           50               50
# 0.1            43               50
# 1              11               50     <- all 10 true features + 1 false positive
# 5              10               50
# 20              7               50     <- now dropping true features too
```

**Soft-thresholding and inverted dropout from scratch:**

```python
def soft_threshold(z, lam):
    return np.sign(z) * np.maximum(np.abs(z) - lam, 0)

print(soft_threshold(np.array([-3, -0.5, 0.2, 2.0]), 1.0))   # [-2. -0.  0.  1.]

def dropout(x, p, training=True, rng=np.random.default_rng(0)):
    if not training or p == 0:
        return x                                           # inference: identity
    mask = rng.random(x.shape) >= p                        # keep with probability 1 - p
    return x * mask / (1 - p)                              # rescale so E[output] = x

out = dropout(np.ones((1000, 1000)), p=0.5)
print(out.mean(), (out == 0).mean())                       # 0.9996 0.5002
```

**Why nested cross-validation matters: tuning on pure noise.** 100 rows, 200 random features, random labels: no model can beat 50% accuracy. Tune an RBF SVM over 25 (C, γ) settings, and repeat for 10 random label draws.

```python
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.svm import SVC

rng = np.random.default_rng(0)
X = rng.normal(size=(100, 200))
grid = {"C": [0.01, 0.1, 1, 10, 100], "gamma": [1e-4, 1e-3, 1e-2, 1e-1, 1]}
non_nested, nested = [], []
for seed in range(10):
    y = np.random.default_rng(seed).integers(0, 2, 100)     # labels are pure noise
    search = GridSearchCV(SVC(), grid, cv=KFold(5, shuffle=True, random_state=seed))
    non_nested.append(search.fit(X, y).best_score_)          # best inner CV score
    nested.append(cross_val_score(search, X, y,              # outer CV around the search
                                  cv=KFold(5, shuffle=True, random_state=seed + 100)).mean())
print(np.mean(non_nested), np.mean(nested))                  # 0.552 vs 0.484
```

The non-nested score claims 55% accuracy on data with no signal: picking the luckiest of 25 settings inflated it by about 5 points. Nested CV, which evaluates the entire tuning procedure on folds it never saw, reports 48%, within noise of the true 50%. The gap grows with the size of the search and shrinks with more data.

## 9.4 In system design

- **Sparse models for cheap serving.** L1-regularized logistic regression over millions of hashed features (common in ads and spam filtering) keeps only the useful weights, which shrinks the model and speeds up serving. The lasso also tells you which upstream features you could stop computing and logging.
- **Regularization strength is retuned when data changes.** More training data needs less regularization; a model retrained on a bigger window with the old λ is usually over-regularized. Include λ in the automated tuning of each retraining run.
- **Early stopping is regularization you get for free** in gradient-boosted trees and neural networks: monitor a validation metric each round or epoch and keep the best checkpoint.
- **Transformers** typically combine AdamW weight decay (often around 0.1, not applied to biases and normalization weights), dropout of 0.1 or less, and label smoothing; very large pretraining runs often use little or no dropout because they see each example only about once.
- **Validation strategy must mirror production** (Chapter 5): time-based folds for forecasting, group folds for repeated users, and a final untouched test set. For expensive deep models, one well-chosen validation split usually replaces k-fold CV.
- **Guard against tuning overfit.** Every time a team checks the test set, it becomes a little less trustworthy. Keep a frozen test set, log how often it is used, and refresh it periodically.
- **Uncertainty with MC dropout** can flag low-confidence predictions for human review or abstention, with no extra models to train.

## 9.5 Interview Q&A

**Q1. What is the difference between L1 and L2 regularization?** L1 adds the sum of absolute weights; it drives many weights to exactly zero, giving sparse models and feature selection, and keeps one of a group of correlated features. L2 adds the sum of squared weights; it shrinks all weights smoothly, never exactly to zero, and spreads weight across correlated features. In Bayesian terms, L1 is a Laplace prior and L2 a Gaussian prior.

**Q2. Why does L1 produce sparse solutions?** Geometrically, the L1 constraint region is a diamond with corners on the axes, and the loss contours usually first touch it at a corner, where some coefficients are zero. Algebraically, the lasso's coordinate update is soft-thresholding, which sets any coefficient whose least-squares value is below λ exactly to zero; L2's update only rescales.

**Q3. When would you choose elastic net?** When you want sparsity but have groups of correlated features: the lasso picks one arbitrarily and can be unstable, while the L2 part of elastic net keeps correlated features together. Also when there are more features than rows, where the lasso can select at most n features.

**Q4. Why must features be standardized before ridge or lasso?** The penalty applies the same λ to every coefficient. A feature measured in small units needs a large coefficient and gets penalized heavily, so the result depends on arbitrary units. Standardizing puts all features on the same scale.

**Q5. How does dropout work, and what happens at test time?** During training each unit is zeroed with probability p, and with inverted dropout the survivors are scaled by 1/(1 − p) so expected activations are unchanged. At test time no units are dropped and no scaling is needed. It prevents co-adaptation and acts like averaging an ensemble of thinned networks.

**Q6. What is the difference between L2 regularization and weight decay?** With plain SGD they are identical. With adaptive optimizers like Adam, an L2 term's gradient gets divided by the per-parameter adaptive scale, so heavily updated weights are under-regularized. AdamW decouples decay from the gradient step, which works better and is the standard for transformers.

**Q7. What is entropy regularization, and where is it used?** Adding a bonus for the entropy of the model's output distribution, which discourages overconfident, collapsed predictions. In reinforcement learning it keeps policies exploring (A3C, PPO, soft actor-critic); in classification it acts as a confidence penalty similar to label smoothing.

**Q8. How do you choose k in k-fold cross-validation?** Smaller k trains on less data, so the estimate is pessimistically biased; LOOCV is nearly unbiased but has high variance and costs n fits. k = 5 or 10 balances the two. Use stratified folds for imbalanced classes, group folds for repeated entities and time-ordered folds for temporal data.

**Q9. What is nested cross-validation, and why is it needed?** The inner loop tunes hyperparameters; the outer loop evaluates the whole tuning procedure on folds the inner loop never saw. Without it, the best score from a search is optimistically biased because it was selected partly for luck; on pure noise, a 25-setting search reported 55% accuracy while nested CV reported 48%.

**Q10. Your model's training loss keeps falling but validation loss started rising at epoch 12. What do you do?** The model is overfitting after epoch 12. Use early stopping (keep the epoch-12 checkpoint), and to train longer usefully add regularization: weight decay, dropout, augmentation or more data. Also check that the validation set is representative and not leaking.

## 9.6 Cheat sheet and pitfalls

**Remember:**

- Ridge: λ‖β‖₂², closed form (XᵀX + λI)⁻¹Xᵀy, Gaussian prior, shrinks but no zeros.
- Lasso: λ‖β‖₁, soft-thresholding, Laplace prior, exact zeros and feature selection.
- Elastic net: both; handles correlated groups and p > n.
- Standardize features; never penalize the intercept.
- AdamW: decoupled weight decay; plain L2 inside Adam under-regularizes.
- Entropy bonus: −βH(π); keeps RL policies exploring, curbs overconfidence.
- Inverted dropout: mask with keep-probability 1 − p, scale by 1/(1 − p), identity at test time.
- k-fold: k = 5 or 10; LOOCV nearly unbiased but high variance; nested CV for honest tuned scores; the one-standard-error rule picks simpler models.

**Pitfalls that cost candidates:**

- Saying L2 performs feature selection, or that L1 "shrinks less".
- Regularizing unscaled features, or fitting the scaler on all data before CV.
- Forgetting to turn dropout off at inference (call `model.eval()` in PyTorch).
- Reporting the best grid-search CV score as the expected test performance.
- Using random k-fold on time series or grouped data.
- Keeping the same λ after the training set grows several times larger.

## 9.7 Further reading

<ol start="19">
  <li><a href="https://www.analyticssteps.com/blogs/l2-and-l1-regularization-machine-learning">L2 and L1 regularization in machine learning</a> (Analytics Steps)</li>
  <li><a href="https://paperswithcode.com/method/entropy-regularization">Entropy regularization</a> (Papers with Code)</li>
  <li><a href="https://en.wikipedia.org/wiki/Cross-validation_(statistics)">Cross-validation (statistics)</a> (Wikipedia)</li>
  <li>Srivastava et al., <a href="https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf">Dropout: A Simple Way to Prevent Neural Networks from Overfitting</a> (JMLR 2014)</li>
</ol>

**Supplementary:** ISLR Chapter 5 (resampling methods: cross-validation and the bootstrap) and Chapter 6 (ridge, lasso and model selection); Loshchilov and Hutter, [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) (AdamW).
