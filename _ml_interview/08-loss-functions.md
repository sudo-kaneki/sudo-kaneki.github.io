---
title: "Loss Functions"
chapter: 8
permalink: "/blog/cracking-the-ml-interview/loss-functions/"
updated: 2026-11-01
description: 'A loss function turns "how wrong was this prediction?" into one number that training can minimize.'
---

A loss function turns "how wrong was this prediction?" into one number that training can minimize. The choice is not cosmetic: it decides what the model's prediction means (a mean, a median, a probability), how much outliers matter, and how well training behaves. Interviewers use losses to check whether you can connect math, statistics and engineering in one answer.

## 8.1 Intuition

**A loss is a statement of what you care about.** Squared error says big misses are much worse than small ones. Absolute error says every unit of error costs the same. Cross-entropy says confident wrong answers are catastrophic. Each choice also implies a **noise model**: minimizing a loss is usually the same as maximizing the likelihood of the data under some assumed distribution (8.2).

**Loss, cost and metric.** The **loss** is the error on one example; the **cost** or **objective** is the average over the data, often plus a regularization term (Chapter 9). The **evaluation metric** (accuracy, F1, MAPE) is what the business reads; it is often not differentiable, so you train on a smooth **surrogate loss** that tracks it.

**Classification: cross-entropy** (reference 15). The model outputs a probability for each class, and cross-entropy charges −log(the probability given to the true class). Giving the true class 0.9 costs 0.105; giving it 0.01 costs 4.6. It is the default because it is the negative log-likelihood of a categorical (or, for two classes, Bernoulli) model, its gradient with a softmax or sigmoid is simply "prediction minus target", and it rewards well-calibrated probabilities.

**Regression: MSE, MAE and Huber** (references 16 to 18). Here r = y − ŷ is the **residual**, the error on one example.

- **MSE (mean squared error)** averages r². Squaring punishes large errors heavily, so a few outliers can dominate training. The best constant prediction under MSE is the **mean**. Its square root, **RMSE (root mean squared error)**, is in the target's units.
- **MAE (mean absolute error)** averages |r|. Every unit of error costs the same, so it is robust to outliers. The best constant prediction is the **median**. Its gradient has constant size (±1), so it does not shrink near the optimum, and it is not differentiable at 0.
- **Huber loss** is quadratic for small residuals (|r| ≤ δ) and linear beyond. It behaves like MSE near the optimum (smooth, shrinking gradients) and like MAE for outliers (bounded gradients). δ sets where "small" ends and is usually chosen relative to the noise scale.

<figure class="figure-pending"><div class="figure-pending__box">diagram coming soon</div><figcaption>computed from the loss formulas · residuals from −3 to 3</figcaption></figure>

| Loss          | Formula per example                  | Best constant           | Outliers                       | Gradient                        | Use it when                                                     |
| ------------- | ------------------------------------ | ----------------------- | ------------------------------ | ------------------------------- | --------------------------------------------------------------- |
| MSE           | r²                                   | Mean                    | Very sensitive                 | 2r: large for large errors      | Errors are roughly Gaussian; large errors really are much worse |
| MAE           | \|r\|                                | Median                  | Robust                         | ±1 (undefined at 0)             | Heavy-tailed noise, outliers you do not trust                   |
| Huber         | ½r² if \|r\| ≤ δ, else δ(\|r\| − ½δ) | Between median and mean | Robust                         | r, capped at ±δ                 | You want MSE's smoothness without its outlier sensitivity       |
| Cross-entropy | −log p(true class)                   | Class frequencies       | Punishes confident errors hard | p − y (with softmax or sigmoid) | Classification; you need probabilities                          |

**Other losses worth recognizing:**

- **Hinge loss**, max(0, 1 − y·f(x)), used by SVMs (Chapter 2): zero once an example is beyond the margin.
- **Quantile (pinball) loss** predicts a chosen quantile rather than the mean; quantile 0.5 is MAE. Used for forecast intervals ("90% of deliveries arrive within this time").
- **Log-cosh loss**, log(cosh(r)): a smooth Huber-like loss, twice differentiable everywhere.
- **Focal loss** and **class-balanced loss** for imbalanced classification (Chapter 6).
- **KL divergence** for matching distributions, as in knowledge distillation and variational autoencoders (8.2).
- **Contrastive and triplet losses** for embeddings: pull matching pairs together and push others apart, as in CLIP and retrieval models.
- **Ranking losses** (pairwise, listwise) for search and recommendations, where the order matters more than each score.

## 8.2 Core math

**Losses come from likelihoods.** If you assume y = f(x) + ε and maximize the likelihood of the data, the loss you minimize is the negative log-likelihood of the noise model:

| Noise assumed            | Negative log-likelihood per example | Loss                 |
| ------------------------ | ----------------------------------- | -------------------- |
| Gaussian, fixed variance | (y − ŷ)² / 2σ² + constant           | MSE                  |
| Laplace                  | \|y − ŷ\| / b + constant            | MAE                  |
| Bernoulli (0/1 label)    | −\[y log p + (1 − y) log(1 − p)\]   | Binary cross-entropy |
| Categorical (K classes)  | −log p_y                            | Cross-entropy        |

So choosing MSE is choosing to believe the noise is Gaussian; heavy-tailed noise calls for MAE or Huber.

**Best constant predictions.** Minimizing Σ(y_i − c)² gives c = mean(y): set the derivative −2Σ(y_i − c) to zero. Minimizing Σ|y_i − c| gives c = median(y): the derivative is (number of points below c) minus (number above), which is zero at the median. For y = \[1, 2, 3, 4, 100\], MSE picks 22 and MAE picks 3. A model trained with MSE predicts the conditional mean of y given x; one trained with MAE predicts the conditional median.

**Cross-entropy, entropy and KL divergence.** For a true distribution p and a predicted distribution q over classes:

$$
H(p) = -\sum_k p_k \log p_k, \qquad H(p, q) = -\sum_k p_k \log q_k, \qquad D_{\text{KL}}(p \,\|\, q) = \sum_k p_k \log\frac{p_k}{q_k} = H(p, q) - H(p)
$$

H(p) is fixed by the data, so minimizing cross-entropy is the same as minimizing the KL divergence from the true distribution to the model's. With a one-hot label, H(p) = 0 and cross-entropy reduces to −log q_y. KL divergence is never negative and is zero only when q = p, but it is **not symmetric**: D_KL(p‖q) ≠ D_KL(q‖p), so it is not a true distance. Example: p = (0.7, 0.2, 0.1), q = (0.5, 0.3, 0.2) gives H(p) = 0.802, H(p, q) = 0.887 and KL = 0.085 nats.

**Softmax and its gradient.** With logits z and p = softmax(z), the cross-entropy gradient with respect to the logits is strikingly simple, with y the one-hot label:

$$
p_k = \frac{e^{z_k}}{\sum_j e^{z_j}}, \qquad L = -\log p_y, \qquad \frac{\partial L}{\partial z_k} = p_k - y_k
$$

The same "prediction minus target" form appears for sigmoid with binary cross-entropy and for linear regression with MSE, which is no coincidence: all three are generalized linear models with their natural (canonical) link.

**Why not MSE for classification?** With a sigmoid output, MSE's gradient is (p − y)·p(1 − p). When the model is confidently wrong (p near 0 for a true 1), the factor p(1 − p) is near 0, so learning stalls exactly when it should be fastest. Cross-entropy's gradient, p − y, stays large.

**Numerical stability: the log-sum-exp trick.** Computing softmax naively overflows for large logits: e^1000 is infinite in floating point. Subtract the maximum logit first, which changes nothing mathematically:

$$
\log p_k = z_k - \log\sum_j e^{z_j} = (z_k - m) - \log\sum_j e^{z_j - m}, \qquad m = \max_j z_j
$$

This is why frameworks take raw logits in losses such as PyTorch's `CrossEntropyLoss` and `BCEWithLogitsLoss` instead of probabilities.

**Label smoothing.** Replace the one-hot target with a softened one, giving ε/K to every class. With K = 3 and ε = 0.1, the target (1, 0, 0) becomes (0.933, 0.033, 0.033). The model can no longer drive the true-class logit to infinity, which improves calibration and acts as a regularizer:

$$
y_k^{\text{LS}} = (1 - \varepsilon)\, y_k + \frac{\varepsilon}{K}
$$

**Huber loss and its gradient.** With threshold δ:

$$
L_\delta(r) = \begin{cases} \tfrac{1}{2} r^2 & |r| \le \delta \\ \delta\big(|r| - \tfrac{1}{2}\delta\big) & |r| > \delta \end{cases}, \qquad \frac{\partial L_\delta}{\partial r} = \operatorname{clip}(r, -\delta, \delta)
$$

The two pieces meet with matching value and slope at |r| = δ, so the loss is smooth. The gradient is the residual clipped to ±δ: an outlier can pull on the fit only with force δ, however wrong it is. The same idea, under the name **smooth L1 loss**, is standard for bounding-box regression in object detection.

**Quantile loss** for quantile τ between 0 and 1 penalizes under-prediction with weight τ and over-prediction with weight 1 − τ, so the minimizer is the τ-th quantile:

$$
L_\tau(r) = \max\big(\tau\, r,\ (\tau - 1)\, r\big), \qquad r = y - \hat y
$$

## 8.3 Code

**Stable cross-entropy from logits, and why the naive version fails:**

```python
import numpy as np

def cross_entropy(logits, labels):
    z = logits - logits.max(axis=1, keepdims=True)              # log-sum-exp trick
    log_probs = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    return -log_probs[np.arange(len(labels)), labels].mean()

logits = np.array([[2.0, 1.0, 0.1], [1000.0, 0.0, -1000.0]])
labels = np.array([0, 0])
print(cross_entropy(logits, labels))                            # 0.2085

naive = -np.log(np.exp(logits[1, 0]) / np.exp(logits[1]).sum())
print(naive)                                                    # nan: exp(1000) overflowed
```

**MSE, MAE and Huber on data with outliers.** 200 points from y = 3x + 5 + noise; then 20 of them (10%) get 30 to 60 added, like corrupted sensor readings.

```python
import numpy as np
from sklearn.linear_model import LinearRegression, HuberRegressor, QuantileRegressor

rng = np.random.default_rng(0)
X = rng.uniform(0, 10, size=(200, 1))
y = 3 * X[:, 0] + 5 + rng.normal(0, 1, 200)
idx = rng.choice(200, 20, replace=False)
y_out = y.copy()
y_out[idx] += rng.uniform(30, 60, 20)                           # 10% gross outliers

models = {
    "MSE (ordinary least squares)": LinearRegression(),
    "Huber": HuberRegressor(epsilon=1.35),
    "MAE (median regression)": QuantileRegressor(quantile=0.5, alpha=0, solver="highs"),
}
for name, m in models.items():
    m.fit(X, y_out)
    print(name, m.coef_[0], m.intercept_)
```

| Loss                         | Slope (true 3) | Intercept (true 5) |
| ---------------------------- | -------------- | ------------------ |
| MSE (ordinary least squares) | 2.115          | 14.335             |
| Huber                        | 2.941          | 5.354              |
| MAE (median regression)      | 2.933          | 5.307              |

Ten percent bad data pushed the least-squares intercept from 5 to over 14; Huber and MAE barely moved. scikit-learn's `HuberRegressor` uses `epsilon` as its threshold on scaled residuals; 1.35 is the classic choice that keeps 95% of least-squares efficiency when the noise really is Gaussian.

**Quantile regression for prediction intervals.** `QuantileRegressor(quantile=0.9)` fitted on the clean data gives a line under which 90.5% of the training points fall, close to the 90% target. Fitting quantiles 0.05 and 0.95 gives a 90% prediction interval.

## 8.4 In system design

- **Match the loss to the business quantity.** Delivery-time estimates shown to customers are judged on late deliveries, so predict a high quantile (say the 90th) with quantile loss rather than the mean. Demand forecasts that drive inventory often use quantile loss too, because understocking and overstocking cost different amounts.
- **The loss decides what a prediction means.** An MSE model predicts the average, so on skewed targets (revenue, time spent) its typical prediction is pulled up by rare large values. Training on log(y) with MSE predicts the geometric mean instead; say which one the product needs.
- **Probabilities need log loss.** Ad bidding, risk pricing and blending scores from several models need calibrated probabilities, so train with cross-entropy and monitor calibration in production, not just ranking metrics.
- **Train on a surrogate, evaluate on the metric.** You cannot backpropagate through accuracy, NDCG or revenue. Train on cross-entropy or a ranking loss, then pick models and thresholds by the real metric on validation data.
- **Multi-task models** add several losses (click, like, share) with weights. The weights are product decisions as much as technical ones, and they must be revisited when one task's loss scale changes.
- **Distillation** trains a small student model on a large teacher's soft probabilities with a KL-divergence loss, often with a temperature that softens both distributions, plus the usual cross-entropy on true labels. It is a standard way to make models cheap enough to serve.
- **Robust losses as a data-quality guard.** If labels come from noisy sources (crowd workers, logs), Huber, MAE or label smoothing limit how much bad labels can drag the model.

## 8.5 Interview Q&A

**Q1. Why is cross-entropy used for classification instead of MSE?** Cross-entropy is the negative log-likelihood of a categorical model, so minimizing it is maximum likelihood. With a softmax or sigmoid, its gradient is p − y, which stays large when the model is confidently wrong; MSE through a sigmoid has a gradient scaled by p(1 − p), which vanishes in exactly that case, and the MSE loss is non-convex in the logits.

**Q2. MSE or MAE: which would you use, and what does each predict?** MSE predicts the conditional mean and suits roughly Gaussian errors where large mistakes are much worse. MAE predicts the conditional median and is robust to outliers and heavy tails. Choose by the noise and by which summary the business needs.

**Q3. What is Huber loss and how do you choose δ?** It is quadratic for residuals up to δ and linear beyond, so it is smooth near the optimum but limits each outlier's pull to δ. Choose δ near the scale of typical, trustworthy errors, for example a multiple of a robust noise estimate such as the median absolute deviation, or tune it on validation data.

**Q4. What is the relationship between cross-entropy and KL divergence?** H(p, q) = H(p) + D_KL(p‖q). The entropy of the true labels does not depend on the model, so minimizing cross-entropy minimizes the KL divergence. KL is non-negative, zero only when the distributions match, and not symmetric.

**Q5. Derive the gradient of softmax cross-entropy with respect to the logits.** L = −z_y + log Σ_j e^{z_j}. Differentiating, ∂L/∂z_k = −\[k = y\] + e^{z_k}/Σ_j e^{z_j} = p_k − y_k. So the gradient is the predicted distribution minus the one-hot label.

**Q6. Why do PyTorch losses take logits rather than probabilities?** Computing log(softmax(z)) in two steps can overflow in exp or produce log(0). Fusing them with the log-sum-exp trick (subtract the maximum logit) is numerically stable, so `CrossEntropyLoss` and `BCEWithLogitsLoss` take raw logits.

**Q7. What does label smoothing do, and when can it hurt?** It replaces one-hot targets with (1 − ε) on the true class and ε/K spread over all classes. It stops logits from growing without bound, improves calibration and generalization, and is standard in image and translation models. It can hurt knowledge distillation (teachers trained with it pass on less information about class similarity) and tasks that need very confident probabilities.

**Q8. Your regression model's RMSE is good but users complain about large errors on a few items. What do you look at?** The error distribution, not just its average: tail percentiles of the absolute error, and the error by segment. If certain items are systematically mis-predicted, look for missing features or bad labels; if the business cares about the tail, train for a high quantile or weight the loss toward those items.

**Q9. How would you train a model to output a 90% prediction interval?** Train two quantile models (or one model with two outputs) with quantile loss at τ = 0.05 and τ = 0.95, then check on validation data that about 90% of true values fall inside. Conformal prediction can adjust the interval to guarantee that coverage.

**Q10. What loss would you use for a ranking problem such as search results?** Pointwise losses (cross-entropy on click or no click) are simple and work well with large data. Pairwise losses (RankNet, BPR) learn which of two items should rank higher. Listwise losses (LambdaRank, softmax over the list) optimize the whole ordering and track ranking metrics like NDCG more closely.

## 8.6 Cheat sheet and pitfalls

**Remember:**

- MSE ↔ Gaussian noise ↔ conditional mean; MAE ↔ Laplace noise ↔ conditional median; quantile loss ↔ any quantile.
- Cross-entropy = negative log-likelihood; H(p, q) = H(p) + KL(p‖q).
- Softmax + cross-entropy gradient = p − y; sigmoid + binary cross-entropy gradient = p − y.
- Huber: ½r² inside δ, linear outside; gradient = r clipped to ±δ.
- Stability: subtract the maximum logit; pass logits to fused losses.
- Label smoothing: target (1 − ε)y + ε/K.
- Train on a differentiable surrogate; select on the business metric.

**Pitfalls that cost candidates:**

- Using MSE for classification, or saying cross-entropy and log loss are different things.
- Forgetting that MSE predicts the mean, then being surprised by poor typical-case predictions on skewed data.
- Applying softmax before a loss that expects logits, which double-applies it.
- Calling KL divergence a distance or treating it as symmetric.
- Comparing RMSE and MAE values directly as if they were on the same footing.
- Choosing the loss without asking what decision the prediction feeds.

## 8.7 Further reading

<ol start="15">
  <li><a href="https://en.wikipedia.org/wiki/Cross_entropy">Cross-entropy</a> (Wikipedia)</li>
  <li><a href="https://en.wikipedia.org/wiki/Mean_squared_error">Mean squared error</a> (Wikipedia)</li>
  <li><a href="https://en.wikipedia.org/wiki/Mean_absolute_error">Mean absolute error</a> (Wikipedia)</li>
  <li><a href="https://en.wikipedia.org/wiki/Huber_loss">Huber loss</a> (Wikipedia)</li>
</ol>
