---
title: "Class Imbalance"
chapter: 6
permalink: "/blog/cracking-the-ml-interview/class-imbalance/"
updated: 2026-11-01
description: "In most valuable ML problems, the class you care about is rare: fraud is under 1% of transactions, a disease may affect 1 patient in 1,000, and clicks are a few percent of ad impressions."
---

In most valuable ML problems, the class you care about is rare: fraud is under 1% of transactions, a disease may affect 1 patient in 1,000, and clicks are a few percent of ad impressions. Interviewers ask about imbalance constantly, and the best answers start with the metric and the decision threshold before reaching for resampling or a special loss.

## 6.1 Intuition

**Why accuracy fails.** If 98% of transactions are legitimate, a model that always predicts "legitimate" scores 98% accuracy and catches no fraud. Accuracy rewards getting the common class right, which is the easy part. The **imbalance ratio** (majority count divided by minority count, here 49:1) tells you how misleading accuracy will be.

**What actually goes wrong in training:**

- **The loss is dominated by the majority class.** With 49 negatives per positive, most of the gradient comes from negatives, so the model learns them well and treats positives as noise.
- **Too few minority examples** to learn their variety. This, not the ratio itself, is often the real problem: 1,000 fraud cases in 50,000 rows is learnable; 10 cases is not, whatever you do.
- **Easy negatives swamp the signal.** In object detection, a single image yields tens of thousands of candidate boxes, almost all obvious background. Each is easy, but together they contribute most of the loss. This is the problem focal loss was designed for.
- **The default 0.5 threshold is wrong.** A well-trained model on a 2% problem rarely predicts above 0.5 even for real positives, because positives are rare _a priori_.

**Three families of fixes:**

| Family                       | Methods                                                                                                         | What it changes                                   | Main risk                                                                                              |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------- | ------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| Data level                   | Random oversampling, random undersampling, SMOTE and variants, data augmentation, collecting more minority data | The training distribution the model sees          | Oversampling can overfit duplicated points; undersampling throws away data; both distort probabilities |
| Loss level (algorithm level) | Class weights, class-balanced loss, focal loss                                                                  | How much each example contributes to the gradient | Distorted probabilities; extra hyperparameters to tune                                                 |
| Decision level               | Threshold moving, cost-sensitive decisions, prior correction                                                    | Only how scores are turned into decisions         | Needs a validation set and a clear cost model                                                          |

- **Random oversampling** duplicates minority rows. **SMOTE (Synthetic Minority Over-sampling Technique)** instead creates new minority points by interpolating between a minority point and one of its nearest minority neighbours. It helps some models on small tabular data, but it can create unrealistic points in overlapping regions and is rarely worth it on large datasets or with tree ensembles.
- **Random undersampling** drops majority rows. It speeds up training, which is why large systems down-sample negatives (Chapter 5), but it discards information.
- **Class weights** multiply each example's loss by a weight for its class, usually inverse frequency. They have the same effect in expectation as oversampling, without duplicating data.
- **Class-balanced loss** (reference 10) argues that inverse frequency over-corrects, because extra samples of a common class add diminishing new information. It weights classes by the inverse of their **effective number of samples** instead (6.2).
- **Focal loss** (references 11 and 12) shrinks the loss of examples the model already classifies confidently, so training concentrates on hard examples, whichever class they belong to.
- **Threshold moving** keeps the model as it is and picks the decision threshold that best fits the business costs, on validation data. It is the cheapest fix and often the most effective.

**Choose metrics that see the minority class:**

| Metric                                                                            | What it measures                                               | Use it when                                                     |
| --------------------------------------------------------------------------------- | -------------------------------------------------------------- | --------------------------------------------------------------- |
| Precision                                                                         | Of the cases flagged, the share that are truly positive        | False alarms are costly (manual review queues)                  |
| Recall (sensitivity)                                                              | Of the true positives, the share that are caught               | Misses are costly (disease screening, fraud)                    |
| F1 score                                                                          | The harmonic mean of precision and recall                      | You need one number and both errors matter                      |
| **PR-AUC (area under the precision-recall curve)**, also called average precision | Ranking quality, focused on the positive class                 | Positives are rare; the standard choice for imbalance           |
| **ROC-AUC (area under the receiver operating characteristic curve)**              | The chance a random positive is scored above a random negative | Comparing rankings when classes are less skewed                 |
| Recall at a fixed precision, or precision at k                                    | Performance at the operating point you will actually use       | Review capacity is fixed ("analysts can check 500 cases a day") |

ROC-AUC can look excellent on very imbalanced data, because its false positive rate divides by the huge number of negatives: 1,000 false alarms among a million negatives is only 0.1%. Precision does not hide them. The baseline PR-AUC of a random model equals the positive rate, so a PR-AUC of 0.24 on a 2.5% problem is about ten times better than chance.

**Imbalance is not always a problem.** If classes are well separated, or if you only need a ranking (who to review first), a plain model with a good metric and a tuned threshold is often all you need. Ask "what decision will be made with this score?" before changing the data.

## 6.2 Core math

**Weighted cross-entropy.** For an example with true class y and predicted probability p_y, multiply the usual loss by a class weight w_y. The common default is inverse frequency, with n_c examples in class c, N in total and C classes:

$$
\ell = -\,w_{y}\,\log p_{y}, \qquad w_c = \frac{N}{C\, n_c}
$$

In expectation, weighting a class by w is the same as duplicating each of its examples w times, which is why class weights and random oversampling behave so similarly.

**Class-balanced loss and the effective number of samples** (reference 10). The insight: data overlaps. The 1,000th photo of a cat adds less new information than the 10th. The authors model each new sample as either covering new ground or overlapping what is already covered, and derive the **effective number of samples** for a class with n examples, with a hyperparameter β between 0 and 1. The class-balanced loss scales any base loss L (softmax cross-entropy, sigmoid cross-entropy or focal loss) by the inverse of the effective number:

$$
E_n = \frac{1 - \beta^{\,n}}{1 - \beta}, \qquad \text{CB loss} = \frac{1 - \beta}{1 - \beta^{\,n_y}}\;L(p, y)
$$

β interpolates between two extremes. At β = 0, E_n = 1 for every class, so there is no re-weighting. As β → 1, E_n → n, which is plain inverse-frequency weighting. In between, E_n grows quickly for small classes and saturates for large ones. For counts of 10,000, 1,000, 100 and 10:

| β                       | Effective numbers      | Normalized weights                   |
| ----------------------- | ---------------------- | ------------------------------------ |
| 0.9                     | 10, 10, 10, 6.5        | Nearly equal: almost no re-weighting |
| 0.999                   | 1,000, 632, 95, 10     | 0.035, 0.056, 0.37, 3.54             |
| → 1 (inverse frequency) | 10,000, 1,000, 100, 10 | 0.004, 0.036, 0.36, 3.60             |

At β = 0.999 the rarest class keeps almost its inverse-frequency weight, but the head class is down-weighted far less (0.035 instead of 0.004). The paper suggests β of 0.99 to 0.9999, larger for larger datasets, and tuning it on validation data.

**Focal loss** (references 11 and 12). Write p_t for the predicted probability of the true class: p_t = p when y = 1 and p_t = 1 − p when y = 0. Cross-entropy is −log p_t. Focal loss multiplies it by a **modulating factor** (1 − p_t)^γ and, optionally, a class weight α_t:

$$
\text{FL}(p_t) = -\,\alpha_t\,(1 - p_t)^{\gamma}\,\log p_t
$$

- The **focusing parameter** γ ≥ 0 controls how strongly easy examples are down-weighted. γ = 0 gives back weighted cross-entropy.
- For a misclassified example, p_t is small, the factor is close to 1, and the loss is almost unchanged.
- For a well-classified example, p_t is close to 1 and the factor crushes the loss. With γ = 2, an example at p_t = 0.9 has 100 times less loss than under cross-entropy; at p_t = 0.6 it is 6 times less; at p_t = 0.1 only 1.2 times less.
- **α_t** balances positives and negatives. In the paper's detector, the best setting was γ = 2 with α = 0.25 for the _rare_ foreground class. That looks backwards, but the modulating factor already suppresses the many easy negatives so much that the positives then need slightly less weight.
- The paper also initializes the final layer's bias so the model starts by predicting a small probability π (0.01) for the rare class, which stops the flood of negatives from destabilizing early training.

<figure class="figure-pending"><div class="figure-pending__box">diagram coming soon</div><figcaption>computed from the focal loss formula · γ = 0, 0.5, 1, 2, 5</figcaption></figure>

The curves show the mechanism: every γ > 0 bends the loss down where p_t is high, so the thousands of easy examples in the shaded region stop dominating the total.

**The gradient**, with z the logit and p = σ(z), for a positive example (y = 1). The extra term is what keeps learning focused on hard cases:

$$
\frac{\partial\,\text{FL}}{\partial z} = \alpha\,(1 - p)^{\gamma}\Big(\gamma\, p\,\log p + p - 1\Big)
$$

At γ = 0 this is α(p − 1), the familiar cross-entropy gradient scaled by α.

**Threshold moving and cost-sensitive decisions.** If a false negative costs C_FN and a false positive costs C_FP, and the model's probabilities are calibrated, flag a case when the expected cost of not flagging it is higher:

$$
\text{flag if}\quad p \cdot C_{FN} > (1 - p)\cdot C_{FP} \quad\Longleftrightarrow\quad p > t^{*} = \frac{C_{FP}}{C_{FP} + C_{FN}}
$$

If missing fraud costs 50 times more than a false alarm, t\* = 1/51 ≈ 0.02, nowhere near 0.5. Without clear costs, choose the threshold that maximizes F1 or meets a precision or recall target on validation data.

**Prior correction after resampling.** Training on a resampled dataset changes the class priors, so predicted probabilities are inflated. If the positive rate was π in the real data and π′ in the training data, correct the predicted probability q with Bayes' rule:

$$
p = \frac{q\,\pi/\pi'}{q\,\pi/\pi' + (1 - q)\,(1 - \pi)/(1 - \pi')}
$$

The negative-down-sampling formula in Chapter 5 is a special case of this. Class weights distort probabilities in the same way, and need the same correction or a re-calibration step (Platt scaling or isotonic regression) on untouched validation data.

## 6.3 Code

**Focal loss and class-balanced weights in PyTorch.** Both are a few lines; interviewers sometimes ask you to write focal loss on the spot.

```python
import torch
import torch.nn.functional as F

def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    """Binary focal loss. logits and targets have the same shape; targets are 0/1 floats."""
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")  # -log p_t, numerically stable
    p = torch.sigmoid(logits)
    p_t = p * targets + (1 - p) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (alpha_t * (1 - p_t) ** gamma * ce).mean()

def class_balanced_weights(counts, beta=0.999):
    """Per-class weights 1/E_n, normalized to sum to the number of classes."""
    counts = torch.as_tensor(counts, dtype=torch.float)
    effective = (1 - beta ** counts) / (1 - beta)
    w = 1.0 / effective
    return w / w.sum() * len(counts)

print(class_balanced_weights([10_000, 1_000, 100, 10]))   # tensor([0.0350, 0.0560, 0.3700, 3.5390])

# Use the weights with any multi-class loss
weights = class_balanced_weights([10_000, 1_000, 100, 10])
loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

# Sanity check: with gamma = 0 and alpha = 0.5, focal loss is half of binary cross-entropy
logits, t = torch.tensor([3.0, -2.0, 0.5]), torch.tensor([1.0, 0.0, 1.0])
assert torch.isclose(2 * focal_loss(logits, t, alpha=0.5, gamma=0),
                     F.binary_cross_entropy_with_logits(logits, t))
```

The library `torchvision.ops.sigmoid_focal_loss` implements the same binary version. Computing −log p_t through `binary_cross_entropy_with_logits` rather than `torch.log(p)` avoids taking the log of 0 when the model is very confident.

**Which fix actually helps? One experiment, six approaches.** A synthetic dataset of 20,000 rows with 2.5% positives and some label noise; a logistic regression; the same test set for all (151 positives in 6,000 rows).

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (precision_recall_curve, average_precision_score,
                             roc_auc_score, f1_score)
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler

X, y = make_classification(n_samples=20_000, n_features=20, n_informative=6,
                           weights=[0.98, 0.02], flip_y=0.01, class_sep=0.8, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
lr = lambda **kw: LogisticRegression(max_iter=2000, **kw)

setups = {
    "plain":                 (lr(), X_tr, y_tr),
    "class_weight=balanced": (lr(class_weight="balanced"), X_tr, y_tr),
    "random oversampling":   (lr(), *RandomOverSampler(random_state=0).fit_resample(X_tr, y_tr)),
    "SMOTE":                 (lr(), *SMOTE(random_state=0).fit_resample(X_tr, y_tr)),
    "random undersampling":  (lr(), *RandomUnderSampler(random_state=0).fit_resample(X_tr, y_tr)),
}
for name, (model, Xf, yf) in setups.items():
    p = model.fit(Xf, yf).predict_proba(X_te)[:, 1]
    print(name, f1_score(y_te, p >= 0.5), average_precision_score(y_te, p), roc_auc_score(y_te, p))

# Threshold moving: keep the plain model, pick the F1-best threshold on a validation split
X_fit, X_val, y_fit, y_val = train_test_split(X_tr, y_tr, test_size=0.25, stratify=y_tr, random_state=0)
prec, rec, thr = precision_recall_curve(y_val, lr().fit(X_fit, y_fit).predict_proba(X_val)[:, 1])
f1 = 2 * prec * rec / (prec + rec + 1e-12)
best = thr[np.argmax(f1[:-1])]          # the last (precision, recall) pair has no threshold
print(best)                             # 0.064

p = setups["plain"][0].predict_proba(X_te)[:, 1]
print(f1_score(y_te, p >= best))        # 0.278 on the test set
```

| Approach                   | Threshold | Precision | Recall    | F1        | PR-AUC    | ROC-AUC   | Mean predicted probability |
| -------------------------- | --------- | --------- | --------- | --------- | --------- | --------- | -------------------------- |
| Always predict negative    | n/a       | n/a       | 0.000     | 0.000     | n/a       | n/a       | n/a                        |
| Plain                      | 0.5       | 0.222     | 0.013     | 0.025     | 0.241     | 0.850     | 0.024                      |
| Class weights (balanced)   | 0.5       | 0.087     | 0.795     | 0.157     | 0.190     | 0.854     | 0.340                      |
| Random oversampling        | 0.5       | 0.087     | 0.801     | 0.157     | 0.189     | 0.853     | 0.338                      |
| SMOTE                      | 0.5       | 0.090     | 0.788     | 0.162     | 0.194     | 0.853     | 0.316                      |
| Random undersampling       | 0.5       | 0.082     | 0.821     | 0.150     | 0.180     | 0.854     | 0.344                      |
| **Plain, tuned threshold** | **0.064** | **0.184** | **0.570** | **0.278** | **0.241** | **0.850** | **0.024**                  |

Always predicting negative scores 97.5% accuracy here; the plain model at 0.5 scores 97.4% while catching 2 of 151 positives.

**What the table teaches:**

1. **Rebalancing did not improve the model's ranking.** ROC-AUC barely moves (0.850 to 0.854) and PR-AUC even drops slightly. For a linear model, re-weighting mainly shifts the intercept, which moves every score up together.
2. **What rebalancing did was move the operating point**: from 1% recall to about 80%, at the cost of precision. You can get any point on that trade-off from the plain model by choosing a threshold.
3. **Threshold tuning won on F1** (0.278 against at most 0.162), with no change to training, and the probabilities stayed calibrated (mean prediction 0.024, close to the true 2.5% rate; the rebalanced models predict about 0.34).
4. **Flexible models change the picture somewhat.** Re-weighting changes which features and splits a tree ensemble or neural network learns, so it can improve ranking, especially with very few positives. Focal loss pays off most in dense object detection, where easy negatives outnumber positives by 1,000 to 1 inside every image. Treat rebalancing as a hyperparameter to validate, not a default.

## 6.4 In system design

- **Start from the decision and its costs.** A fraud system might auto-block above one score, send cases to human review in a middle band sized to the review team's daily capacity, and approve the rest. The thresholds come from costs and capacity, not from 0.5, and they are re-tuned as fraud rates change.
- **Report metrics that match the decision**: precision and recall at the operating threshold, PR-AUC for model comparison, and per-segment recall (new merchants, new countries), where rare-class performance usually breaks first.
- **Down-sample negatives at scale, then recalibrate** (Chapter 5). Ads and feed systems keep all positives and a fraction of negatives to save compute, and correct the probabilities afterwards because bids and ranking blends depend on them.
- **Label delay.** Fraud labels (chargebacks) can take weeks to arrive. Rows that look negative today may later turn positive, so the freshest training data is biased toward negatives. Wait for labels to mature, or model the delay.
- **Get more minority data.** Often the best fix is outside the model: active learning to label likely positives, rules that route suspicious cases to analysts and feed their labels back, data augmentation for images and text, or synthetic data checked carefully for realism.
- **When positives are extremely rare or unknown**, frame the problem as **anomaly detection**: learn what normal looks like (isolation forests, autoencoders) and flag deviations. Combine it with a supervised model once enough labeled cases exist.
- **Long-tailed classification** (thousands of product categories, most with few examples) uses class-balanced loss, focal loss, re-sampling schedules, or **decoupled training**: learn features with natural sampling, then re-train only the classifier layer on balanced data.
- **Object detection** is focal loss's home turf: one-stage detectors (RetinaNet, the YOLO family) score tens of thousands of candidate boxes per image, nearly all background.

## 6.5 Interview Q&A

**Q1. Your fraud model has 99% accuracy. Is it good?** Not necessarily. If fraud is 1% of transactions, predicting "not fraud" for everything also scores 99%. Ask for precision, recall and PR-AUC on the fraud class, at the threshold that will actually be used.

**Q2. ROC-AUC or PR-AUC for imbalanced data?** PR-AUC, usually. ROC-AUC's false positive rate divides by the huge number of negatives, so many false alarms barely move it; precision counts them directly. A random model's PR-AUC equals the positive rate, which gives a clear baseline. ROC-AUC is still useful for comparing rankings and does not change with the class ratio.

**Q3. Compare oversampling, undersampling and class weights.** Oversampling duplicates or synthesizes minority rows: no data lost, but more training time and a risk of overfitting duplicates. Undersampling drops majority rows: faster, but it discards information. Class weights reweight the loss: the same expected effect as oversampling without extra rows. All three shift the predicted probabilities upward, so recalibrate if probabilities are used.

**Q4. How does SMOTE work, and when does it hurt?** For a minority point, pick one of its k nearest minority neighbours and create a new point at a random position on the line between them. It can hurt when classes overlap (synthetic points land in majority territory), with high-dimensional or categorical features where interpolation makes no sense, and with tree ensembles that gain little from it. Apply it only to the training fold, never before splitting.

**Q5. Explain focal loss. Why does it help?** It multiplies cross-entropy by (1 − p_t)^γ, which shrinks the loss of well-classified examples. When easy negatives vastly outnumber everything else, as with background boxes in object detection, they otherwise dominate the total loss and gradient even though each one contributes little. With γ = 2, an example at p_t = 0.9 contributes 100 times less than under cross-entropy, so training concentrates on hard cases.

**Q6. What do α and γ do in focal loss?** γ sets how strongly easy examples are down-weighted (0 = cross-entropy, 2 is the common default). α re-weights the positive class versus the negative. They interact: with γ = 2, the paper's best α for the rare class was 0.25, because the modulating factor already removed most of the negatives' weight.

**Q7. What is the effective number of samples?** From the class-balanced loss paper: because samples overlap in information, a class with n samples behaves like E_n = (1 − β^n)/(1 − β) unique samples. Weighting by 1/E_n gives a smooth middle ground between no re-weighting (β = 0) and inverse frequency (β → 1), and avoids hugely over-weighting the rarest classes.

**Q8. Why is a 0.5 threshold usually wrong for imbalanced problems?** A calibrated model's probabilities reflect the base rate, so with 2% positives even true positives often score below 0.5. The right threshold depends on the costs: flag when p > C_FP / (C_FP + C_FN), or pick the threshold that meets the required precision or recall on validation data.

**Q9. You trained with class weights. Can you use the model's probabilities directly?** No. Weighting shifts the effective class prior, so probabilities are inflated for the minority class. Apply a prior correction or recalibrate on untouched validation data (Platt scaling or isotonic regression) before using them in expected-cost or bidding calculations.

**Q10. You have only 50 positive examples. What do you do?** Use stratified cross-validation and report confidence intervals, since metrics will be noisy. Prefer simpler or regularized models, strong features and pretrained embeddings. Get more positives: active learning, weak supervision from rules, augmentation. Consider anomaly detection, and make sure every positive is checked for label quality.

## 6.6 Cheat sheet and pitfalls

**Remember:**

- Fix in this order: right metric, then threshold, then loss or data changes, validated like any hyperparameter.
- Metrics: precision, recall, F1, PR-AUC (random baseline = positive rate); avoid accuracy.
- Class weights: w_c = N/(C·n_c); in expectation equal to oversampling.
- Class-balanced loss: weight 1/E_n with E_n = (1 − β^n)/(1 − β); β = 0.99 to 0.9999.
- Focal loss: −α_t(1 − p_t)^γ log p_t; γ = 2, α = 0.25; 100× less loss at p_t = 0.9.
- Cost-optimal threshold: t\* = C_FP / (C_FP + C_FN).
- Resampling or weighting distorts probabilities: apply a prior correction or recalibrate.
- SMOTE interpolates between minority neighbours; fit it on the training fold only.

**Pitfalls that cost candidates:**

- Quoting accuracy, or ROC-AUC alone, on a 1% problem.
- Resampling before the train/test split, which leaks duplicated or synthetic points into the test set.
- Evaluating on a resampled (balanced) test set, which hides real-world precision.
- Using the probabilities of a re-weighted model as if they were calibrated.
- Reaching for SMOTE or focal loss before trying a tuned threshold.
- Forgetting that a too-small number of positives, not the ratio, is often the real limit.

## 6.7 Further reading

<ol start="10">
  <li>Cui et al., <a href="https://arxiv.org/pdf/1901.05555.pdf">Class-Balanced Loss Based on Effective Number of Samples</a> (CVPR 2019)</li>
  <li>Lin et al., <a href="https://arxiv.org/pdf/1708.02002.pdf">Focal Loss for Dense Object Detection</a> (ICCV 2017)</li>
  <li><a href="https://medium.com/swlh/focal-loss-an-efficient-way-of-handling-class-imbalance-4855ae1db4cb">Focal loss: an efficient way of handling class imbalance</a> (The Startup, Medium)</li>
</ol>
