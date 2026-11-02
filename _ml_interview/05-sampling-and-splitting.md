---
title: "Sampling and Data Splitting"
chapter: 5
permalink: "/blog/cracking-the-ml-interview/sampling-and-splitting/"
updated: 2026-11-01
description: "How you sample data decides what your model can learn, and how you split it decides whether your evaluation can be trusted."
---

How you sample data decides what your model can learn, and how you split it decides whether your evaluation can be trusted. A wrong split is the most common reason an offline score of 0.95 becomes 0.70 in production, so interviewers probe this topic in both fundamentals and design rounds.

## 5.1 Intuition

**Population, frame and sample.** The **population** is everything you want the model to work on: all future loan applicants, all photos users will upload. You rarely see it all. The **sampling frame** is the part you can actually draw from, and the **sample** is what you draw. **Sampling bias** appears when the frame or the method makes some parts of the population more likely to be included than others, and no amount of data volume fixes it.

**Probability sampling** gives every unit a known, non-zero chance of selection, so results can be generalized:

| Method        | How it works                                                                        | Use it when                                       | Watch out for                                          |
| ------------- | ----------------------------------------------------------------------------------- | ------------------------------------------------- | ------------------------------------------------------ |
| Simple random | Every unit has an equal chance                                                      | You have a complete list and no special structure | Rare groups may be missed by chance                    |
| Systematic    | Take every k-th unit after a random start                                           | Sampling from a list or stream                    | Hidden periodic patterns in the order                  |
| Stratified    | Split the population into groups (**strata**), then sample within each              | Rare but important groups must be represented     | Needs the group labels in advance                      |
| Cluster       | Randomly choose whole groups (schools, stores), then use all or some of their units | Listing every unit is impractical                 | Units within a cluster are similar, so precision drops |

**Non-probability sampling** selects units by convenience or judgment. It is cheaper but can be badly biased:

| Method               | How it works                                  | Typical bias                        |
| -------------------- | --------------------------------------------- | ----------------------------------- |
| Convenience          | Use whatever data is easy to get              | Over-represents easy-to-reach users |
| Voluntary response   | People choose to take part (reviews, surveys) | Over-represents strong opinions     |
| Judgment (purposive) | Experts pick "representative" units           | Reflects the experts' assumptions   |
| Snowball             | Participants recruit others                   | Over-represents connected networks  |
| Quota                | Fill fixed numbers per group, non-randomly    | Bias within each group remains      |

Most ML training data is a convenience sample: whatever the product happened to log. Say this explicitly in design rounds, and name which users or situations it under-represents.

**Sampling methods specific to ML:**

- **Reservoir sampling** keeps a uniform random sample of fixed size k from a stream whose length is unknown or too large to store. It is a favourite coding question (5.2 has the proof).
- **Weighted sampling** picks items with probability proportional to a weight, for example to oversample recent or high-value events.
- **Importance sampling** draws from a convenient distribution q but reweights each sample by p/q to estimate quantities under the distribution p you care about. It underlies off-policy evaluation of recommenders, where logs come from the old policy.
- **Negative sampling** trains recommenders and embedding models by comparing each positive pair (user, item clicked) with a few randomly drawn negatives instead of every possible item.
- **Hard-negative mining** deliberately picks negatives the model currently confuses with positives.
- **Active learning** sends the examples the model is least sure about to human labelers, to get the most improvement per label.
- **Up-sampling and down-sampling** rebalance rare classes; Chapter 6 covers them in depth.

**Splitting: train, validation, test.** The **training set** fits the parameters, the **validation set** chooses hyperparameters and models, and the **test set** estimates final performance once, at the end. Using the test set for decisions turns it into a second validation set, and its score stops being an honest estimate.

- Common ratios are 70/15/15 or 80/10/10 for thousands to hundreds of thousands of rows. With millions of rows, 98/1/1 is fine, because 1% is already large enough for a precise estimate. With very little data, use cross-validation (Chapter 9) instead of a single validation set.
- The split must mimic how the model meets new data in production. That one principle drives every choice below.

| Split      | How                                                    | Use when                                                          | What goes wrong if you skip it                              |
| ---------- | ------------------------------------------------------ | ----------------------------------------------------------------- | ----------------------------------------------------------- |
| Random     | Shuffle rows, cut into parts                           | Rows are independent and identically distributed                  | Nothing, if rows truly are independent                      |
| Stratified | Random within each class                               | Imbalanced classes or small data                                  | Test set may contain almost no positives                    |
| Group      | All rows of one entity go to the same split            | Repeated users, patients, sessions, or near-duplicate images      | The model memorizes entities; scores are inflated           |
| Time-based | Train on earlier data, validate and test on later data | Forecasting, and any model that predicts the future from the past | Future information leaks into training; scores are inflated |

<figure class="figure-pending"><div class="figure-pending__box">diagram coming soon</div><figcaption>choosing a split · 3 questions, 4 strategies</figcaption></figure>

**Data leakage** is any way information from the evaluation data, or from the future, reaches training. The classic forms:

- **Duplicate leakage:** the same or near-identical rows appear in train and test (re-uploaded photos, copied text).
- **Group leakage:** different rows of the same user or patient sit on both sides.
- **Temporal leakage:** training uses data from after the prediction time, including features computed with future data (Chapter 1's point-in-time correctness).
- **Preprocessing leakage:** scalers, imputers, PCA or target encoders are fitted on all data before splitting.
- **Target leakage:** a feature that is only known because of the label, such as "refund issued" when predicting fraud.

## 5.2 Core math

**Reservoir sampling (Algorithm R).** Keep the first k items. For the t-th item after that (t > k, counting from 1), draw j uniformly from 1 to t; if j ≤ k, replace reservoir slot j with the new item. Memory is O(k) and the stream is read once.

**Why every item ends up with probability k/n**, by induction. Suppose that after t items each one is in the reservoir with probability k/t. Item t + 1 enters with probability k/(t + 1). An item already in the reservoir is evicted only if the new item enters _and_ lands on its slot, which happens with probability k/(t + 1) × 1/k = 1/(t + 1). So it survives with probability:

$$
\frac{k}{t}\left(1 - \frac{1}{t+1}\right) = \frac{k}{t}\cdot\frac{t}{t+1} = \frac{k}{t+1}
$$

Every item, old or new, now has the same probability k/(t + 1), and the claim holds at every step, ending at k/n.

**Weighted reservoir sampling (A-Res).** To sample k items with probability proportional to weights w_i from a stream, give each item the key u_i^(1/w_i), with u_i uniform on (0, 1), and keep the k largest keys in a min-heap. Heavier items get keys closer to 1.

**Stratified sampling.** With strata h of sizes N_h out of N, **proportional allocation** samples n_h = n·N_h/N from each. **Neyman allocation** also weights by each stratum's standard deviation σ_h, sending more samples where values vary most. The population mean is estimated by weighting each stratum's sample mean:

$$
n_h^{\text{prop}} = n\,\frac{N_h}{N}, \qquad n_h^{\text{Neyman}} = n\,\frac{N_h\,\sigma_h}{\sum_{g} N_g\,\sigma_g}, \qquad \bar x_{\text{st}} = \sum_h \frac{N_h}{N}\,\bar x_h
$$

Its variance is never larger than simple random sampling's under proportional allocation, and much smaller when strata are internally similar.

**Importance sampling.** To estimate an expectation under distribution p using samples from q, reweight each sample by the **importance weight** p/q:

$$
\mathbb{E}_{p}\big[f(X)\big] = \mathbb{E}_{q}\Big[f(X)\,\frac{p(X)}{q(X)}\Big] \approx \frac{1}{n}\sum_{i=1}^{n} f(x_i)\,\frac{p(x_i)}{q(x_i)}, \qquad x_i \sim q
$$

The estimate is unbiased, but its variance explodes when q is tiny where p is large, because a few samples get huge weights. In off-policy evaluation this estimator is called **IPS (inverse propensity scoring)**, and the weights are usually clipped to keep the variance manageable.

**Negative sampling** in word2vec draws negatives from the unigram distribution raised to the 3/4 power, which samples rare words more often than their raw frequency:

$$
P_{\text{neg}}(w) = \frac{\text{count}(w)^{0.75}}{\sum_{w'} \text{count}(w')^{0.75}}
$$

**How big must the test set be?** An accuracy estimate a on n independent test rows has standard error √(a(1 − a)/n). At a = 0.9 and n = 1,000, the standard error is about 0.0095, so the 95% interval is roughly ±1.9 points: too wide to trust a 1-point improvement. Narrowing it to ±0.5 points needs about 1.96² × 0.09 / 0.005² ≈ 13,800 rows.

$$
\text{SE}(a) = \sqrt{\frac{a(1-a)}{n}}, \qquad n \approx \frac{z^2\, a(1-a)}{h^2} \ \ \text{for a half-width } h
$$

## 5.3 Code

**Reservoir sampling and a uniformity check:**

```python
import random, heapq
from collections import Counter

def reservoir_sample(stream, k, rng=random):
    reservoir = []
    for i, item in enumerate(stream):        # i is 0-based: item is the (i+1)-th seen
        if i < k:
            reservoir.append(item)
        else:
            j = rng.randint(0, i)            # uniform over 0..i inclusive
            if j < k:
                reservoir[j] = item          # kept with probability k/(i+1)
    return reservoir

rng = random.Random(0)
counts = Counter()
for _ in range(100_000):
    counts.update(reservoir_sample(range(10), 3, rng))
print(sorted(counts.items()))
# every item appears about 30,000 times (from 29,810 to 30,166): probability 3/10 each

def weighted_reservoir(stream, k, rng=random):   # A-Res: keep the k largest u ** (1/w)
    heap = []
    for item, w in stream:
        key = rng.random() ** (1.0 / w)
        if len(heap) < k:
            heapq.heappush(heap, (key, item))
        elif key > heap[0][0]:
            heapq.heapreplace(heap, (key, item))
    return [item for _, item in heap]
# with weights a=1, b=1, c=2, d=4 and k=1, 50,000 runs pick a 12.7%, b 12.3%, c 25.1%, d 49.9%
# (exact: 12.5%, 12.5%, 25%, 50%)
```

**Stratified, group and time-based splits, and what leakage does to a score:**

```python
import numpy as np
from sklearn.model_selection import train_test_split, GroupShuffleSplit, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier

# Stratified: 5% positives
y = np.array([1] * 50 + [0] * 950)
X = np.arange(1000).reshape(-1, 1)
_, _, _, y_te = train_test_split(X, y, test_size=0.1, random_state=3)
_, _, _, y_te_s = train_test_split(X, y, test_size=0.1, random_state=3, stratify=y)
print(y_te.sum(), y_te_s.sum())   # 4 vs exactly 5 positives in a test set of 100

# Group leakage: 200 users x 20 rows; each user's label is random, unrelated to the features
rng = np.random.default_rng(0)
user_vec = rng.normal(size=(200, 10))
user_label = rng.integers(0, 2, 200)
groups = np.repeat(np.arange(200), 20)
Xg = user_vec[groups] + rng.normal(scale=0.1, size=(4000, 10))  # rows of a user look alike
yg = user_label[groups]
model = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)

X_tr, X_te, y_tr, y_te = train_test_split(Xg, yg, test_size=0.25, random_state=0)
print(model.fit(X_tr, y_tr).score(X_te, y_te))      # 0.998  (random split)

tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0).split(Xg, yg, groups))
print(model.fit(Xg[tr], yg[tr]).score(Xg[te], yg[te]))  # 0.475  (group split: chance level)

# Time-based: each fold trains on the past and tests on the next block
for tr, te in TimeSeriesSplit(n_splits=3).split(np.arange(12)):
    print(tr, te)
# [0 1 2] [3 4 5]
# [0 1 2 3 4 5] [6 7 8]
# [0 1 2 3 4 5 6 7 8] [ 9 10 11]
```

The leakage demo is worth remembering. The labels carry no signal at all, yet a random split reports 99.8% accuracy, because the model memorizes each user's rows and recognizes the same users in the test set. The group split reveals the truth: 47.5%, a coin flip. TimeSeriesSplit's `gap` argument leaves rows out between train and test when features use trailing windows.

## 5.4 In system design

- **Sampling at scale.** A large ads or feed system logs billions of impressions a day, almost all of them non-clicks. A standard design keeps every positive and keeps negatives at a rate w (say 10%). The model then over-predicts clicks, so its outputs must be **recalibrated** before they are used for bidding or ranking across models. If q is the predicted probability on the down-sampled data:

$$
p = \frac{q}{q + (1 - q)/w}
$$

- **Stable, hash-based splits.** Assign entities to splits with a deterministic hash, such as hash(user_id) mod 100 < 10 → test. The assignment survives retraining and data growth, never moves a user between splits, and needs no stored split table.
- **Evaluation sets mirror deployment.** Use a time-based holdout (train on weeks 1 to 8, test on week 9) and **backtesting**: repeat the evaluation over several rolling windows to see whether the model is stable over time. Split by user to measure cold-start performance, and by region or device to measure generalization to new segments.
- **Slice-stratified golden sets.** Build a fixed, carefully labeled evaluation set stratified by the slices that matter (country, device, new vs returning users, rare classes), and report metrics per slice, not just overall.
- **Feedback loops and selection bias.** A deployed model decides what gets shown, so the logs only contain outcomes for what the old model chose. Training on them reinforces its past choices. Reserve a small **exploration** slice (for example 1% random recommendations) to collect unbiased data, and use IPS-weighted evaluation for new policies.
- **Version the splits.** Store the split definition (hash function, cut-off dates, dedup rules) with the dataset version, so every experiment is reproducible and comparable.

## 5.5 Interview Q&A

**Q1. Sample k items uniformly from a stream of unknown length.** Use reservoir sampling: keep the first k items, then replace a random slot with the t-th item with probability k/t. It needs O(k) memory and one pass, and induction shows every item ends up selected with probability k/n.

**Q2. Stratified sampling or cluster sampling: what's the difference?** Stratified sampling divides the population into groups and samples _within every_ group, which guarantees representation and lowers variance. Cluster sampling randomly picks _some whole_ groups, which saves cost when listing every unit is hard, but raises variance because units within a cluster are similar.

**Q3. When is a random train/test split wrong?** When rows are not independent: repeated users or patients (use a group split), time-ordered data where the model predicts the future (use a time-based split), near-duplicates (deduplicate first), or heavy class imbalance on small data (stratify).

**Q4. How would you cross-validate a time-series model?** Use forward-chaining (expanding or rolling window) splits: train on data up to time t, validate on the next block, then move forward. Never shuffle, and leave a gap between train and validation when features use trailing windows or labels take time to arrive.

**Q5. Why not tune hyperparameters on the test set?** Every decision made by looking at the test score fits the model to that test set, so the final score becomes optimistically biased. Tune on validation data or cross-validation and touch the test set once at the end.

**Q6. Offline AUC is 0.95, but the model performs much worse in production. What do you check?** Leakage in the split (group, temporal, duplicates, target leakage), preprocessing fitted on all data, training-serving skew (features computed differently online), a training sample that does not represent production traffic, and distribution drift since the training window. A time-based, group-aware holdout usually reveals the gap before launch.

**Q7. You down-sampled negatives to 10% when training a click model. What else must you do?** Recalibrate the predicted probabilities with p = q / (q + (1 − q)/0.1), because the model learned on data with ten times the true click rate. Rankings within the model are unchanged, but anything that uses the probability itself, such as bids, thresholds or blending with other models, would be wrong.

**Q8. What is importance sampling, and what is its main risk?** Estimating an expectation under one distribution using samples from another, by weighting each sample by p/q. The risk is high variance: where q is much smaller than p, a few samples get enormous weights and dominate the estimate. Clipping or normalizing the weights trades a little bias for much lower variance.

**Q9. You have 10 million labeled rows. How would you split them?** Something like 98/1/1: 100,000 validation and test rows each already estimate accuracy to within a few tenths of a point. Before choosing ratios, choose the split type (random, group or time-based), because that matters far more than the percentages.

**Q10. How do you stop near-duplicates from leaking across splits?** Deduplicate before splitting: exact hashes for identical rows, perceptual hashes for images, MinHash or embedding similarity for text. Then split by duplicate cluster, so all near-copies land on the same side.

## 5.6 Cheat sheet and pitfalls

**Remember:**

- Probability sampling: simple random, systematic, stratified, cluster. Non-probability: convenience, voluntary, judgment, snowball, quota.
- Reservoir sampling: keep item t with probability k/t; O(k) memory; each item ends at k/n.
- Weighted reservoir: key = u^(1/w), keep the k largest.
- Importance sampling: weight by p/q; unbiased but can have high variance; IPS in off-policy evaluation.
- Negative down-sampling at rate w → recalibrate: p = q / (q + (1 − q)/w).
- Train fits, validation chooses, test reports once.
- Split to mimic production: time-based for the future, group for repeated entities, stratified for imbalance.
- Test-set standard error = √(a(1 − a)/n); ±0.5 points at 90% accuracy needs about 14,000 rows.

**Pitfalls that cost candidates:**

- Random-splitting time series or data with repeated users.
- Fitting scalers, encoders or PCA before splitting.
- Tuning on the test set, or reusing one test set for months of experiments.
- Down-sampling negatives and forgetting to recalibrate.
- Ignoring that logged data was chosen by the previous model.
- Arguing about 80/20 versus 70/30 while ignoring the split _type_.

## 5.7 Further reading

<ol start="8">
  <li><a href="https://www.scribbr.com/methodology/sampling-methods/">Sampling methods: types, techniques and examples</a> (Scribbr)</li>
  <li><a href="https://machinelearningmastery.com/train-test-split-for-evaluating-machine-learning-algorithms/">Train-test split for evaluating machine learning algorithms</a> (Machine Learning Mastery)</li>
</ol>
