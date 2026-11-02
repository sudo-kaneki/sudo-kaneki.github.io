---
title: "Interpretability"
chapter: 4
permalink: "/blog/cracking-the-ml-interview/interpretability/"
updated: 2026-11-01
description: "An accurate model nobody can explain is hard to debug, hard to trust and, in regulated domains, sometimes illegal to deploy."
---

An accurate model nobody can explain is hard to debug, hard to trust and, in regulated domains, sometimes illegal to deploy. Interviewers test whether you can explain a model's behavior at two levels, the whole model and a single prediction, and whether you know where popular explanation methods mislead.

## 4.1 Intuition

**Why explain a model?**

- **Debugging.** Explanations expose shortcuts: a pneumonia model that keys on the hospital's scanner watermark rather than the lungs, or a fraud model leaning on a feature that leaks the label.
- **Trust and adoption.** Doctors, loan officers and moderators act on predictions only if they understand them.
- **Regulation.** Lenders in many jurisdictions must give applicants the main reasons for a credit denial. The EU's **GDPR (General Data Protection Regulation)** gives people rights around automated decisions, and the EU AI Act adds transparency duties for high-risk systems.
- **Fairness auditing.** Checking whether a protected attribute, or a proxy for it such as postcode, drives decisions.
- **Scientific insight.** Learning which factors actually relate to the outcome.

**Three axes classify every method:**

- **Intrinsic or post-hoc.** An **intrinsically interpretable** model is readable by design: linear models, small trees, **GAMs (generalized additive models)**, which add up one learned curve per feature. A **post-hoc** method explains a model after training, usually a black box such as a boosted ensemble or a neural network.
- **Global or local.** A **global** explanation describes the whole model ("income matters most overall"). A **local** explanation describes one prediction ("this loan was declined mainly because of the debt ratio").
- **Model-specific or model-agnostic.** Model-specific methods use the model's internals (tree structure, gradients); model-agnostic methods only query its predictions, so they work on anything.

<figure class="figure-pending"><div class="figure-pending__box">diagram coming soon</div><figcaption>interpretability methods · intrinsic or post-hoc, global or local</figcaption></figure>

| Method                                                     | Scope                           | Model-agnostic?                        | Question it answers                                                           |
| ---------------------------------------------------------- | ------------------------------- | -------------------------------------- | ----------------------------------------------------------------------------- |
| Coefficients                                               | Global                          | No (linear models)                     | How does y change per unit of each feature?                                   |
| Impurity importance                                        | Global                          | No (trees)                             | How much did splits on this feature reduce impurity during training?          |
| Permutation importance                                     | Global                          | Yes                                    | How much worse does the model get if this feature's information is destroyed? |
| **PDP (partial dependence plot)**                          | Global                          | Yes                                    | On average, how does the prediction change as one feature varies?             |
| **ICE (individual conditional expectation)** curves        | Local                           | Yes                                    | The same, for each individual row                                             |
| **LIME (Local Interpretable Model-agnostic Explanations)** | Local                           | Yes                                    | Which features drive the model near this point, by a simple local fit?        |
| **SHAP (SHapley Additive exPlanations)**                   | Local, and global when averaged | Yes, with fast model-specific versions | How much did each feature push this prediction away from the average?         |
| Counterfactual explanations                                | Local                           | Yes                                    | What is the smallest change that would flip the decision?                     |
| Saliency maps, integrated gradients                        | Local                           | No (differentiable models)             | Which input pixels or tokens most affect the output?                          |

**What makes an explanation good?** It must be **faithful** (it reflects what the model actually does), not just **plausible** (it sounds reasonable to a human). It should be **stable**, so similar inputs get similar explanations, and **understandable** to its audience. Faithfulness is the one people forget: attention weights in transformers and an LLM's written chain of thought can look like explanations without reliably reflecting the computation behind the output.

**Explanations describe the model, not the world.** A SHAP value says how a feature moved this model's output, not that changing the feature would change the real outcome. Causal claims need causal methods.

## 4.2 Core math

**Permutation importance.** Score the model on held-out data (s), then shuffle column j, which breaks its link to the target while keeping its distribution, and score again. Repeat K times and average the drop:

$$
I_j = s - \frac{1}{K}\sum_{k=1}^{K} s_{k,j}
$$

It measures what the model _relies on_, using any metric you care about. Two caveats: compute it on validation data, not training data, or you measure what the model memorized; and with **correlated features**, shuffling one lets its twin stand in, so both look less important than they are.

**Impurity importance**, for comparison, sums the impurity decrease of every split on feature j, averaged over trees (Chapter 3). It is free to compute but uses training data, and it favours features with many distinct values, which offer more possible split points.

**Partial dependence and ICE.** The PDP for feature j averages the model's prediction over the data while forcing feature j to a value v. Each term of the average, taken per row, is that row's ICE curve:

$$
\text{PD}_j(v) = \frac{1}{n}\sum_{i=1}^{n} f\big(v,\ x_{i,-j}\big), \qquad \text{ICE}_{i,j}(v) = f\big(v,\ x_{i,-j}\big)
$$

Here x\_{i,−j} means row i with feature j removed. If the ICE curves are parallel, the PDP summarizes them well; if they cross, feature j interacts with others and the average hides it. Both assume feature j can vary independently of the rest, so with correlated features they evaluate impossible rows (a 40 m² flat with eight bedrooms). **ALE (accumulated local effects)** plots avoid this by averaging only over nearby, realistic rows.

**LIME.** To explain prediction f(x), sample perturbed points z around x, query the model, and fit a simple model g (usually sparse linear) that matches f near x. Points are weighted by proximity π_x, and Ω penalizes complexity:

$$
\xi(x) = \arg\min_{g \in G}\ \sum_{z} \pi_x(z)\,\big(f(z) - g(z)\big)^2 + \Omega(g), \qquad \pi_x(z) = \exp\!\Big(-\frac{D(x, z)^2}{\sigma^2}\Big)
$$

g's coefficients are the explanation: **local slopes**. The kernel width σ decides how local "local" is, and different reasonable choices can give different explanations (the code in 4.3 shows the slopes changing about fourfold).

**Shapley values and SHAP.** Shapley values come from cooperative game theory: the features are players who jointly produce the prediction, and each gets a fair share of the payout. Feature j's share averages its marginal contribution over every subset S of the other features F \\ {j}:

$$
\phi_j = \sum_{S \subseteq F \setminus \{j\}} \frac{|S|!\,(p - |S| - 1)!}{p!}\,\Big[v\big(S \cup \{j\}\big) - v(S)\Big]
$$

The value function v(S) is the model's expected output when only the features in S are known. The common **interventional** version fills the unknown features from a background dataset B:

$$
v(S) = \frac{1}{|B|}\sum_{b \in B} f\big(x_S,\ b_{-S}\big)
$$

Shapley values are the **only** attribution satisfying four fairness properties, which is why SHAP became the standard:

1. **Efficiency (local accuracy):** the attributions add up exactly to the prediction minus the average prediction, Σ_j φ_j = f(x) − E\[f(X)\].
2. **Symmetry:** two features that contribute identically get equal credit.
3. **Dummy:** a feature that never changes the output gets zero.
4. **Additivity:** for a sum of models, attributions add up.

Exact computation needs all 2^p subsets. **KernelSHAP** estimates it model-agnostically with a weighted linear regression over sampled subsets (LIME with a special kernel). **TreeSHAP** computes exact values for tree ensembles in polynomial time, which is why SHAP is so common with gradient-boosted trees. Averaging |φ_j| over many rows gives a global importance ranking.

**SHAP and LIME answer different questions.** SHAP says how much feature j's _actual value_ moved this prediction away from the average prediction. LIME says how the prediction would _change_ if feature j moved a little. For the same row, a feature can have a positive SHAP value but a negative LIME slope, as the code in 4.3 shows.

**Integrated gradients (IG)**, for differentiable models such as neural networks, integrates the gradient along a straight path from a **baseline** input x′ (a black image, an all-padding sentence) to the actual input x:

$$
\text{IG}_j(x) = (x_j - x'_j)\int_{0}^{1} \frac{\partial f\big(x' + \alpha\,(x - x')\big)}{\partial x_j}\, d\alpha, \qquad \sum_j \text{IG}_j(x) = f(x) - f(x')
$$

The right-hand identity, **completeness**, is IG's version of SHAP's efficiency. Plain gradients (saliency maps) are cheaper but noisy and can be zero where the model is saturated; IG fixes that, at the cost of choosing a sensible baseline.

**Counterfactual explanations** search for the closest input x′ that gets the desired outcome, changing only features a person can act on: "the loan would be approved if the debt ratio were below 0.35". For a distance d and target outcome y′:

$$
x' = \arg\min_{x'}\ d(x, x') \quad \text{subject to} \quad f(x') = y'
$$

## 4.3 Code

One script, four lessons, on the breast-cancer data (y = 1 means benign). The SHAP library would do parts 3 and 4 in two lines each; writing them by hand once is the best way to understand what those two lines compute.

```python
import itertools, math
import numpy as np, pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance, partial_dependence
from sklearn.linear_model import Ridge

X, y = load_breast_cancer(return_X_y=True, as_frame=True)              # y = 1 means benign
X["random_id"] = np.random.default_rng(0).integers(0, 10_000, len(X))  # pure noise, many values
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)

# 1. Impurity vs permutation importance
rf = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1).fit(X_tr, y_tr)
impurity = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
perm = permutation_importance(rf, X_te, y_te, n_repeats=20, random_state=0, n_jobs=-1)
perm = pd.Series(perm.importances_mean, index=X.columns).sort_values(ascending=False)
# impurity:    worst radius 0.130, worst area 0.122, worst perimeter 0.118, worst concave points 0.115
# permutation: worst radius 0.0161, worst area 0.0042, area error 0.0024, worst texture 0.0021
# random_id:   impurity 0.0025, permutation 0.0

# 2. Partial dependence of the benign probability on one feature
pdp = partial_dependence(rf, X_tr, ["worst area"], grid_resolution=5)
# worst area  328    749    1170   1591   2013
# average     0.684  0.641  0.527  0.524  0.524

# 3. Exact Shapley values for a 4-feature model
feats = ["mean radius", "mean texture", "worst concave points", "worst area"]
gb = HistGradientBoostingClassifier(random_state=0).fit(X_tr[feats], y_tr)
f = lambda D: gb.predict_proba(D)[:, 1]
x = X_te[feats].iloc[[30]]                                  # a borderline case, f(x) = 0.522
background = X_tr[feats].sample(100, random_state=0)

def v(S):                                                   # interventional value function
    D = background.copy()
    for j in S:
        D[j] = x[j].values[0]
    return f(D).mean()

p = len(feats)
phi = {}
for j in feats:
    others = [k for k in feats if k != j]
    phi[j] = sum(math.factorial(len(S)) * math.factorial(p - len(S) - 1) / math.factorial(p)
                 * (v(S + (j,)) - v(S))
                 for r in range(p) for S in itertools.combinations(others, r))
# phi: mean radius +0.011, mean texture +0.012, worst concave points -0.271, worst area +0.049
# base value v({}) = 0.720; 0.720 + sum(phi) = 0.522 = f(x)   <- efficiency holds exactly

# 4. A LIME-style local surrogate, at two kernel widths
rng = np.random.default_rng(0)
mu, sd = X_tr[feats].mean(), X_tr[feats].std()
z0 = ((x - mu) / sd).values[0]                              # work in standardized units
for width in [0.25, 1.0]:
    Z = z0 + rng.normal(scale=width, size=(2000, p))        # perturb around x
    preds = f(pd.DataFrame(Z * sd.values + mu.values, columns=feats))
    w = np.exp(-((Z - z0) ** 2).sum(1) / (2 * width**2 * p))  # proximity weights
    g = Ridge(alpha=1.0).fit(Z - z0, preds, sample_weight=w)
# width 0.25: mean radius +0.089, mean texture -0.335, worst concave points -1.056, worst area -0.826
# width 1.00: mean radius +0.014, mean texture -0.097, worst concave points -0.277, worst area -0.252
```

**What the output teaches:**

1. **Correlated features share credit.** Radius, perimeter and area of the tumour are near-duplicates. Impurity importance spreads credit across all three (about 0.12 each). Permutation importance gives most to radius, and little to area and perimeter, because when one is shuffled its twins stand in. Neither ranking is "the truth"; group correlated features before reading importances.
2. **Noise gets impurity credit.** The pure-noise `random_id` gets a small but non-zero impurity importance (0.0025, more than two real features), because its 10,000 distinct values offer many split points to overfit. Permutation importance on test data gives it exactly 0.
3. **The PDP shows shape, not just rank:** larger tumours lower the benign probability until about 1,200, then the curve flattens.
4. **SHAP adds up; LIME depends on its settings.** The four Shapley values sum exactly to f(x) − 0.720. A high `worst concave points` value is the main reason this case sits near 0.5. LIME agrees on that feature's direction, but its slopes change about fourfold between kernel widths. For `worst area`, SHAP is positive (this row's area is small compared with the background, which pushes toward benign) while LIME's slope is negative (making the area larger would lower the probability). Both are right; they answer different questions.

## 4.4 In system design

Interpretability shows up in design rounds as a requirement to plan for, not an afterthought. Chapter 15 returns to it as an operational concern; here are the patterns to name.

- **Reason codes in regulated decisions.** A credit or insurance system must tell a declined applicant why. A common design computes TreeSHAP for each decision, maps the most negative features to approved human-readable reasons ("high debt-to-income ratio"), and stores them with the decision for audit.
- **Choose interpretability up front when it is mandatory.** If regulators or clinicians must follow the logic, a monotonic GBDT, a GAM or a scorecard may beat a black box plus post-hoc explanations. Post-hoc explanations are approximations; an interpretable model _is_ its explanation.
- **Cost and latency.** Exact SHAP is exponential in features; TreeSHAP is fast for trees, but KernelSHAP or integrated gradients on a large model can take far longer than the prediction itself. Compute explanations asynchronously or on demand rather than on the request path, unless the product shows them live.
- **Debugging and monitoring.** Run SHAP on error slices to find what the model gets wrong, and track mean |SHAP| per feature over time: a sudden shift in what the model relies on is an early warning of data drift or a broken upstream feature.
- **User-facing explanations** ("because you watched…") are often simpler than the model's real reasoning: a separate, deliberately readable layer. Say so explicitly if you propose one.
- **LLM systems.** Retrieval-augmented generation (RAG) systems explain answers by citing the retrieved sources. An LLM's written reasoning is useful but not a faithful account of its computation, so verify claims rather than treat the rationale as proof.

## 4.5 Interview Q&A

**Q1. What is the difference between global and local interpretability?** Global interpretability describes the model's overall behavior, such as which features matter most across all predictions. Local interpretability explains a single prediction, such as why this applicant was declined. Mean |SHAP| bridges them: local values averaged into a global ranking.

**Q2. Impurity importance or permutation importance?** Impurity importance is free but computed on training data, biased toward high-cardinality features, and only available for trees. Permutation importance works for any model and any metric and, computed on held-out data, reflects what the model relies on to generalize. Both mislead with correlated features, which share or hide each other's importance.

**Q3. Explain SHAP values to an engineer.** Each feature's SHAP value is its average marginal contribution to the prediction across all orders in which features could be added, from Shapley values in game theory. They add up exactly to the prediction minus the average prediction, which makes them easy to read as a waterfall from the base value to the final output.

**Q4. Why are Shapley values considered the "fair" attribution?** They are the unique attribution satisfying efficiency (they sum to the payout), symmetry (equal contributors get equal credit), dummy (non-contributors get zero) and additivity. Any other attribution method breaks at least one of these.

**Q5. How does LIME work, and what are its weaknesses?** It perturbs the input around the instance, queries the model, weights the samples by proximity and fits a simple interpretable model whose coefficients are the explanation. Weaknesses: results depend on the kernel width and the sampling, so explanations can be unstable; perturbed samples can be unrealistic; and the surrogate may not match the model even locally.

**Q6. What does a partial dependence plot show, and when does it mislead?** The average prediction as one feature varies, with the others held at their observed values. It misleads when features are correlated, because it averages over impossible combinations, and when there are interactions, because the average hides opposite effects in subgroups; check ICE curves or use ALE plots.

**Q7. Are attention weights explanations?** Not reliably. Attention shows where the model mixed information in one layer, but many layers, residual connections and value vectors also shape the output, and research has shown that very different attention patterns can give the same prediction. Use gradient-based methods such as integrated gradients, or test by ablation.

**Q8. Your model's top SHAP feature is customer ID length. What do you do?** Treat it as a likely leak or a proxy rather than real signal. Check how the feature is created, whether ID formats changed over time or differ by data source, and whether it correlates with the label through the data collection process. Remove or fix it and retrain.

**Q9. An interviewer says: "SHAP shows that income has the biggest effect, so raising incomes would reduce defaults." Respond.** SHAP explains the model, not the world. It says income moves this model's predictions the most, which may be correlation through other factors. Whether changing income changes defaults is a causal question that needs an experiment or causal inference methods.

**Q10. When would you choose an interpretable model over a black box plus SHAP?** When the stakes are high and the decision must be justified (lending, healthcare, criminal justice), when regulators require the logic itself, or when the accuracy gap is small. Post-hoc explanations approximate the model and can be wrong; an interpretable model's explanation is exact.

## 4.6 Cheat sheet and pitfalls

**Remember:**

- Axes: intrinsic vs post-hoc, global vs local, model-specific vs model-agnostic.
- Permutation importance: I_j = baseline score − shuffled score, on held-out data.
- PDP averages predictions over the data with one feature fixed; ICE shows each row; ALE handles correlated features.
- LIME: weighted local surrogate; coefficients are local slopes; sensitive to kernel width.
- SHAP: Shapley values; φ's sum to f(x) − E\[f(X)\]; TreeSHAP for trees, KernelSHAP for anything; mean |φ| for global importance.
- Integrated gradients: path integral of gradients from a baseline; attributions sum to f(x) − f(x′).
- Good explanations are faithful, stable and understandable; plausible is not enough.

**Pitfalls that cost candidates:**

- Treating feature importance or SHAP values as causal effects.
- Reading importances of correlated features one at a time.
- Computing permutation importance on training data.
- Presenting attention weights or an LLM's chain of thought as faithful explanations.
- Forgetting that SHAP values depend on the background data used as the baseline.
- Proposing explanations on the request path without considering their compute cost.

## 4.7 Further reading

<ol start="6">
  <li><a href="https://blog.ml.cmu.edu/2020/08/31/6-interpretability/">Interpretability in machine learning</a> (ML@CMU blog)</li>
</ol>

**Supplementary:** Christoph Molnar, [Interpretable Machine Learning](https://christophm.github.io/interpretable-ml-book/) (free online book); Lundberg and Lee, [A Unified Approach to Interpreting Model Predictions](https://arxiv.org/abs/1705.07874) (the SHAP paper); Ribeiro, Singh and Guestrin, ["Why Should I Trust You?"](https://arxiv.org/abs/1602.04938) (the LIME paper); Sundararajan, Taly and Yan, [Axiomatic Attribution for Deep Networks](https://arxiv.org/abs/1703.01365) (integrated gradients).
