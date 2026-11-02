---
title: "ML Infrastructure and Interpretability in Practice"
chapter: 15
permalink: "/blog/cracking-the-ml-interview/ml-infrastructure/"
updated: 2026-11-01
description: "The infrastructure loop around a model: data, feature stores, training pipelines, registries, serving, drift monitoring and interpretability in production."
---

Training a good model is perhaps a fifth of the work of an ML system; the rest is the infrastructure that feeds it data, trains it reproducibly, serves it reliably, watches it in production and retrains it when the world changes. This closing chapter ties the book together into that loop, and shows how interpretability (Chapter 4) becomes an operational tool. In design rounds, the strongest candidates are the ones who talk about this loop without being asked.

## 15.1 Intuition

**ML infrastructure** (reference 46) is the set of tools and systems that take a model from idea to production and keep it healthy. Its components, roughly in the order data flows:

- **Data layer:** storage (warehouse, lake; Chapter 1), pipelines that ingest and transform data in batch or streaming, **data validation** that catches schema changes, missing values and out-of-range values before they reach a model, and **data versioning** so any model can be traced to the exact data it saw.
- **Feature store:** computes features once and serves the same definitions to training (offline store, point-in-time correct history) and to serving (online store, low-latency lookups). Its main job is preventing **training-serving skew**: features computed one way in training and another way in production.
- **Training infrastructure:** compute (GPU clusters scheduled by Kubernetes or a cluster scheduler, Chapter 7), **pipeline orchestration** (Airflow, Kubeflow Pipelines, Argo, Dagster) that runs data preparation, training and evaluation as reproducible steps, and **experiment tracking** (MLflow, Weights & Biases) that records code version, data version, hyperparameters, metrics and artifacts for every run.
- **Model registry:** a versioned catalogue of trained models with their lineage, evaluation results and approval status (staging, production, archived). Deployment pulls from the registry, never from a laptop.
- **Serving:** **batch** prediction (score everything nightly, store results), **online** serving behind an API with latency targets, or **streaming** (score events as they arrive). Plus optimized runtimes, autoscaling, caching and compression (Chapter 13).
- **Monitoring:** system health (latency, errors, throughput, cost), **data drift** (input distributions changing), **prediction drift**, and model quality once labels arrive. Alerts trigger investigation or retraining.
- **CI/CD for ML:** automated tests for data, features and models, not just code, and automated, staged deployment (Chapter 14).

<figure class="figure-pending"><div class="figure-pending__box">diagram coming soon</div><figcaption>components of an ML platform · data to monitoring and back</figcaption></figure>

**MLOps maturity.** A widely used framing from Google describes three levels:

| Level                          | How models are built and shipped                                                                    | Typical symptoms                                                                  |
| ------------------------------ | --------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| 0: manual                      | Notebooks; a data scientist hands a model file to engineers; rare, manual retraining                | Models go stale; training and serving code diverge; results are hard to reproduce |
| 1: automated training pipeline | The pipeline itself is deployed; retraining runs on a schedule or trigger with automatic validation | Faster, reliable retraining; pipeline changes are still manual                    |
| 2: CI/CD for pipelines         | Changes to data, features and model code are tested, built and deployed automatically               | Many models and frequent experiments managed safely                               |

Most teams do not need level 2 for every model. Match the investment to how often the model must change and how costly its failures are.

**Why models decay in production.** The world changes after training:

- **Data drift (covariate shift):** the distribution of inputs P(x) changes, such as a new user segment, a new device or a seasonal pattern.
- **Label shift (prior shift):** the class balance P(y) changes, such as fraud rising during a holiday.
- **Concept drift:** the relationship P(y | x) itself changes, such as new fraud tactics or changing user tastes. It is the most damaging and the hardest to detect, because inputs can look exactly the same (15.3 shows accuracy falling from 99.95% to 49.9% with no input drift at all).
- **Upstream breakage:** a pipeline change, unit change, renamed category or broken logger silently corrupts a feature. This is the most common cause of sudden failures, and data validation catches most of it.
- **Feedback loops:** the model's own decisions change future training data (Chapter 5).

**Interpretability in practice.** Chapter 4 covered the methods; in production, interpretability becomes infrastructure (reference 47 covers the tooling side of MLOps):

- **Debugging and incident response:** when a metric drops, per-feature attributions over time (mean |SHAP| per feature, by day) show which inputs the model started relying on differently, often pointing straight at a broken upstream feature.
- **Explanation services:** regulated products compute reason codes alongside each prediction, store them with the decision, and serve them to customer-support teams and auditors.
- **Documentation:** **model cards** describe a model's intended use, training data, evaluation by subgroup and known limitations; **datasheets** do the same for datasets. Both are increasingly expected by regulators and enterprise customers.
- **Fairness and slice monitoring:** track performance and outcomes by relevant groups and segments in production, not just at launch.
- **Testing behaviour:** behavioural tests (invariance to irrelevant changes, such as a name in a resume; expected direction of effects, such as higher income never lowering a credit score) run in CI like unit tests.

## 15.2 Core math

**Population Stability Index (PSI).** Bin a feature using the reference (training) distribution, usually into 10 quantile bins, and compare the share of data in each bin: e_i in the reference and a_i in current traffic:

$$
\text{PSI} = \sum_{i=1}^{B} (a_i - e_i)\,\ln\frac{a_i}{e_i}
$$

PSI is a symmetrized KL divergence over the bins. The common rule of thumb: below 0.1 is stable, 0.1 to 0.25 is a moderate shift worth watching, above 0.25 is a significant shift. Because it measures the size of the shift rather than its statistical significance, it does not explode just because traffic is large; with small samples, though, it is noisy.

**Kolmogorov–Smirnov (KS) test.** For a continuous feature with empirical cumulative distribution functions F_ref and F_cur, the statistic is the largest vertical gap between them:

$$
D = \sup_x \big|F_{\text{ref}}(x) - F_{\text{cur}}(x)\big|
$$

The test's p-value shrinks with sample size, so with millions of events almost every feature "drifts significantly" (15.3 shows a harmless 0.1-standard-deviation shift with p = 10⁻⁴⁰). Use D itself, or PSI, as the effect size, and alert on thresholds tuned to what actually hurts the model.

**Categorical features** use the **chi-squared test** on the table of category counts, or PSI over categories. **Jensen–Shannon divergence**, JS(P, Q) = ½KL(P‖M) + ½KL(Q‖M) with M = ½(P + Q), is a bounded, symmetric alternative often used for prediction distributions and embeddings.

**Detecting concept drift needs labels or proxies.** Input statistics cannot reveal a change in P(y | x). Options, from best to weakest:

- monitor model quality on fresh labeled data as labels arrive (with known label delay);
- label a small random sample of production traffic regularly;
- track proxy outcomes (clicks, complaints, manual overrides by human reviewers);
- monitor prediction distributions and confidence, which catch some but not all changes.

**Label shift correction.** If only the class prior changes (P(y) shifts but P(x | y) stays the same), a calibrated classifier's probabilities can be corrected by Bayes' rule, using new priors π′\_k and training priors π_k (the same prior correction as Chapter 6):

$$
p'(y = k \mid x) \propto \frac{\pi'_k}{\pi_k}\; p(y = k \mid x)
$$

**Serving capacity: Little's law.** The average number of requests in flight equals arrival rate × average time in the system. At 2,000 requests per second and 50 ms per request, 2,000 × 0.05 = 100 requests are being processed at any moment, so you need capacity for at least 100 concurrent requests, plus headroom for peaks:

$$
L = \lambda\, W
$$

**Latency is reported as percentiles**, not averages: p50 for the typical user, p99 for the slowest 1% of requests, which matters most when one page makes many model calls (with 20 calls per page, a page is slow whenever any one call is slow, so 1 − 0.99²⁰ ≈ 18% of pages see at least one p99-slow call).

## 15.3 Code

**PSI and KS on synthetic drift.** A reference sample of 50,000 values from a standard normal distribution, compared with 50,000 current values under different shifts:

```python
import numpy as np
from scipy import stats

def psi(expected, actual, bins=10, eps=1e-6):
    """Population Stability Index, with bins from the reference distribution's quantiles."""
    edges = np.quantile(expected, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, eps, None), np.clip(a, eps, None)
    return float(np.sum((a - e) * np.log(a / e)))

rng = np.random.default_rng(0)
ref = rng.normal(0, 1, 50_000)                         # training-time distribution
cases = {"no drift": rng.normal(0, 1, 50_000),
         "mean +0.1 sd": rng.normal(0.1, 1, 50_000),
         "mean +0.5 sd": rng.normal(0.5, 1, 50_000),
         "std x1.5": rng.normal(0, 1.5, 50_000),
         "mean +1 sd": rng.normal(1.0, 1, 50_000)}
for name, cur in cases.items():
    ks = stats.ks_2samp(ref, cur)
    print(name, psi(ref, cur), ks.statistic, ks.pvalue)
```

| Current traffic               | PSI   | KS statistic D | KS p-value   | Verdict by PSI rule of thumb   |
| ----------------------------- | ----- | -------------- | ------------ | ------------------------------ |
| No drift                      | 0.000 | 0.005          | 0.56         | Stable                         |
| Mean +0.1 standard deviations | 0.010 | 0.043          | 1.6 × 10⁻⁴⁰  | Stable                         |
| Mean +0.5 standard deviations | 0.241 | 0.197          | below 10⁻³⁰⁰ | Moderate, close to significant |
| Standard deviation × 1.5      | 0.200 | 0.099          | 2.2 × 10⁻²¹⁵ | Moderate                       |
| Mean +1 standard deviation    | 0.919 | 0.383          | below 10⁻³⁰⁰ | Significant                    |

With 50,000 samples, the KS test calls even the tiny 0.1-standard-deviation shift overwhelmingly significant, while PSI rates it negligible. Repeating a harmless 0.02-standard-deviation shift at growing sample sizes makes the point sharper: the KS p-value goes from 0.20 at 500 samples to 0.014 at 50,000 and 10⁻¹³ at 500,000, while PSI shrinks toward zero as noise averages out (0.069, 0.002, 0.0006, 0.0004 for 500, 5,000, 50,000 and 500,000). PSI's value at 500 samples also shows that small windows give noisy PSI readings.

**Concept drift that input monitoring cannot see, and input drift that does no harm:**

```python
from sklearn.linear_model import LogisticRegression

n = 20_000
X_old = rng.normal(size=(n, 2))
y_old = (X_old[:, 0] + X_old[:, 1] > 0).astype(int)
model = LogisticRegression().fit(X_old, y_old)

# Concept drift: same input distribution, new relationship
X_new = rng.normal(size=(n, 2))
y_new = (X_new[:, 0] - X_new[:, 1] > 0).astype(int)
print(model.score(X_old, y_old), model.score(X_new, y_new))      # 0.9995 -> 0.4994
print([psi(X_old[:, j], X_new[:, j]) for j in range(2)])          # [0.0005, 0.0004]
print(psi(model.predict_proba(X_old)[:, 1], model.predict_proba(X_new)[:, 1]))  # 0.0007

# Covariate shift: inputs move a lot, the relationship is unchanged
X_shift = rng.normal(loc=[1.0, 0.0], size=(n, 2))
y_shift = (X_shift[:, 0] + X_shift[:, 1] > 0).astype(int)
print(psi(X_old[:, 0], X_shift[:, 0]), model.score(X_shift, y_shift))  # PSI 0.929, accuracy 0.9996
```

The two halves of this experiment are the most important monitoring lesson in the book. When the rule linking inputs to labels changed, accuracy collapsed from 99.95% to 49.9% (a coin flip), yet every input PSI stayed below 0.001 and even the prediction distribution looked unchanged (PSI 0.0007): only labels reveal concept drift. When the inputs shifted dramatically (PSI 0.93, prediction PSI 0.47) but the relationship held, accuracy stayed at 99.96%: drift alerts flag change, not necessarily harm. Monitor inputs to catch broken pipelines early, and monitor outcomes to know whether the model still works.

**Categorical drift with a chi-squared test.** If traffic moves from 50/30/20% web/iOS/Android to 40/30/30%, 10,000 events per window give χ² = 287 and p ≈ 6 × 10⁻⁶³:

```python
cats = ["web", "ios", "android"]
ref_cat = rng.choice(cats, 10_000, p=[0.5, 0.3, 0.2])
cur_cat = rng.choice(cats, 10_000, p=[0.4, 0.3, 0.3])
table = np.array([[np.sum(ref_cat == c) for c in cats], [np.sum(cur_cat == c) for c in cats]])
chi2, p, _, _ = stats.chi2_contingency(table)
```

## 15.4 In system design

This section doubles as a template for the last ten minutes of any ML design round: after the model, walk through the platform around it.

- **Training pipeline:** scheduled or triggered; versioned inputs (data snapshot, feature definitions, code commit); automated evaluation against the current production model and a golden set; registration only if it passes.
- **Serving choice:** batch when predictions can be precomputed (daily recommendations, churn scores), online when they depend on fresh context (search ranking, fraud at checkout), streaming when events must be scored as they arrive. State the latency budget and estimate the load with Little's law.
- **Feature consistency:** one feature definition shared by training and serving through a feature store, plus logging of the exact features used at prediction time, so training data can be built from what the model actually saw.
- **Monitoring plan:**
  - **System:** latency p50/p99, errors, throughput and cost per 1,000 predictions.
  - **Data:** schema checks, missing-value rates, and PSI on key features against the training window.
  - **Predictions:** score distribution and the share of predictions above the decision threshold.
  - **Quality:** metrics on delayed labels and on regularly labeled samples, sliced by segment.
- **Alerting and response:** thresholds tuned to business impact, runbooks for on-call engineers, automatic rollback to the previous model version, and a fallback (rules or a simpler model) if the model service is down.
- **Retraining strategy:** scheduled (weekly) for slowly changing domains, triggered by drift or performance decay for faster ones, or continual/online learning for very fast ones such as ads; always validated before promotion.
- **Governance and interpretability:** model cards, lineage from prediction back to model version and training data, reason codes where decisions affect people, access control for sensitive data, and audits by segment.
- **Build versus buy:** managed platforms (SageMaker, Vertex AI, Azure ML, Databricks) and open-source stacks (MLflow, Kubeflow, Feast, Ray) trade control for speed; choose by team size, scale and compliance needs, and say why.

## 15.5 Interview Q&A

**Q1. What is training-serving skew, and how do you prevent it?** A difference between how features or data look in training and in production: different code paths, different data sources, time-travel leaks in training, or stale online features. Prevent it with shared feature definitions (a feature store), logging the features actually used at prediction time and training on those logs, point-in-time correct joins, and comparing online and offline feature distributions.

**Q2. Data drift versus concept drift: how do you detect each?** Data drift is a change in the input distribution P(x); detect it with PSI, KS or chi-squared tests on features and prediction distributions. Concept drift is a change in P(y | x); input statistics cannot see it, so you need labels (delayed ground truth, regular labeled samples) or outcome proxies. In the experiment above, concept drift halved accuracy while every input PSI stayed below 0.001.

**Q3. What does a feature store do?** It stores feature definitions and their computed values, serving an offline store of point-in-time correct history for training and an online low-latency store for serving, from the same definitions. That removes duplicated feature code, prevents training-serving skew and lets teams reuse features.

**Q4. What is a model registry, and why use one?** A versioned store of trained models with metadata: training data and code versions, metrics, owner and stage (staging, production, archived). Deployments pull a specific registered version, which makes releases reproducible, approvals auditable and rollback a one-step operation.

**Q5. How would you decide when to retrain a model?** Based on how fast the domain changes and the cost of staleness: schedule retraining when change is steady (weekly or daily), and add triggers on monitored signals (drift beyond a threshold, quality decay on fresh labels). Always validate a retrained model against the current one before promoting it, since retraining on broken data makes things worse.

**Q6. Batch, online or streaming inference: how do you choose?** Batch when predictions can be computed ahead of time for a known set of entities: cheapest, simplest, tolerant of slow models. Online when predictions depend on request-time context and must be returned within a latency budget. Streaming when events must be scored continuously as they arrive, as in fraud or anomaly detection. Many systems mix them: precompute candidates in batch, rerank online.

**Q7. What would you monitor for a deployed fraud model?** System metrics (latency, errors, throughput), input health (schema, missing values, PSI on key features), prediction distribution and alert rate, delayed-label metrics (precision, recall, chargebacks caught) by merchant and region, analyst override rates as a fast proxy, and cost. Each with thresholds, owners and runbooks.

**Q8. Why is a KS test p-value a poor drift alarm at scale?** With millions of samples, the test detects tiny, harmless shifts as overwhelmingly significant, so every feature alarms constantly. Use an effect-size measure such as PSI or the KS statistic itself, with thresholds tuned to changes that actually hurt model quality.

**Q9. How does interpretability help in production, beyond explaining single predictions?** Tracking feature attributions over time localizes the cause of a metric drop (often a broken upstream feature); reason codes satisfy regulatory and customer-support needs; model cards document intended use and limitations; slice and fairness monitoring catches harms to specific groups; behavioural tests in CI stop regressions in expected model behaviour.

**Q10. Describe MLOps maturity levels. Where would you start for a new team?** Level 0 is manual (notebooks, hand-offs, rare retraining); level 1 automates the training pipeline with continuous retraining and validation; level 2 adds CI/CD for the pipelines themselves. A new team should start by making one model reproducible and monitored (versioned data and code, a simple scheduled pipeline, a registry, basic monitoring), then automate further as the number of models and the rate of change grow.

## 15.6 Cheat sheet and pitfalls

**Remember:**

- The loop: data → pipelines and validation → feature store → training and tracking → registry → serving → monitoring → retraining.
- Prevent training-serving skew with shared feature definitions and logged serving features.
- Drift types: data (P(x)), label (P(y)), concept (P(y | x)), plus upstream breakage, the most common cause of sudden failures.
- PSI = Σ(a − e) ln(a/e); under 0.1 stable, 0.1 to 0.25 moderate, over 0.25 significant.
- KS p-values flag trivial shifts at scale; use effect sizes.
- Concept drift needs labels or outcome proxies.
- Little's law: concurrency = request rate × latency; report p50 and p99 latency.
- MLOps maturity: manual → automated training → CI/CD for pipelines.
- Interpretability in production: attribution tracking, reason codes, model cards, slice monitoring, behavioural tests.

**Pitfalls that cost candidates:**

- Ending the design at "train the model and deploy it".
- Monitoring only system metrics, or only input drift.
- Alerting on every statistically significant shift.
- Retraining automatically without validating against the current model.
- Ignoring label delay when planning quality monitoring.
- Proposing a heavyweight platform for a single, slowly changing model.

## 15.7 Further reading

<ol start="46">
  <li><a href="https://www.run.ai/guides/machine-learning-engineering/machine-learning-infrastructure">Machine learning infrastructure</a> (Run:ai guide)</li>
  <li><a href="https://fullstackdeeplearning.com/spring2021/lecture-6/">Lecture 6: MLOps Infrastructure &amp; Tooling</a> (Full Stack Deep Learning, Spring 2021). The source list labels this "Interpretability in ML", but the linked lecture covers infrastructure and tooling; the same course covers explainability in <a href="https://fullstackdeeplearning.com/spring2021/lecture-10/">Lecture 10: Testing &amp; Explainability</a>.</li>
</ol>

**Supplementary:** Google Cloud, [MLOps: Continuous delivery and automation pipelines in machine learning](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning) (the maturity levels); Sculley et al., [Hidden Technical Debt in Machine Learning Systems](https://papers.nips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html) (NeurIPS 2015); Mitchell et al., [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993).
