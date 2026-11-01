---
title: "Deployment and Online Experimentation"
chapter: 14
permalink: "/blog/cracking-the-ml-interview/deployment-and-experimentation/"
updated: 2026-11-01
description: "A model that wins offline has only earned the right to be tested."
---

A model that wins offline has only earned the right to be tested. Real users, real data and real business metrics often disagree with offline metrics, and a bad deployment can cost money or trust within minutes. This chapter covers how to release models safely and how to measure, with statistics, whether a new model is actually better. It is a staple of ML system design rounds.

## 14.1 Intuition

**Offline versus online evaluation.** Offline metrics (AUC, NDCG, RMSE on held-out data) are cheap and fast, but they are computed on historical data that the old model shaped, they ignore how users react, and they rarely equal the business goal (revenue, retention, time to resolution). Online evaluation shows the model to real traffic and measures what happens. The usual sequence is: offline evaluation → shadow → canary → A/B test → full rollout, with monitoring throughout.

**Release strategies: is it safe?**

- **Shadow deployment** (reference 41) sends a copy of live requests to the new model but returns only the current model's answers to users. You check latency, error rates, crashes, and how often the two models agree, on real traffic, with zero user risk. It cannot measure user reactions, because users never see the new outputs.
- **Canary release** (reference 43) routes a small share of traffic (1 to 5%) to the new model, watches **guardrail metrics** (errors, latency, crash rate, key business metrics) and ramps up step by step (5% → 25% → 50% → 100%), rolling back automatically if a guardrail breaks. Named after canaries that warned miners of gas.
- **Blue-green deployment** runs the old (blue) and new (green) versions side by side and switches all traffic at once, with instant rollback by switching back. It is about infrastructure safety rather than measurement.
- **Feature flags** decouple deploying code from enabling it, so a model can be turned on for a group of users, or off instantly, without redeploying.

**Experiment strategies: is it better?**

- **A/B testing** (reference 42) randomly assigns users to the current model (control, A) or the new one (treatment, B) and compares a pre-chosen metric with a significance test. Randomization makes the groups comparable, so the difference is caused by the model. It is the gold standard, but it needs many users to detect small effects, and it takes days to weeks.
- **Interleaving** (reference 44), for rankers and recommenders, shows each user a single list that mixes results from both rankers, then credits each click to the ranker that supplied the item. Because every user compares both rankers at once, differences between users cancel out, and Netflix reported it needs far fewer users than A/B testing to find the better ranker. It measures preference, not business impact, so it is used to screen many candidates quickly, followed by an A/B test for the winners.
- **Multi-armed bandits** (reference 45) treat variants as slot-machine arms: allocation adapts as results arrive, shifting traffic toward variants that are doing well. They balance **exploration** (learning which arm is best) and **exploitation** (using the best arm), which lowers the cost of testing (**regret**), at the price of weaker statistical guarantees. Good for short-lived decisions such as headlines or promotions, and for many variants.

<figure class="figure-pending"><div class="figure-pending__box">diagram coming soon</div><figcaption>deployment and experimentation strategies · how each routes traffic</figcaption></figure>

| Strategy     | Users exposed to new model         | Measures                              | Typical duration       | Main risk or limit                                 |
| ------------ | ---------------------------------- | ------------------------------------- | ---------------------- | -------------------------------------------------- |
| Shadow       | None                               | Latency, errors, prediction agreement | Hours to days          | Doubles inference cost; no user feedback           |
| Canary       | 1 to 5%, ramping up                | Safety and guardrail metrics          | Hours to days per step | Small samples; not a full experiment               |
| Blue-green   | All, after the switch              | Infrastructure health                 | Minutes                | All users hit at once if something slipped through |
| A/B test     | A random share (often 50%)         | Causal effect on business metrics     | 1 to 4 weeks           | Needs large samples; slow                          |
| Interleaving | All users in the test, both models | Relative preference between rankers   | Days                   | Rankers only; not a business metric                |
| Bandit       | Adaptive                           | Cumulative reward while learning      | Ongoing                | Biased estimates; harder to analyze                |

**What makes online experiments go wrong:**

- **Peeking:** checking results daily and stopping as soon as p < 0.05 inflates false positives several-fold (14.3 shows 4.5% become 22.9%).
- **Novelty and primacy effects:** users react to change itself; early results may not last.
- **Interference (network effects):** in marketplaces and social products, treated users affect control users, so per-user randomization is biased; randomize by region, time slot or cluster instead.
- **Sample ratio mismatch (SRM):** if a 50/50 split arrives as 50.8/49.2 with millions of users, assignment or logging is broken; check it before reading any metric.
- **Multiple comparisons:** testing 20 metrics at 5% significance finds about one false win by chance.
- **Wrong unit or metric:** randomize by user, not request, when users see many requests; pick one primary metric in advance and track **guardrail metrics** that must not get worse.

## 14.2 Core math

**Hypothesis testing for an A/B test.** The **null hypothesis** H₀ says the treatment has no effect. The **p-value** is the probability of a result at least as extreme as the one observed if H₀ were true; rejecting H₀ when p < α (usually 0.05) caps the **false positive rate** (type I error) at α. **Power** (1 − β, usually 0.8) is the probability of detecting a real effect of a given size; missing one is a type II error.

**Comparing two conversion rates.** With conversions x_A of n_A users in control and x_B of n_B in treatment, rates p̂_A and p̂_B, and pooled rate p̂:

$$
z = \frac{\hat p_B - \hat p_A}{\sqrt{\hat p\,(1 - \hat p)\left(\frac{1}{n_A} + \frac{1}{n_B}\right)}}, \qquad \hat p = \frac{x_A + x_B}{n_A + n_B}
$$

The 95% confidence interval for the difference is (p̂_B − p̂_A) ± 1.96 √(p̂_A(1 − p̂_A)/n_A + p̂_B(1 − p̂_B)/n_B). For means (revenue, time spent) use a t-test or Welch's t-test; for heavy-tailed metrics, consider trimming, a log transform, or bootstrap intervals.

**Sample size.** To detect an absolute difference δ = p₁ − p₀ with significance α (two-sided) and power 1 − β, each arm needs approximately:

$$
n \approx \frac{\Big(z_{1-\alpha/2}\sqrt{2\bar p(1 - \bar p)} + z_{1-\beta}\sqrt{p_0(1 - p_0) + p_1(1 - p_1)}\Big)^2}{\delta^2}, \qquad \bar p = \frac{p_0 + p_1}{2}
$$

A handy rule of thumb is n ≈ 16σ²/δ² per arm for α = 0.05 and 80% power. The key fact: required sample size grows with 1/δ², so halving the detectable effect quadruples the users needed.

| Baseline rate | Relative lift to detect | Absolute lift | Users per arm (α = 0.05, power 0.8) |
| ------------- | ----------------------- | ------------- | ----------------------------------- |
| 5%            | 10%                     | 0.5 points    | 31,234                              |
| 5%            | 5%                      | 0.25 points   | 122,124                             |
| 5%            | 2%                      | 0.1 points    | 752,703                             |
| 20%           | 5%                      | 1 point       | 25,583                              |

**Variance reduction with CUPED (Controlled-experiment Using Pre-Experiment Data).** Adjust each user's metric Y with a pre-experiment covariate X (often the same metric before the test), using θ = Cov(X, Y)/Var(X):

$$
Y_{\text{cuped}} = Y - \theta\,(X - \bar X), \qquad \operatorname{Var}(Y_{\text{cuped}}) = \operatorname{Var}(Y)\,(1 - \rho^2)
$$

With a correlation ρ = 0.7 between X and Y, variance falls by 49%, so the same test needs about half the users. Because X is measured before assignment, the adjustment does not bias the treatment effect.

**Why peeking breaks the test.** A p-value assumes one look at a pre-set sample size. Checking repeatedly and stopping at the first p < 0.05 gives the noise many chances to cross the line. Fixes: commit to a sample size in advance, or use methods built for continuous monitoring: **sequential tests** with adjusted boundaries (O'Brien–Fleming, alpha spending) or always-valid p-values.

**Multiple comparisons.** Testing m independent metrics at level α gives a chance 1 − (1 − α)^m of at least one false positive: 64% for m = 20. The **Bonferroni** correction tests each at α/m; the **Benjamini–Hochberg** procedure controls the false discovery rate with more power.

**Interleaving.** In **team-draft interleaving**, the two rankers take turns, in random order each round, adding their highest-ranked item not yet in the list, and each item remembers which team added it. For each query, the ranker whose items got more clicks wins. Under H₀ (equal rankers) each is equally likely to win, so a **binomial (sign) test** on wins versus losses decides significance. Because each user compares both rankers on the same query, user-to-user variation (some users click ten times as much as others) cancels out.

**Multi-armed bandits.** With K arms, true mean rewards μ_k and best mean μ\*, the **regret** after T rounds is the reward lost by not always pulling the best arm, where k_t is the arm pulled at round t:

$$
R_T = T\,\mu^{*} - \sum_{t=1}^{T} \mu_{k_t}
$$

An even A/B split has regret growing linearly in T; good bandit algorithms grow only logarithmically.

- **ε-greedy:** with probability ε pick a random arm (explore), otherwise the arm with the best observed mean (exploit). Simple, but it keeps exploring at rate ε forever unless ε decays.
- **UCB1 (Upper Confidence Bound)**, with n_k pulls of arm k so far and observed mean μ̂_k, picks the arm with the highest optimistic estimate, so uncertain arms get tried:

$$
k_t = \arg\max_k \Big(\hat\mu_k + \sqrt{\frac{2 \ln t}{n_k}}\Big)
$$

- **Thompson sampling** keeps a posterior for each arm's reward; for click/no-click rewards, a Beta(1 + successes, 1 + failures) distribution. Each round it draws one sample from every posterior and pulls the arm with the highest sample, so each arm is chosen with the probability that it is the best. It is simple, robust and usually the strongest in practice.
- **Contextual bandits** (LinUCB, neural bandits) choose arms based on user and item features, which makes them a lightweight form of personalization.

## 14.3 Code

**Sample size and a two-proportion z-test:**

```python
import numpy as np
from scipy import stats

def sample_size_per_arm(p0, mde_rel, alpha=0.05, power=0.8):
    p1 = p0 * (1 + mde_rel)
    z_a, z_b = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    pbar = (p0 + p1) / 2
    num = z_a * np.sqrt(2 * pbar * (1 - pbar)) + z_b * np.sqrt(p0 * (1 - p0) + p1 * (1 - p1))
    return int(np.ceil(num ** 2 / (p1 - p0) ** 2))

print(sample_size_per_arm(0.05, 0.10))          # 31,234 users per arm

def two_prop_ztest(x_a, n_a, x_b, n_b):
    p_a, p_b = x_a / n_a, x_b / n_b
    p = (x_a + x_b) / (n_a + n_b)
    z = (p_b - p_a) / np.sqrt(p * (1 - p) * (1 / n_a + 1 / n_b))
    return p_b - p_a, z, 2 * (1 - stats.norm.cdf(abs(z)))

print(two_prop_ztest(1500, 30_000, 1650, 30_000))
# lift 0.5 points (5.0% -> 5.5%), z = 2.75, p = 0.006; 95% CI for the lift: 0.14 to 0.86 points
```

**What peeking does: 2,000 simulated A/A tests** (both arms identical, 5% conversion, 1,000 users per arm per day, 20 days), tested daily:

```python
rng = np.random.default_rng(0)
n_sims, days, per_day, p = 2000, 20, 1000, 0.05
fp_peek = fp_end = 0
for _ in range(n_sims):
    a = rng.binomial(1, p, size=(days, per_day)).sum(1).cumsum()
    b = rng.binomial(1, p, size=(days, per_day)).sum(1).cumsum()
    n = per_day * np.arange(1, days + 1)
    pooled = (a + b) / (2 * n)
    z = (b / n - a / n) / np.sqrt(pooled * (1 - pooled) * 2 / n)
    pvals = 2 * (1 - stats.norm.cdf(np.abs(z)))
    fp_peek += (pvals < 0.05).any()                # stop at the first "significant" day
    fp_end += pvals[-1] < 0.05                     # one test at the planned end
print(fp_peek / n_sims, fp_end / n_sims)           # 0.229 vs 0.045
```

There is no real difference between the arms, yet stopping at the first significant day declares a winner in 22.9% of tests, almost five times the promised 5%. A single test at the planned end stays at 4.5%.

**Interleaving versus A/B testing.** Two rankers order 50 candidate items by noisy estimates of true relevance; ranker A's noise is slightly lower, so it is slightly better. Users click with probability (position bias) × (relevance) × (their own activity level), and activity varies a lot between users (log-normal). For each number of queries, 100 repeated experiments measure how often each method correctly finds A better at p < 0.05:

```python
def team_draft(rng, a, b, k=10):
    """Each round, a coin flip picks which ranker drafts first; each adds its best unused item."""
    out, team, used, ia, ib = [], [], set(), 0, 0
    while len(out) < k:
        for t in (["A", "B"] if rng.random() < 0.5 else ["B", "A"]):
            if len(out) == k:
                break
            src, i = (a, ia) if t == "A" else (b, ib)
            while src[i] in used:
                i += 1
            out.append(src[i]); team.append(t); used.add(src[i])
            if t == "A": ia = i + 1
            else: ib = i + 1
    return np.array(out), np.array(team)
# Per query: credit clicks to the team that supplied each item; the team with more clicks wins.
# Significance: binomial test of A's wins against B's wins (ties dropped).
```

| Queries | A/B test power | Interleaving power |
| ------- | -------------- | ------------------ |
| 1,000   | 7%             | 13%                |
| 2,000   | 12%            | 33%                |
| 4,000   | 19%            | 42%                |
| 8,000   | 26%            | 84%                |
| 16,000  | 37%            | 95%                |

Interleaving reaches 84% power with 8,000 queries; the A/B test has only 37% power even with twice as many. In the A/B test, the difference between rankers is buried under the difference between heavy and light users; interleaving compares both rankers on the same user, so that variation cancels.

**Bandits versus an even split.** Three variants with true conversion rates of 4%, 5% and 6%, 20,000 visitors, averaged over 50 runs:

```python
true_ctr = np.array([0.04, 0.05, 0.06])

def simulate(policy, seed, T=20_000):
    r = np.random.default_rng(seed)
    K = len(true_ctr)
    counts, wins = np.zeros(K), np.zeros(K)
    a, b = np.ones(K), np.ones(K)                          # Beta(1, 1) priors for Thompson sampling
    regret, total = np.zeros(T), 0.0
    for t in range(T):
        if policy == "even split":
            k = t % K
        elif policy == "epsilon-greedy":
            explore = r.random() < 0.1 or counts.min() == 0
            k = r.integers(K) if explore else np.argmax(wins / np.maximum(counts, 1))
        elif policy == "UCB1":
            k = t if t < K else np.argmax(wins / counts + np.sqrt(2 * np.log(t + 1) / counts))
        else:                                              # Thompson sampling
            k = np.argmax(r.beta(a, b))
        reward = r.random() < true_ctr[k]
        counts[k] += 1; wins[k] += reward; a[k] += reward; b[k] += 1 - reward
        total += true_ctr.max() - true_ctr[k]
        regret[t] = total
    return regret, counts
```

<figure class="figure-pending"><div class="figure-pending__box">diagram coming soon</div><figcaption>bandit simulation in 14.3 · 20,000 visitors, 50 runs per policy</figcaption></figure>

| Policy                  | Share of traffic on the best variant (6%) |
| ----------------------- | ----------------------------------------- |
| Even split (A/B/C test) | 33%                                       |
| UCB1                    | 46%                                       |
| ε-greedy (ε = 0.1)      | 71%                                       |
| Thompson sampling       | 82%                                       |

Thompson sampling lost under a quarter of what the even split lost, sending 82% of traffic to the best variant by the end. UCB1's optimism made it explore a lot at these small rate differences. The even split, though, is the one that yields clean, unbiased estimates of every variant's effect, which is why A/B tests remain the standard for decisions that must be measured rigorously.

## 14.4 In system design

Every ML design round should end with how the model reaches users and how you know it worked. A complete answer covers these steps:

1. **Offline gate:** the new model must beat the current one on the offline metric and on key slices, with no regressions on a frozen golden set (Chapter 5).
2. **Shadow:** run it on mirrored traffic to compare latency, error rates, prediction distributions and agreement with the current model.
3. **Canary:** expose 1 to 5% of traffic with automatic rollback on guardrails (error rate, p99 latency, crash rate, a business-health metric).
4. **A/B test:** a pre-registered primary metric (for example revenue per user or 7-day retention), guardrail metrics, a sample size from a power calculation, and a fixed duration covering full weekly cycles (at least one to two weeks).
5. **Ramp and monitor:** roll out gradually, keep a small long-term holdout (1 to 5% of users who never get the change) to measure lasting effects, and monitor drift (Chapter 15).

Further points to raise:

- **Choosing the randomization unit.** User-level for most consumer products; session- or request-level only when users cannot notice inconsistency; cluster- or region-level for marketplaces and social networks where users interact (a ride-hailing pricing test changes drivers' behaviour for everyone).
- **Experimentation platforms** handle hashing users into variants, mutually exclusive experiment layers, metric pipelines, SRM checks and dashboards; mention one rather than describing a hand-rolled split.
- **Interleaving for rankers.** Use interleaving to screen many ranking candidates quickly, then A/B test the finalists on business metrics.
- **Bandits for short-lived or many-variant decisions:** headlines, thumbnails, promotions, cold-start content. Log the probability with which each arm was shown (the **propensity**) so offline analysis can correct for adaptive allocation.
- **Logging for counterfactual evaluation.** Logging propensities in production also enables off-policy evaluation of future models with importance sampling (Chapter 5), reducing how many ideas need a live test.
- **Business metric versus model metric.** A model with better AUC can lose an A/B test (it may change which items users see in ways offline data cannot capture). Treat the A/B result as the decision, and investigate the gap.

## 14.5 Interview Q&A

**Q1. What is the difference between shadow deployment and a canary release?** In shadow mode the new model receives a copy of live traffic but its outputs are never shown, so you test performance and correctness with zero user risk but learn nothing about user reactions. A canary serves the new model's outputs to a small share of real users, testing safety under real conditions with limited blast radius before ramping up.

**Q2. How do you decide the sample size and duration of an A/B test?** Choose the primary metric, its baseline rate or variance, the minimum effect worth detecting, the significance level (0.05) and power (0.8), then compute the users needed per arm (about 16σ²/δ²). Divide by eligible daily traffic for the duration, and round up to whole weeks to cover weekday and weekend behaviour.

**Q3. Your A/B test shows p = 0.04 on day 3. Do you ship?** Not yet. Stopping early at the first significant result inflates false positives (a simulation of daily peeking gave 22.9% instead of 5%). Run to the planned sample size, or use a sequential test designed for interim looks; also check the sample ratio, novelty effects and guardrail metrics.

**Q4. What is a sample ratio mismatch, and why does it matter?** The observed split between arms differs significantly from the planned one (for example 50.8/49.2 with millions of users, which a chi-squared test flags). It signals broken assignment, logging, bot filtering or crashes that affect one arm, which can bias every metric. Fix the cause before trusting results.

**Q5. How does interleaving work, and why is it more sensitive than A/B testing?** Each user sees one list combining results from both rankers (in team-draft, the rankers take turns picking items), and clicks are credited to the ranker that supplied each item. Every user compares both rankers on the same query, so variation between users, often the largest source of noise, cancels out; Netflix found it needs far fewer users. It measures preference, not business metrics, so finalists still get an A/B test.

**Q6. When would you use a multi-armed bandit instead of an A/B test?** When the cost of showing a worse variant matters more than a precise effect estimate: short campaigns, many variants, cold-start recommendations or continuously changing content. A/B tests remain better for decisions that need unbiased, statistically rigorous effect sizes on long-term metrics.

**Q7. Explain Thompson sampling.** Keep a posterior distribution over each arm's reward rate (Beta for clicks). Each round, sample one value from each posterior and pull the arm with the highest sample; update that arm's posterior with the result. Arms are chosen in proportion to the probability they are best, so exploration fades naturally as evidence accumulates.

**Q8. What are guardrail metrics?** Metrics that must not get worse even if the primary metric improves: latency, error and crash rates, unsubscribes, complaint rates, revenue for a test aimed at engagement. A treatment that wins on the primary metric but breaks a guardrail is not shipped.

**Q9. Your new recommender wins offline by 3% NDCG but loses the A/B test. Why might that be?** The offline data was generated by the old model, so it cannot reward items users never saw (selection bias); the offline metric may not match the business metric; the new model may cause less diversity, novelty fatigue or popularity bias; or there may be training-serving skew or latency differences. Investigate with logged propensities, segment analysis and a check of serving features.

**Q10. How would you test a new pricing model in a two-sided marketplace?** User-level randomization is biased because riders and drivers in both arms share supply. Use switchback experiments (alternate the whole market between treatment and control over time slots) or geographic cluster randomization, accept wider confidence intervals, and watch both sides' metrics.

## 14.6 Cheat sheet and pitfalls

**Remember:**

- Release safety: shadow (no exposure) → canary (small, ramp) → full rollout; blue-green for instant switch and rollback.
- Measurement: A/B test for causal business impact; interleaving for sensitive ranker comparison; bandits to minimize regret.
- Two-proportion z-test; sample size ≈ 16σ²/δ² per arm; halving δ quadruples n.
- Don't peek without a sequential design; check SRM; correct for multiple metrics.
- CUPED: Y − θ(X − X̄) cuts variance by a factor of 1 − ρ².
- UCB1: μ̂ + √(2 ln t / n). Thompson: sample from Beta posteriors, pick the max.
- Randomize by the unit users experience; use clusters or switchbacks when users interact.

**Pitfalls that cost candidates:**

- Jumping from offline metrics straight to 100% rollout.
- Stopping an A/B test the first day p < 0.05.
- Randomizing by request when the same user sees both variants.
- Declaring victory on one of many metrics without correction.
- Using bandit-allocated data as if it were a clean A/B comparison.
- Forgetting guardrails, novelty effects and long-term holdouts.

## 14.7 Further reading

<ol start="41">
  <li><a href="https://christophergs.com/machine%20learning/2019/03/30/deploying-machine-learning-applications-in-shadow-mode/">Deploying machine learning models in shadow mode</a> (Christopher Samiullah)</li>
  <li><a href="https://en.wikipedia.org/wiki/A/B_testing">A/B testing</a> (Wikipedia)</li>
  <li><a href="https://blog.getambassador.io/cloud-native-patterns-canary-release-1cb8f82d371a">Cloud-native patterns: canary release</a> (Ambassador Labs)</li>
  <li><a href="https://netflixtechblog.com/interleaving-in-online-experiments-at-netflix-a04ee392ec55">Innovating faster on personalization algorithms at Netflix using interleaving</a> (Netflix Technology Blog)</li>
  <li><a href="https://vwo.com/blog/multi-armed-bandit-algorithm/">Multi-armed bandit algorithm</a> (VWO)</li>
</ol>

**Supplementary:** Kohavi, Tang and Xu, _Trustworthy Online Controlled Experiments_ (Cambridge University Press, 2020), the standard book on A/B testing in practice.
