---
title: "Ensemble Learning"
chapter: 3
permalink: "/blog/cracking-the-ml-interview/ensemble-learning/"
updated: 2026-11-01
description: "An ensemble combines several models so that their errors cancel out."
---

An ensemble combines several models so that their errors cancel out. Bagging cuts **variance** by averaging models trained in parallel, boosting cuts **bias** by training models in sequence on what earlier ones got wrong, and stacking learns how best to combine different models. **Gradient-boosted decision trees (GBDTs)**, ensembles of small trees added one after another, are still the default for tabular data, so expect this chapter in almost every ML interview.

## 3.1 Intuition

**Why combining helps.** Ask one doctor and you get one opinion, with that doctor's blind spots. Ask five doctors who trained in different places and take the majority, and you are less likely to be wrong, _provided their mistakes differ_. If all five learned from the same textbook, the panel is no better than one. Ensembles work only when their members make **different errors**.

**Bias and variance.** A model's expected error splits into bias (systematically wrong, too simple, underfits), variance (too sensitive to the particular training sample, overfits) and irreducible noise. Each ensemble method attacks one of these:

|                     | Bagging                            | Boosting                                       | Stacking                                                                |
| ------------------- | ---------------------------------- | ---------------------------------------------- | ----------------------------------------------------------------------- |
| Main target         | Variance                           | Bias (and some variance)                       | Both, by combining different model families                             |
| Training            | Parallel, independent              | Sequential, each model depends on the last     | Base models in parallel, then a meta-model                              |
| Base learners       | Strong, high-variance (deep trees) | Weak, high-bias (shallow trees, stumps)        | Diverse (trees, linear models, SVMs, k-nearest neighbours, neural nets) |
| How members combine | Average or majority vote           | Weighted sum                                   | A learned meta-model                                                    |
| Overfitting risk    | Low                                | Higher; needs learning rate and early stopping | Moderate; needs out-of-fold predictions                                 |
| Famous examples     | Random forest                      | AdaBoost, XGBoost, LightGBM, CatBoost          | Kaggle winning solutions                                                |

**Decision trees, the building block.** Every ensemble in this chapter is built from decision trees. Section 2.5 explains how a single tree is grown, split and pruned, and why a fully grown tree has low bias but very high variance. Ensembles keep the tree's strengths and remove that variance.

**Bagging (bootstrap aggregating).** Draw B bootstrap samples (n rows sampled _with replacement_ from n rows), train one model on each, then average their predictions (regression) or take a majority vote (classification). Each model sees a slightly different dataset, so each overfits in a different way, and averaging smooths those quirks out.

**Random forest** is bagging over decision trees plus one twist: at every split, a tree may choose only from a random subset of m of the p features. This stops every tree from splitting on the same dominant feature, which **decorrelates** the trees and makes averaging far more effective. Common defaults are m ≈ √p for classification and m ≈ p/3 for regression; setting m = p turns a random forest back into plain bagging. A small m helps most when many features are correlated with each other.

The trees in a forest are grown deep and **not pruned**. Each has low bias and high variance, and averaging removes the variance.

**Variable importance.** A forest of hundreds of trees cannot be read like a single tree, so it is summarized by **variable importance**: for each feature, add up how much its splits reduced RSS (regression) or the Gini index (classification), averaged over all trees. This impurity-based importance is fast but biased toward features with many distinct values. **Permutation importance**, which shuffles one feature on validation data and measures the drop in accuracy, is more reliable; Chapter 4 covers both.

**Boosting.** Train a weak model, see where it fails, and train the next model to fix those failures. Repeat. Each model is a small correction, and the sum of many corrections becomes a strong model.

- **AdaBoost (Adaptive Boosting)** re-weights the _training examples_: misclassified points get heavier weights, so the next learner focuses on them.
- **Gradient boosting** fits each new model to the _negative gradient of the loss_, which for squared error is simply the residual (actual minus predicted). It works with any differentiable loss.
- **XGBoost (eXtreme Gradient Boosting), LightGBM (Light Gradient Boosting Machine) and CatBoost (Categorical Boosting)** are engineered gradient-boosting libraries. XGBoost adds regularization and a second-order (Newton) step. LightGBM bins features into histograms and grows trees leaf-wise for speed. CatBoost handles categorical features natively and uses ordered boosting to reduce target leakage.

**Boosting learns slowly on purpose.** Each tree is small and is shrunk by a learning rate before it is added, so no single tree can fit the data hard. The model creeps toward the signal over hundreds of steps, which tends to generalize better than one large model fitted in a single pass.

**BART (Bayesian Additive Regression Trees)** borrows from both families. Like boosting, it fits each tree to what the other trees leave unexplained; like bagging, it injects randomness, by repeatedly perturbing the trees and averaging over many such fits. It performs well with little tuning, though it comes up less often in industry interviews than GBDTs.

**Stacking.** Train several different base models (level 0), for example a random forest, an **SVM (support vector machine)**, which separates classes with the widest possible margin, and a gradient-boosted model. Then train a **meta-model** (level 1) that takes their predictions as inputs and learns how much to trust each. The meta-model must be trained on **out-of-fold** predictions: predictions each base model made on data it did not train on. Otherwise the meta-model learns to trust whichever base model memorized the training set best. **Blending** is a simpler variant that uses one held-out set instead of cross-validation.

<figure class="figure-pending"><div class="figure-pending__box">diagram coming soon</div><figcaption>three ways to combine models</figcaption></figure>

Bagging's members never see each other; each boosting tree depends on the ones before it; a stacking meta-model learns how much to trust each base model.

## 3.2 Core math

**Bias-variance decomposition** (squared error, y = f(x) + ε, noise variance σ²):

$$
\mathbb{E}\big[(y - \hat f(x))^2\big] = \underbrace{\big(\mathbb{E}[\hat f(x)] - f(x)\big)^2}_{\text{bias}^2} + \underbrace{\mathbb{E}\big[(\hat f(x) - \mathbb{E}[\hat f(x)])^2\big]}_{\text{variance}} + \sigma^2
$$

**Why averaging reduces variance, and why correlation limits it.** Average B models, each with variance σ² and pairwise correlation ρ:

$$
\operatorname{Var}\Big(\frac{1}{B}\sum_{b=1}^{B}\hat f_b(x)\Big) = \rho\,\sigma^2 + \frac{1-\rho}{B}\,\sigma^2
$$

As B grows, the second term vanishes but the first, ρσ², stays. With ρ = 0, 100 models cut variance to 1% of one model's. With ρ = 0.7, they only reach 70%. This one formula explains why random forests subsample features: lowering ρ matters more than adding trees.

**How much of the data does a bootstrap sample see?** The chance a given row is never drawn in n draws is:

$$
\left(1 - \frac{1}{n}\right)^{n} \xrightarrow{\,n\to\infty\,} e^{-1} \approx 0.368
$$

So each bootstrap sample contains about 63.2% of the unique rows, roughly two-thirds. The other 36.8% are that model's **out-of-bag (OOB)** rows. Predicting each row with only the trees that never saw it gives a free validation estimate; with enough trees, the OOB error is virtually equivalent to leave-one-out cross-validation.

**AdaBoost** (labels y in {−1, +1}, sample weights w_i summing to 1). At round t, fit a weak learner h_t and compute its weighted error ε_t. Its vote weight and the weight update are:

$$
\alpha_t = \tfrac{1}{2}\ln\frac{1-\varepsilon_t}{\varepsilon_t}, \qquad w_i \leftarrow \frac{w_i \exp\!\big(-\alpha_t\, y_i\, h_t(x_i)\big)}{Z_t}, \qquad H(x) = \operatorname{sign}\Big(\sum_t \alpha_t h_t(x)\Big)
$$

A learner with error below 0.5 gets a positive vote; correctly classified points (y_i h_t(x_i) = +1) lose weight and misclassified points gain it. Z_t renormalizes the weights. AdaBoost is equivalent to stagewise minimization of the exponential loss exp(−yF(x)), which is why it is sensitive to label noise.

**Gradient boosting.** Start from a constant F_0 (the mean, for squared error). At each step m, compute pseudo-residuals, fit a tree h_m to them, and take a small step with learning rate ν:

$$
r_{im} = -\left[\frac{\partial L\big(y_i, F(x_i)\big)}{\partial F(x_i)}\right]_{F = F_{m-1}}, \qquad F_m(x) = F_{m-1}(x) + \nu\, h_m(x)
$$

For L = ½(y − F)², the pseudo-residual is y − F: literally the residual. For log loss it is y − p, the label minus the predicted probability. Boosting is gradient descent in function space: each tree is one step downhill.

**Boosting regression trees, step by step.** This is the residual-fitting form taught in _An Introduction to Statistical Learning_. Set f̂(x) = 0 and r_i = y_i for every training row, then for b = 1, …, B:

1. Fit a tree f̂^b with d splits (d + 1 leaves) to the data (X, r).
2. Add a shrunken copy to the model: f̂(x) ← f̂(x) + λ f̂^b(x).
3. Update the residuals: r_i ← r_i − λ f̂^b(x_i).

The final model is the sum of all the shrunken trees:

$$
\hat f(x) = \sum_{b=1}^{B} \lambda\, \hat f^{\,b}(x)
$$

It has three tuning parameters, and interviewers often ask about each:

| Parameter                              | What it controls                                                   | Typical choice                                                     |
| -------------------------------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------ |
| B, number of trees                     | Total fitting capacity; unlike bagging, too many trees can overfit | Cross-validation or early stopping                                 |
| λ, shrinkage (learning rate)           | How fast the model learns                                          | 0.001 to 0.1; a smaller λ needs a larger B                         |
| d, splits per tree (interaction depth) | The order of feature interactions each tree can capture            | Small; d = 1 (stumps) gives an additive model with no interactions |

**XGBoost's objective** adds a penalty on tree complexity (T leaves with weights w_j) and uses a second-order Taylor expansion of the loss. With G_j and H_j the sums of first and second derivatives over the rows in leaf j:

$$
\Omega(f) = \gamma T + \tfrac{1}{2}\lambda \sum_{j=1}^{T} w_j^2, \qquad w_j^{*} = -\frac{G_j}{H_j + \lambda}
$$

The gain of a split compares the children with the parent; a split is kept only if the gain exceeds γ, which acts as built-in pruning:

$$
\text{Gain} = \frac{1}{2}\left[\frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\right] - \gamma
$$

## 3.3 Code

**Gradient boosting from scratch** (squared error). Writing this in an interview shows you understand boosting, not just the library call:

```python
import numpy as np
from sklearn.tree import DecisionTreeRegressor

class TinyGBM:
    def __init__(self, n_trees=100, lr=0.1, max_depth=3):
        self.n_trees, self.lr, self.max_depth = n_trees, lr, max_depth

    def fit(self, X, y):
        self.f0 = y.mean()                       # F_0: best constant for squared error
        pred = np.full(len(y), self.f0)
        self.trees = []
        for _ in range(self.n_trees):
            residual = y - pred                  # negative gradient of 1/2 (y - F)^2
            tree = DecisionTreeRegressor(max_depth=self.max_depth).fit(X, residual)
            pred += self.lr * tree.predict(X)    # small step: F_m = F_{m-1} + lr * h_m
            self.trees.append(tree)
        return self

    def predict(self, X):
        return self.f0 + self.lr * sum(t.predict(X) for t in self.trees)
```

On a synthetic regression set (1,000 rows, 10 features), 200 trees cut test **MSE (mean squared error)** from about 21,400 (predicting the mean) to about 1,800. This class is the step-by-step algorithm above with λ = `lr` and depth-limited trees; the only difference is that it starts from the mean instead of zero.

**Bagging, boosting and stacking with scikit-learn** on the breast-cancer dataset (569 rows, 30 features):

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                              HistGradientBoostingClassifier, StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)

# Bagging: random forest with a free out-of-bag estimate
rf = RandomForestClassifier(n_estimators=300, max_features="sqrt",
                            oob_score=True, n_jobs=-1, random_state=0).fit(X_tr, y_tr)
print(rf.oob_score_, rf.score(X_te, y_te))

# Boosting: AdaBoost, and histogram gradient boosting with early stopping
ada = AdaBoostClassifier(n_estimators=200, learning_rate=0.5, random_state=0).fit(X_tr, y_tr)
gb = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=500,
                                    early_stopping=True, validation_fraction=0.1,
                                    random_state=0).fit(X_tr, y_tr)

# Stacking: diverse base models, logistic-regression meta-model on out-of-fold probabilities
stack = StackingClassifier(
    estimators=[("rf", RandomForestClassifier(n_estimators=300, random_state=0)),
                ("svm", make_pipeline(StandardScaler(), SVC(probability=True, random_state=0))),
                ("gb", HistGradientBoostingClassifier(random_state=0))],
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5, stack_method="predict_proba").fit(X_tr, y_tr)
```

In one run, test accuracy landed between 0.94 and 0.96 for all four, and the random forest's OOB estimate (0.962) was close to its test score (0.944). On a dataset this small, the differences are within noise; the lesson is the API and the OOB estimate, not a ranking. Note `cv=5` in `StackingClassifier`: scikit-learn builds the out-of-fold predictions for you.

## 3.4 In system design

- **GBDTs are the default for tabular problems:** fraud scoring, credit risk, churn, **CTR (click-through rate)** prediction, the share of shown items that get clicked, and the ranking stage of search and recommendations (LambdaMART is gradient boosting with a ranking loss). They need little preprocessing, handle missing values, and train fast on CPUs.
- **A strong baseline.** In a design round, propose a GBDT baseline on hand-built features before a deep model; it sets the bar the deep model must beat.
- **Serving cost.** Inference time grows with trees × depth. A model with 2,000 trees of depth 8 may miss a 10 ms budget; reduce trees with early stopping, or compile the model with Treelite or export it to **ONNX (Open Neural Network Exchange)**, a portable model format with fast runtimes.
- **Ensembles of deep models** (several checkpoints or seeds averaged) are common in competitions and for uncertainty estimates, but multiply serving cost. **Distillation**, training one small model to mimic the ensemble's predictions, is a common compromise.
- **Stacking is rare in production.** Each base model is another pipeline to monitor, retrain and debug. Use it when the accuracy gain is worth the operational cost, and say so explicitly.
- **Random forests in practice** give robust baselines, OOB validation, and feature-importance estimates, but are usually beaten by tuned boosting on accuracy.

## 3.5 Interview Q&A

**Q1. What is the difference between bagging and boosting?** Bagging trains models independently on bootstrap samples and averages them, reducing variance; it suits high-variance learners like deep trees. Boosting trains models sequentially, each correcting the previous ensemble's errors, reducing bias; it suits weak learners like shallow trees. Bagging parallelizes easily; boosting is sequential across rounds, though each tree's construction can be parallelized.

**Q2. Why does a random forest subsample features at each split?** To decorrelate the trees. The variance of an average is ρσ² + (1−ρ)σ²/B, so with highly correlated trees, adding more barely helps. Restricting each split to a random subset (by default √p features for classification) stops every tree from leaning on the same strong feature.

**Q3. Can a random forest overfit? Does adding trees cause overfitting?** Adding trees does not make a random forest overfit; the error converges as B grows. The individual trees can still overfit noisy data, and the forest inherits that through its correlation term. Control it with `max_depth`, `min_samples_leaf` and `max_features`.

**Q4. What is the out-of-bag error?** Each bootstrap sample omits about 36.8% of rows. Predicting each row using only the trees that did not see it gives a validation estimate for free, close to cross-validation in quality.

**Q5. Explain gradient boosting in two sentences.** Start from a constant prediction, then repeatedly fit a small tree to the negative gradient of the loss at the current predictions and add a shrunken version of it. For squared error the negative gradient is the residual, so each tree literally predicts what the ensemble is still getting wrong.

**Q6. What does the learning rate do in boosting, and how does it interact with the number of trees?** It shrinks each tree's contribution. Smaller rates generalize better but need more trees, so tune them together and choose the number of trees by early stopping on a validation set.

**Q7. How is XGBoost different from classic gradient boosting?** It uses second-order gradient information, adds L1/L2 regularization on leaf weights and a per-leaf penalty γ, learns a default direction for missing values, and is engineered for speed (sparsity-aware splits, parallel split finding, cache-aware layouts). LightGBM goes further with histogram binning and leaf-wise growth; CatBoost specializes in categorical features.

**Q8. Why is boosting more sensitive to label noise than bagging?** Boosting keeps focusing on the examples it gets wrong, and mislabeled points are exactly those, so later learners chase noise. AdaBoost's exponential loss makes this worse because a badly misclassified point's weight grows exponentially. Mitigate with a lower learning rate, subsampling, early stopping, or a robust loss.

**Q9. How do you avoid leakage in stacking?** Train the meta-model only on out-of-fold predictions: split the training data into K folds, and for each fold, predict it with base models trained on the other K−1 folds. At test time, use base models retrained on all training data (or the average of the fold models).

**Q10. When would an ensemble not help?** When the members make the same mistakes (highly correlated models), when the error is mostly bias that averaging cannot fix (bagging linear models), or when latency, memory or interpretability budgets rule out multiple models.

**Q11. Why prune a single tree but not the trees in a random forest?** A single deep tree has high variance, so pruning trades a little bias for a large cut in variance. In a forest, averaging already removes the variance, so each tree is grown deep to keep its bias low.

**Q12. What does the interaction depth d control in boosting?** The number of splits per tree, which limits how many features one tree can combine. With d = 1 (stumps) each tree uses one feature and the model is additive; d = 2 allows pairwise interactions, and so on. Small d often works well because boosting adds many trees, and larger d raises the risk of overfitting.

## 3.6 Cheat sheet and pitfalls

**Remember:**

- Bagging → variance down, parallel, strong learners, average or vote. Boosting → bias down, sequential, weak learners, weighted sum. Stacking → a learned combiner over diverse models.
- Trees split greedily by Gini, entropy or RSS. A single tree is pruned (cost-complexity, α by cross-validation); forest trees are grown deep and averaged.
- Variance of an average: ρσ² + (1−ρ)σ²/B. Diversity (low ρ) matters more than count.
- Random forest: m ≈ √p (classification) or p/3 (regression); m = p is bagging; more trees never cause overfitting.
- Bootstrap sample ≈ 63.2% unique rows; the other 36.8% give the OOB estimate, close to leave-one-out cross-validation.
- AdaBoost: α_t = ½ ln((1−ε_t)/ε_t); re-weights examples; minimizes exponential loss.
- Gradient boosting: fit trees to negative gradients; residuals for squared error. Tuning parameters B, λ, d: a small λ needs a large B, d = 1 gives an additive model, and too large a B can overfit.
- XGBoost leaf weight: w\* = −G/(H+λ); a split is kept if its gain exceeds γ.
- Tabular data default: a tuned GBDT (XGBoost, LightGBM, CatBoost) with early stopping.

**Pitfalls that cost candidates:**

- Saying "bagging reduces bias" or "boosting reduces variance" as the main effect: it is the other way round.
- Claiming more trees make a random forest overfit, or forgetting that more boosting rounds _can_ overfit.
- Boosting without early stopping or a validation set.
- Training a stacking meta-model on in-sample predictions.
- Treating tree feature importances as causal or reliable for correlated features (Chapter 4 covers better tools).
- Proposing a large ensemble in a design round without accounting for latency and operational cost.

## 3.7 Further reading

<ol start="3">
  <li><a href="https://en.wikipedia.org/wiki/Bootstrap_aggregating">Bootstrap aggregating</a> (Wikipedia)</li>
  <li><a href="https://aws.amazon.com/what-is/boosting/">What is boosting?</a> (AWS)</li>
  <li><a href="https://machinelearningmastery.com/stacking-ensemble-machine-learning-with-python/">Stacking ensemble machine learning with Python</a> (Machine Learning Mastery)</li>
</ol>

**Supplementary:** Gareth James, Daniela Witten, Trevor Hastie and Robert Tibshirani, [An Introduction to Statistical Learning](https://www.statlearning.com/) (ISLR), Chapter 8, Tree-Based Methods, for trees, bagging, random forests, boosting and BART; Section 2.2 covers the bias-variance trade-off. The 2023 edition with Python labs adds Jonathan Taylor as co-author.
