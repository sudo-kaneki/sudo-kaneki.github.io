---
title: "Classical Models: Supervised and Unsupervised Learning"
chapter: 2
permalink: "/blog/cracking-the-ml-interview/classical-models/"
updated: 2026-11-01
description: "Before ensembles, deep networks or LLMs, interviewers check that you can derive, explain and choose between the classical models."
---

Before ensembles, deep networks or LLMs, interviewers check that you can derive, explain and choose between the classical models. Linear and logistic regression, SVMs, decision trees, k-means and PCA are still the baselines in production systems and the warm-up questions in almost every ML round. This chapter follows _An Introduction to Statistical Learning_ (ISLR) closely.

Because it covers several models, each section is self-contained: intuition, core math, code and interview questions for one model. System-design notes, the cheat sheet and further reading come at the end of the chapter.

## 2.1 Supervised and unsupervised learning

The first split in ML is whether your training data comes with answers. Every model in this chapter sits on one side of that split, and the vocabulary below frames every later question an interviewer asks.

### Intuition

- **Supervised learning**: each training example has inputs X and a known output y, called the **label** or target. The model learns a function f so that y ≈ f(X) on new data. When y is a number (price, delivery time) the task is **regression**; when y is a category (spam or not, fraud or not) it is **classification**.
- **Unsupervised learning**: there is only X, no y. The goal is to find structure: groups of similar items (**clustering**) or a smaller set of features that keeps most of the information (**dimensionality reduction**). There is no right answer to check against, so evaluation is harder and more subjective.
- **Semi-supervised learning** uses a few labels plus many unlabeled examples. **Self-supervised learning** creates labels from the data itself, such as predicting the next word or a masked word; it is how LLMs and embedding models are pretrained.

| Task                     | Output                      | Examples                                 | Models in this chapter                                                 |
| ------------------------ | --------------------------- | ---------------------------------------- | ---------------------------------------------------------------------- |
| Regression               | A number                    | House price, delivery time               | Linear regression, regression trees                                    |
| Classification           | A category                  | Spam, fraud, disease                     | Logistic regression, SVMs, classification trees, kNN, naive Bayes, LDA |
| Clustering               | Group IDs, no labels needed | Customer segments, near-duplicate images | k-means, hierarchical clustering                                       |
| Dimensionality reduction | Fewer, combined features    | Visualizing or compressing embeddings    | PCA                                                                    |

**Prediction versus inference.** ISLR separates two reasons for building a model. In **prediction** you care only that ŷ is accurate, and the model can be a black box. In **inference** you want to understand how each input affects y: which features matter, in which direction and by how much. Simple, interpretable models such as linear and logistic regression suit inference.

**Parametric versus non-parametric.** A **parametric** model assumes a fixed form, such as a straight line, and estimates a fixed number of parameters; it needs less data but fails if the form is wrong. A **non-parametric** model, such as k-nearest neighbours or a decision tree, lets the data decide the shape; it is more flexible but needs more data and can overfit.

**Flexibility versus interpretability.** More flexible models fit more complex patterns, which lowers bias but raises variance and makes the model harder to explain. Roughly, from least to most flexible: lasso, linear regression, generalized additive models, decision trees, SVMs with non-linear kernels, bagging and boosting, deep networks.

<figure class="figure-pending"><div class="figure-pending__box">diagram coming soon</div><figcaption>flexibility versus interpretability · after ISLR, Figure 2.7</figcaption></figure>

The positions are qualitative, after ISLR's figure: the further right a model sits, the more patterns it can fit and the harder its predictions are to explain.

### Core math

ISLR writes the relationship between inputs and output as a true function plus noise ε, which has mean zero and cannot be predicted from X:

$$
y = f(X) + \varepsilon
$$

The expected squared error of a prediction ŷ = f̂(X) splits into a part a better model could remove and a part no model can:

$$
\mathbb{E}\big[(y - \hat y)^2\big] = \underbrace{\big(f(X) - \hat f(X)\big)^2}_{\text{reducible}} + \underbrace{\operatorname{Var}(\varepsilon)}_{\text{irreducible}}
$$

Training error always falls as flexibility rises, but test error is U-shaped: it falls while the model learns real signal, then rises once it starts fitting noise. Always choose models by test or validation error, never training error.

### Interview Q&A

**Q1. What is the difference between supervised and unsupervised learning? Give an example of each.** Supervised learning trains on labeled examples to predict the label, for example predicting whether a transaction is fraud from past labeled transactions. Unsupervised learning finds structure in unlabeled data, for example grouping customers into segments by purchase behavior.

**Q2. Is predicting a 1-to-5 star rating regression or classification?** Either can work, and the interviewer wants the trade-off. Regression respects the order (predicting 4 for a true 5 is better than predicting 1) but can output 3.7; classification gives valid labels but treats 1 and 5 as equally different from 4. Ordinal regression handles both.

**Q3. What is the difference between parametric and non-parametric models?** Parametric models assume a fixed functional form with a fixed number of parameters (linear regression); non-parametric models let complexity grow with the data (kNN, trees). Parametric models are data-efficient and interpretable but biased if the form is wrong; non-parametric models are flexible but need more data and are prone to overfitting.

**Q4. What is self-supervised learning, and why does it matter for LLMs?** It creates training labels from unlabeled data itself, such as the next token in a sentence. It lets models learn from huge unlabeled corpora; supervised fine-tuning on a smaller labeled set comes afterwards.

## 2.2 Linear regression

Linear regression predicts a number as a weighted sum of features. It is the first model to try on any regression problem and the yardstick every fancier model must beat.

### Intuition

- Fit the straight line (with many features, the flat **hyperplane**) that makes the squared vertical distances from the points to it as small as possible. This is **ordinary least squares (OLS)**.
- Each coefficient β_j is the expected change in y for a one-unit increase in feature x_j, **holding all other features fixed**. That last phrase is what interviewers listen for.
- **Why squared error?** It is smooth and convex, so there is one minimum with a closed-form solution. It is also the maximum-likelihood choice when the noise is Gaussian. The downside: big errors count heavily, so outliers pull the line (Chapter 8 covers MAE and Huber loss).
- "Linear" means linear in the **parameters**, not the features. Adding x², log x or a product x₁·x₂ (an **interaction term**: the effect of one feature depends on another) still gives a linear model.
- **Categorical features** enter as **dummy variables**: one 0/1 column per level, minus one reference level. Keeping all levels plus an intercept makes the columns perfectly collinear, the **dummy variable trap**.

**Assumptions**, and which ones matter:

| Assumption                               | What it means                                                | If violated                                                    |
| ---------------------------------------- | ------------------------------------------------------------ | -------------------------------------------------------------- |
| Linearity                                | y depends linearly on the features (after any transforms)    | Biased predictions; add transforms or use a flexible model     |
| Independent errors                       | One row's error says nothing about another's                 | Standard errors too small; common with time series             |
| Constant variance (**homoscedasticity**) | Errors have the same spread at every level of the prediction | Unreliable p-values; transform y or use robust standard errors |
| Normal errors                            | Errors follow a bell curve                                   | Only p-values and confidence intervals suffer, not predictions |
| No perfect **multicollinearity**         | No feature is an exact linear combination of others          | Coefficients cannot be uniquely estimated                      |

### Core math

With n rows, p features and a column of ones for the intercept, the model in matrix form is:

$$
y = X\beta + \varepsilon, \qquad \hat\beta = \arg\min_{\beta}\, \lVert y - X\beta \rVert^2
$$

Setting the gradient to zero gives the **normal equations** and the closed-form solution:

$$
X^\top X\,\hat\beta = X^\top y \quad\Longrightarrow\quad \hat\beta = (X^\top X)^{-1} X^\top y
$$

With one feature, this reduces to the formulas worth memorizing:

$$
\hat\beta_1 = \frac{\sum_i (x_i - \bar x)(y_i - \bar y)}{\sum_i (x_i - \bar x)^2}, \qquad \hat\beta_0 = \bar y - \hat\beta_1 \bar x
$$

**Goodness of fit.** R² is the share of variance in y the model explains, with RSS the residual sum of squares and TSS = Σ(y_i − ȳ)² the total sum of squares. **Adjusted R²** penalizes extra features, so it only rises if a new feature helps more than chance would:

$$
R^2 = 1 - \frac{\text{RSS}}{\text{TSS}}, \qquad R^2_{\text{adj}} = 1 - (1 - R^2)\,\frac{n-1}{n-p-1}
$$

**Is a coefficient significant?** Estimate the noise variance as σ̂² = RSS / (n − p − 1). The standard errors are the square roots of the diagonal of σ̂²(XᵀX)⁻¹, and the t-statistic t_j = β̂_j / SE(β̂_j) tests whether β_j = 0.

**Multicollinearity** is measured by the **variance inflation factor (VIF)**, where R_j² comes from regressing feature j on all the other features. A VIF above 5 to 10 signals trouble:

$$
\text{VIF}_j = \frac{1}{1 - R_j^2}
$$

**Cost.** The closed form costs about O(np² + p³): building XᵀX, then solving a p × p system. Gradient descent costs O(np) per step, which wins when p is large or the data does not fit in memory.

### Code

The normal equations by hand, checked against scikit-learn:

```python
import numpy as np
from sklearn.linear_model import LinearRegression

rng = np.random.default_rng(0)
n = 200
X = rng.normal(size=(n, 2))
y = 3 + 2 * X[:, 0] - 1 * X[:, 1] + rng.normal(scale=0.5, size=n)  # true beta = (3, 2, -1)

Xb = np.c_[np.ones(n), X]                     # add the intercept column
beta = np.linalg.solve(Xb.T @ Xb, Xb.T @ y)   # solve the normal equations, no explicit inverse
print(beta.round(3))                          # [ 2.999  1.975 -0.94 ]

lr = LinearRegression().fit(X, y)
print(lr.intercept_.round(3), lr.coef_.round(3))  # 2.999 [ 1.975 -0.94 ]
```

In practice, never invert XᵀX explicitly: `np.linalg.solve` or `np.linalg.lstsq` are faster and numerically stabler. For p-values and confidence intervals, statsmodels' `OLS` prints the full regression table.

### Interview Q&A

**Q5. What are the assumptions of linear regression, and which matter for prediction?** Linearity, independent errors, constant error variance, normal errors and no perfect multicollinearity. For pure prediction, linearity matters most; independence, constant variance and normality mainly affect standard errors, p-values and confidence intervals.

**Q6. What is multicollinearity, and why is it a problem?** Two or more features are highly correlated, so the model cannot tell their effects apart. Coefficients become unstable, can flip sign, and get large standard errors, although predictions may still be fine. Detect it with correlation matrices or VIF; fix it by dropping or combining features, or by using ridge regression (Chapter 9).

**Q7. What happens if XᵀX is not invertible?** This happens with perfectly collinear features or with more features than rows (p > n). OLS has no unique solution. Remove redundant features, use the pseudo-inverse, or add ridge regularization, which makes the matrix XᵀX + λI invertible.

**Q8. R² versus adjusted R²: which would you report, and can R² be negative?** R² never decreases when you add a feature, even a useless one, so adjusted R² is better for comparing models with different numbers of features. On training data with an intercept, R² is between 0 and 1. On test data it can be negative: the model predicts worse than simply predicting the mean.

**Q9. How do you interpret a coefficient when y is log-transformed?** If the model predicts log y, a one-unit increase in x_j multiplies y by e^(β_j), which is about a 100·β_j percent change when β_j is small. Log-transforming a skewed target like price is common for exactly this reason.

**Q10. Your residual plot fans out as predictions grow. What does that mean, and what do you do?** The error variance grows with the prediction: heteroscedasticity. Coefficients stay unbiased but standard errors are wrong. Transform y (log or square root), use weighted least squares, or report robust standard errors.

**Q11. Closed-form solution or gradient descent?** The closed form is exact and fast when p is small to moderate. Gradient descent scales to many features and to data streamed in batches, and it is the only option once you add losses or regularizers without closed forms, such as lasso.

## 2.3 Logistic regression

Logistic regression is the standard baseline for classification: a linear score squashed into a probability. Despite its name, it classifies; "regression" refers to modeling the log-odds as a linear function.

### Intuition

- Compute a linear score z = β₀ + β₁x₁ + … + β_p x_p, exactly as in linear regression. Then pass it through the **sigmoid** (logistic) function, which maps any number to a probability between 0 and 1: large positive z gives a probability near 1, large negative z near 0, and z = 0 gives 0.5.
- **Why not just use linear regression on 0/1 labels?** Its predictions can fall below 0 or above 1, and a few extreme points drag the line and move the decision boundary.
- The **odds** are p / (1 − p): a probability of 0.8 is odds of 4 to 1. Logistic regression assumes the **log-odds** are linear in the features. So a one-unit increase in x_j multiplies the odds by e^(β_j), the **odds ratio**.
- The **decision boundary** is where z = 0, a straight line or hyperplane, so logistic regression is a **linear classifier**. Curved boundaries need engineered features (squares, interactions) or a different model.
- It is trained by **maximum likelihood estimation (MLE)**: choose the coefficients that make the observed labels most probable. That is the same as minimizing **log loss**, also called binary cross-entropy. There is no closed form, but the loss is convex, so gradient descent or Newton's method finds the single global minimum.
- **Multiclass:** the **softmax** (multinomial) version gives each class its own score and normalizes them into probabilities; the alternative is one-vs-rest, one binary model per class.
- Probabilities from logistic regression are usually well **calibrated**: among cases predicted at 0.7, roughly 70% are positive. That makes it popular wherever the probability itself is used, as in ad pricing or risk scores.

### Core math

The model, and the same model written as log-odds:

$$
p(x) = \sigma(z) = \frac{1}{1 + e^{-z}}, \quad z = \beta_0 + \beta^\top x, \qquad\qquad \log\frac{p(x)}{1 - p(x)} = \beta_0 + \beta^\top x
$$

The likelihood of labels y_i ∈ {0, 1}, and the negative log-likelihood (log loss) that training minimizes:

$$
L(\beta) = \prod_{i=1}^{n} p_i^{\,y_i}(1 - p_i)^{1 - y_i}, \qquad \ell(\beta) = -\sum_{i=1}^{n} \Big[\, y_i \log p_i + (1 - y_i)\log(1 - p_i) \Big]
$$

Using σ′(z) = σ(z)(1 − σ(z)), the gradient comes out remarkably clean, the same form as linear regression's. The Hessian, XᵀWX with W = diag(p_i(1 − p_i)), is positive semi-definite, which proves the loss is convex:

$$
\nabla_\beta\, \ell = X^\top (p - y)
$$

For K classes, softmax replaces the sigmoid:

$$
P(y = k \mid x) = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}, \qquad z_k = \beta_{k0} + \beta_k^\top x
$$

### Code

Logistic regression from scratch with gradient descent and a small L2 penalty, on the breast-cancer dataset with standardized features:

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
sc = StandardScaler().fit(X_tr)
X_tr, X_te = sc.transform(X_tr), sc.transform(X_te)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def fit_logreg(X, y, lr=0.1, epochs=2000, l2=1e-3):
    Xb = np.c_[np.ones(len(X)), X]
    w = np.zeros(Xb.shape[1])
    for _ in range(epochs):
        p = sigmoid(Xb @ w)
        grad = Xb.T @ (p - y) / len(y) + l2 * np.r_[0, w[1:]]  # don't penalize the intercept
        w -= lr * grad
    return w

w = fit_logreg(X_tr, y_tr)
p_te = sigmoid(np.c_[np.ones(len(X_te)), X_te] @ w)
print(((p_te > 0.5) == y_te).mean())                                  # 0.965
print(LogisticRegression(max_iter=1000).fit(X_tr, y_tr).score(X_te, y_te))  # 0.958
```

The two land within one test example of each other; the small gap comes from different regularization strengths. In scikit-learn, `C` is the _inverse_ of regularization strength: a larger `C` means a weaker penalty.

### Interview Q&A

**Q12. Why do we use log loss instead of mean squared error for logistic regression?** Log loss is the negative log-likelihood of a Bernoulli model, so minimizing it is maximum likelihood. Combined with the sigmoid it is convex, and its gradient (p − y) stays large when the model is confidently wrong. MSE through a sigmoid is non-convex and its gradient vanishes exactly when predictions are badly wrong.

**Q13. A coefficient is 0.7. What does it mean?** Holding other features fixed, a one-unit increase in that feature multiplies the odds of the positive class by e^0.7 ≈ 2.0, roughly doubling them. It does not add a fixed amount to the probability; the change in probability depends on where you start on the sigmoid.

**Q14. Is logistic regression a linear model? Can it learn a curved boundary?** Yes, its decision boundary is linear in the features it is given. It can learn curved boundaries in the original space if you add non-linear features such as squares or interactions, because the boundary is then linear in the expanded features.

**Q15. What happens when the classes are perfectly separable?** The likelihood keeps improving as the coefficients grow, so they diverge toward infinity and the optimizer never converges. Regularization (L2 or L1) keeps the coefficients finite.

**Q16. How do you choose the classification threshold?** Not automatically at 0.5. Pick it from the costs of false positives and false negatives, or to meet a target precision or recall on a validation set. With rare positives, the best threshold is often far below 0.5.

**Q17. Do you need to scale features for logistic regression?** Unregularized MLE gives the same predictions either way. But scaling speeds up gradient descent, and it matters with regularization, because the penalty treats all coefficients alike and a feature measured in large units would be penalized unfairly.

## 2.4 Support vector machines

A **support vector machine (SVM)** separates two classes with the boundary that stays as far as possible from both. ISLR builds it in three steps, and interviewers expect you to know the names: the maximal margin classifier, the support vector classifier, and the support vector machine.

### Intuition

- **Hyperplane.** In p dimensions, a hyperplane is the flat surface β₀ + βᵀx = 0: a line in 2D, a plane in 3D. Points on one side get class +1, points on the other −1.
- **Maximal margin classifier (hard margin).** If the classes can be separated, infinitely many hyperplanes do it. Pick the one with the widest **margin**, the gap between the hyperplane and the closest points. Those closest points are the **support vectors**: move any of them and the boundary moves; move any other point (without crossing the margin) and nothing changes.
- **Support vector classifier, SVC (soft margin).** Real data overlaps, and a hard margin is fragile: one outlier can swing it. The SVC lets some points sit inside the margin or on the wrong side, each paying a penalty through a **slack variable** ε_i. A tuning parameter trades margin width against violations.
- **Support vector machine.** The SVC with a **kernel**, which lets the boundary curve. A kernel measures similarity between two points as if they had been mapped into a much higher-dimensional space, without computing that mapping: the **kernel trick**. A linear boundary there is a curved boundary in the original space.

**Read the slack values:** ε_i = 0 means the point is on the correct side of the margin; 0 < ε_i ≤ 1 means inside the margin but correctly classified; ε_i > 1 means misclassified. Only points on or inside the margin are support vectors.

<figure class="figure-pending"><div class="figure-pending__box">diagram coming soon</div><figcaption>soft-margin SVM · boundary, margin edges and support vectors</figcaption></figure>

The ringed points are the support vectors. Four sit exactly on the margin edges; the circle inside the margin and the triangle on the wrong side are soft-margin violations, and they are support vectors too. Delete any un-ringed point and the boundary does not move.

### Core math

**Hard margin**, in the standard form used by most textbooks and libraries. The margin width is 2 / ‖w‖, so maximizing it means minimizing ‖w‖:

$$
\min_{w,\,b}\ \tfrac{1}{2}\lVert w \rVert^2 \quad \text{subject to} \quad y_i\,(w^\top x_i + b) \ge 1 \ \text{ for all } i
$$

**Soft margin.** Add slack ξ_i and pay C for each unit of it:

$$
\min_{w,\,b,\,\xi}\ \tfrac{1}{2}\lVert w \rVert^2 + C\sum_{i=1}^{n}\xi_i \quad \text{subject to} \quad y_i\,(w^\top x_i + b) \ge 1 - \xi_i,\ \ \xi_i \ge 0
$$

**Watch the meaning of C.** In the form above, and in scikit-learn, C is a _penalty_: a large C allows few violations, giving a narrow margin, low bias and high variance. ISLR instead writes the soft margin with C as a _budget_ (Σε*i ≤ C), where a large C allows \_more* violations. State which convention you mean in an interview.

**Hinge loss form.** The same problem can be written as a loss plus a penalty. The **hinge loss** is zero for points beyond the margin and grows linearly inside it; it is close in shape to logistic regression's log loss, which is why the two models often perform similarly:

$$
\min_{\beta_0,\,\beta}\ \sum_{i=1}^{n} \max\!\big(0,\ 1 - y_i f(x_i)\big) + \lambda \lVert \beta \rVert^2, \qquad f(x) = \beta_0 + \beta^\top x
$$

**Kernels.** The solution can be written using only inner products with the training points, and only support vectors (the set S) get non-zero weights α_i. Swapping the inner product for a kernel K gives the SVM:

$$
f(x) = \beta_0 + \sum_{i \in S} \alpha_i\, K(x, x_i)
$$

| Kernel                      | K(x, x′)         | Boundary it can draw                                         |
| --------------------------- | ---------------- | ------------------------------------------------------------ |
| Linear                      | xᵀx′             | Straight (the SVC)                                           |
| Polynomial of degree d      | (1 + xᵀx′)^d     | Polynomial curves                                            |
| Radial basis function (RBF) | exp(−γ‖x − x′‖²) | Any smooth shape; γ sets how local each point's influence is |

A large γ makes each point's influence very local, giving wiggly boundaries that can overfit; a small γ gives smooth, nearly linear ones. Tune γ and C together.

### Kernels and the kernel trick, in depth

Kernels are the part of SVMs interviewers probe hardest, because they test whether you understand _why_ the method works rather than just its API. The argument has three steps.

**Step 1: the SVM only needs inner products.** The soft-margin problem has an equivalent **dual form**: instead of one weight per feature, it learns one weight α_i per training point. In the dual, the data appear only through inner products x_iᵀx_j:

$$
\max_{\alpha}\ \sum_{i=1}^{n}\alpha_i - \frac{1}{2}\sum_{i=1}^{n}\sum_{j=1}^{n}\alpha_i\alpha_j\, y_i y_j\, x_i^\top x_j \quad \text{subject to} \quad 0 \le \alpha_i \le C,\ \ \sum_{i=1}^{n}\alpha_i y_i = 0
$$

Points with α_i > 0 are the support vectors; every other point has α_i = 0. A new point is classified by the sign of f(x) = Σ α_i y_i x_iᵀx + b, again only inner products. (ISLR's form f(x) = β₀ + Σ α_i K(x, x_i) folds y_i into α_i.)

**Step 2: map to a richer space.** To get a curved boundary, send each point through a **feature map** φ into a bigger space and fit a _linear_ SVM there. Every inner product x_iᵀx_j becomes φ(x_i)ᵀφ(x_j). A straight boundary in the new space is a curved one back in the original space, as the figure below shows for one feature.

<figure class="figure-pending"><div class="figure-pending__box">diagram coming soon</div><figcaption>feature map x → (x, x²) · 1D data made linearly separable</figcaption></figure>

On the x line, one class sits in the middle and the other on both sides, so no single threshold works. Adding x² lifts the outer points high, and a horizontal line now separates them. A kernel does this kind of lifting implicitly, in many more dimensions.

**Step 3: the trick.** For many useful maps, φ(x)ᵀφ(z) can be computed directly from x and z, without ever building φ. That shortcut is the **kernel**, K(x, z) = φ(x)ᵀφ(z). In two dimensions with φ(x) = (x₁², √2·x₁x₂, x₂²):

$$
\phi(x)^\top\phi(z) = x_1^2 z_1^2 + 2\,x_1 x_2\, z_1 z_2 + x_2^2 z_2^2 = (x_1 z_1 + x_2 z_2)^2 = (x^\top z)^2
$$

Three multiplications in the mapped space collapse into one dot product, squared. With x = (1, 2) and z = (3, −1), both sides equal 1.

**Why it matters.** The saving grows explosively. With p = 100 features, a degree-3 polynomial map has 176,851 features (every monomial up to degree 3), yet the kernel (1 + xᵀz)³ costs one 100-term dot product. The RBF kernel goes further: expanding the exponential shows it contains polynomial kernels of _every_ degree, so its feature space is infinite-dimensional. No explicit map could ever be computed, yet the kernel is one exponential:

$$
e^{-\gamma\lVert x - z\rVert^2} = e^{-\gamma\lVert x\rVert^2}\, e^{-\gamma\lVert z\rVert^2} \sum_{k=0}^{\infty} \frac{(2\gamma)^k}{k!}\,(x^\top z)^k
$$

**Picture the RBF kernel as bumps.** Each support vector places a Gaussian bump of influence around itself, and the prediction is a weighted sum of these bumps plus a bias. γ sets the bump width, roughly 1/√γ: a large γ makes narrow bumps that each cover only their own neighbourhood (a wiggly, overfit boundary); a small γ makes wide bumps that blend into a smooth, nearly linear boundary.

**What counts as a valid kernel?** K must equal an inner product in _some_ feature space. **Mercer's condition** gives the practical test: for any set of points, the **Gram matrix** (kernel matrix) with entries K(x_i, x_j) must be symmetric and **positive semi-definite (PSD)**, meaning it has no negative eigenvalues. Sums, positive multiples and products of valid kernels are valid. An arbitrary similarity score that is not PSD can make the SVM problem non-convex.

| Kernel         | Choose it when                                                                                | Watch out for                                                |
| -------------- | --------------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| Linear         | Many features relative to rows (text with p ≫ n), or you need speed and interpretable weights | Misses non-linear structure                                  |
| RBF            | The default for dense, low-to-moderate dimensional data with non-linear structure             | Tune C and γ together on a log-scale grid; scale features    |
| Polynomial     | Feature interactions of a known degree matter                                                 | Numerically unstable at high degree; more parameters to tune |
| Domain kernels | Strings, sequences or graphs (string kernels, graph kernels)                                  | Must still produce a PSD Gram matrix                         |

**Costs and approximations.** The Gram matrix is n × n: at 100,000 rows that is 10¹⁰ entries, about 80 GB in 64-bit floats. Prediction cost also grows with the number of support vectors. Two approximations make kernels scale by building an explicit, low-dimensional feature map and then training a fast linear model on it:

- **Nyström approximation**: builds a low-rank approximation of the Gram matrix from a sample of m training points.
- **Random Fourier features**: random cosine features whose inner products approximate the RBF kernel. Linear-attention transformers such as the Performer use the same idea to approximate softmax attention.

**The trick goes beyond SVMs.** Any algorithm that touches data only through inner products can be kernelized: kernel ridge regression, kernel PCA, kernel k-means, and Gaussian processes, where the kernel is the covariance between function values.

A kernel approximation in practice: an RBF Nyström map with 100 components feeding a plain logistic regression, on the same moons split as the code below:

```python
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression

# The kernel trick, checked by hand
x, z = np.array([1.0, 2.0]), np.array([3.0, -1.0])
phi = lambda v: np.array([v[0]**2, np.sqrt(2) * v[0] * v[1], v[1]**2])
print(phi(x) @ phi(z), (x @ z) ** 2)       # 1.0 1.0

# An approximate RBF kernel feeding a linear model
X, y = make_moons(n_samples=500, noise=0.25, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
model = make_pipeline(StandardScaler(),
                      Nystroem(kernel="rbf", gamma=1.0, n_components=100, random_state=0),
                      LogisticRegression(max_iter=1000)).fit(X_tr, y_tr)
print(model.score(X_te, y_te))               # 0.92, close to the exact RBF SVM's 0.94
```

### Code

A linear SVC against an RBF SVM on two interleaving half-moons, tuning C and γ by cross-validation:

```python
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = make_moons(n_samples=500, noise=0.25, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

linear = make_pipeline(StandardScaler(), SVC(kernel="linear", C=1.0)).fit(X_tr, y_tr)
print(linear.score(X_te, y_te))                       # 0.827

grid = GridSearchCV(make_pipeline(StandardScaler(), SVC(kernel="rbf")),
                    {"svc__C": [0.1, 1, 10, 100], "svc__gamma": [0.1, 1, 10]},
                    cv=5).fit(X_tr, y_tr)
print(grid.best_params_, grid.score(X_te, y_te))      # {'svc__C': 10, 'svc__gamma': 1} 0.94
print(grid.best_estimator_[-1].n_support_.sum())      # 66 support vectors out of 350 rows
```

The curved RBF boundary fits the moons far better than a straight line. Only 66 of 350 training points are support vectors; the other 284 could be deleted without changing the model.

### Interview Q&A

**Q18. What are support vectors, and why do they matter?** They are the training points on or inside the margin, and they alone determine the boundary. That makes the model sparse at prediction time and robust to points far from the boundary, but sensitive to points near it.

**Q19. What does C control in an SVM?** In scikit-learn's convention, C is the penalty on margin violations. A large C tolerates few violations: narrow margin, low bias, high variance. A small C tolerates many: wide margin, more bias, less variance. Choose it by cross-validation, and mention that ISLR's C is a budget with the opposite direction.

**Q20. Explain the kernel trick. What makes a valid kernel, and how would you scale a kernel SVM?** In its dual form the SVM depends on the data only through inner products. A kernel computes the inner product the points would have after a mapping into a higher-dimensional space (for RBF, an infinite-dimensional one) without computing the mapping, so a linear boundary there becomes a curved boundary in the original space at little extra cost. A function is a valid kernel if every Gram matrix it produces is symmetric positive semi-definite (Mercer's condition). The Gram matrix grows with n², so for large data use the Nyström approximation or random Fourier features to build explicit approximate features and train a linear model on them.

**Q21. SVM or logistic regression?** Their losses are similar, so with a linear kernel they often perform alike. Logistic regression gives calibrated probabilities and scales to huge datasets; SVMs give a maximum-margin boundary, handle non-linearity cleanly with kernels, and do well with many features and few rows. SVM "probabilities" need an extra calibration step, Platt scaling, which fits a sigmoid to the SVM's scores.

**Q22. Why must you scale features before an SVM?** The margin and the RBF kernel are both based on distances. A feature measured in thousands would dominate one measured in fractions, so the boundary would ignore the small-scale feature.

**Q23. Why are kernel SVMs rare on very large datasets?** Training kernel SVMs scales between roughly O(n²) and O(n³) in the number of rows, and the kernel matrix needs O(n²) memory. Beyond a few hundred thousand rows, linear models, gradient-boosted trees or neural networks are cheaper.

**Q24. How does an SVM handle more than two classes?** By combining binary SVMs: **one-vs-one** trains a classifier for every pair of classes and takes a vote, while **one-vs-rest** trains one classifier per class against all others and picks the highest score.

## 2.5 Decision trees

A decision tree predicts by asking a sequence of yes/no questions about the features. It is the most explainable model in this chapter and the building block of random forests and gradient boosting (Chapter 3).

### Intuition

- A tree splits the feature space into rectangles. Each internal **node** asks one question ("income ≥ 50k?"), each branch is an answer, and each final region, a **leaf**, predicts the mean of its training points (a **regression tree**) or their majority class (a **classification tree**).
- Trees are grown by **recursive binary splitting**: at each node, try every feature and every threshold, and keep the split that most reduces error. Regression uses **RSS (residual sum of squares)**; classification uses an impurity measure such as the **Gini index** or **entropy**. The search is **greedy**: it takes the best split now and never looks ahead.
- Left alone, a tree keeps splitting until each leaf is pure, which memorizes the training data: low bias, very high variance. A slightly different sample can produce a completely different tree.
- Control it by **stopping early** (maximum depth, minimum samples per leaf) or by **cost-complexity pruning**: grow a large tree, then cut back branches whose error reduction is not worth their size. ISLR prefers pruning, because a weak split early on can enable a very good split below it.
- Splits are **axis-aligned**: each question involves one feature. A diagonal boundary needs a staircase of many splits, which is where linear models win.

<figure class="figure-pending"><div class="figure-pending__box">diagram coming soon</div><figcaption>a classification tree · 3 questions, 4 leaves</figcaption></figure>

To classify an applicant, start at the top and follow the answers down to a leaf. Each path from the root to a leaf is one readable rule, such as "income ≥ 50k and debt-to-income < 0.4 → approve".

| Strengths                                                                    | Weaknesses                                                                     |
| ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| Easy to explain; mirrors human decision-making                               | Usually less accurate than other models on its own                             |
| Can be drawn and read by non-experts                                         | High variance: small data changes give a different tree                        |
| Handles categorical features without dummy variables in many implementations | Axis-aligned splits approximate diagonal or smooth boundaries poorly           |
| Needs no feature scaling; robust to monotonic transforms                     | A greedy search can miss better trees                                          |
| Captures interactions automatically                                          | Predictions are piecewise constant: no extrapolation beyond the training range |

When the true relationship is close to linear, linear regression beats a tree; when it is highly non-linear with interactions, a tree can win. Ensembles (Chapter 3) keep the tree's strengths and remove most of its variance.

### Core math

**Impurity.** For a classification node m, let p̂_mk be the share of its training points that belong to class k. The Gini index G_m and entropy D_m are:

$$
G_m = \sum_{k=1}^{K} \hat p_{mk}\,(1 - \hat p_{mk}) = 1 - \sum_{k=1}^{K} \hat p_{mk}^{\,2} \qquad\qquad D_m = -\sum_{k=1}^{K} \hat p_{mk}\,\log_2 \hat p_{mk}
$$

Both are zero for a pure node and largest when classes are evenly mixed. A split is chosen to maximize the **impurity decrease** (for entropy, the **information gain**): the parent's impurity minus the children's impurity, weighted by their sizes:

$$
\Delta I = I(\text{parent}) - \frac{n_L}{n}\,I(\text{left}) - \frac{n_R}{n}\,I(\text{right})
$$

**Worked example.** A node holds 5 positives and 5 negatives, so its Gini index is 0.5 and its entropy is 1 bit. A split sends 4 positives and 1 negative left, 1 positive and 4 negatives right. Each child has Gini 1 − (0.8² + 0.2²) = 0.32 and entropy of about 0.722 bits. The Gini decrease is 0.5 − 0.32 = 0.18, and the information gain is 1 − 0.722 ≈ 0.278 bits.

**Regression trees** minimize RSS over the regions R_j the tree creates, where ŷ\_{R_j} is the mean response in region j:

$$
\text{RSS} = \sum_{j=1}^{J}\sum_{i \in R_j} \big(y_i - \hat y_{R_j}\big)^2
$$

**Cost-complexity pruning** grows a large tree, then for each penalty α ≥ 0 finds the subtree T that minimizes training error plus α times its number of leaves |T|. A larger α gives a smaller tree, and α is chosen by cross-validation:

$$
\sum_{m=1}^{|T|}\;\sum_{i:\,x_i \in R_m} \big(y_i - \hat y_{R_m}\big)^2 + \alpha\,|T|
$$

**Cost of a split search.** For one feature, sort the n values once (O(n log n)) and sweep the candidate thresholds between consecutive values, updating class counts as you go. Histogram-based libraries such as LightGBM bin values first to make this much faster.

### Code

A fully grown tree against one pruned by cost-complexity, with α picked by 5-fold cross-validation:

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)

full = DecisionTreeClassifier(random_state=0).fit(X_tr, y_tr)

alphas = full.cost_complexity_pruning_path(X_tr, y_tr).ccp_alphas[:-1]  # last alpha = root only
cv = [cross_val_score(DecisionTreeClassifier(random_state=0, ccp_alpha=a), X_tr, y_tr, cv=5).mean()
      for a in alphas]
pruned = DecisionTreeClassifier(random_state=0, ccp_alpha=alphas[int(np.argmax(cv))]).fit(X_tr, y_tr)

for name, m in [("full", full), ("pruned", pruned)]:
    print(name, m.get_n_leaves(), m.get_depth(), m.score(X_te, y_te))
# full    18 leaves, depth 5, test accuracy 0.902
# pruned   5 leaves, depth 3, test accuracy 0.902
```

Pruning kept the same test accuracy with a fifth of the leaves: a tree you can print on a slide. Compare this 0.902 with about 0.94 to 0.96 for the ensembles on the same split in Chapter 3; that gap is why single trees are rarely used alone.

### Interview Q&A

**Q25. Gini index or entropy: which should you use to grow a tree?** Both measure node impurity, are zero for pure nodes and largest for evenly mixed ones, and usually pick very similar splits. Gini avoids a logarithm and is scikit-learn's default; entropy connects to information gain. The plain classification error rate is a poor splitting criterion because it barely changes as nodes get purer, though it is a reasonable metric when pruning.

**Q26. How does a tree pick a threshold for a continuous feature?** It sorts the feature's values and evaluates candidate thresholds between consecutive distinct values, keeping the one with the largest impurity decrease. Doing this for every feature at every node is the main training cost.

**Q27. Why are decision trees called unstable?** The greedy search means an early split decides everything below it. A small change in the data can change the top split and therefore the whole tree. That high variance is exactly what bagging and random forests fix.

**Q28. How do you stop a tree from overfitting?** Limit depth, require a minimum number of samples per leaf or per split, require a minimum impurity decrease, or prune with cost-complexity and pick α by cross-validation. If accuracy still matters more than a single readable tree, switch to an ensemble.

**Q29. Do decision trees need feature scaling? How do they handle missing values?** No scaling is needed: a split depends only on the order of values, so any monotonic transform gives the same tree. For missing values, CART-style trees can use **surrogate splits** (backup questions on correlated features), and libraries such as XGBoost, LightGBM and recent scikit-learn learn which branch missing values should follow; otherwise impute first.

**Q30. When would you choose a tree over a linear model?** When the relationship is strongly non-linear or full of interactions, when features are mixed types with little preprocessing, or when stakeholders need a flowchart they can follow. Choose the linear model when the relationship is roughly linear, data is small, or you need smooth extrapolation.

## 2.6 Clustering

Clustering groups unlabeled points so that points in the same group are similar and points in different groups are not. The two methods every interviewer expects are k-means and hierarchical clustering; DBSCAN is a common follow-up.

### Intuition

**k-means** splits the data into a chosen number K of clusters, each represented by its **centroid** (mean point). **Lloyd's algorithm** alternates two steps until nothing changes:

1. **Assign** each point to its nearest centroid.
2. **Update** each centroid to the mean of the points assigned to it.

- Each step can only lower the total within-cluster distance, so the algorithm always converges, but to a **local** optimum that depends on the starting centroids. Run it several times from different starts and keep the best result. **k-means++** picks starting centroids spread far apart, which usually gives better and faster results.
- k-means assumes roughly **spherical clusters of similar size**, because it uses Euclidean distance to a mean. It struggles with elongated or nested shapes, very different densities, and outliers, which drag the mean.
- **Scale your features first.** A feature measured in thousands would dominate a feature measured in fractions.
- **Choosing K:** the **elbow method** plots the within-cluster sum of squares against K and looks for the bend; the **silhouette score** measures how much closer each point is to its own cluster than to the next one; often the business decides ("we can run five marketing campaigns").

**Hierarchical (agglomerative) clustering** needs no K up front. It starts with every point as its own cluster and repeatedly merges the two closest clusters until one remains. The merge history is drawn as a **dendrogram**, a tree whose merge heights show how far apart the clusters were; cutting it at a chosen height gives the clusters. "Closest clusters" depends on the **linkage**:

| Linkage  | Distance between clusters A and B                      | Behavior                                                                                                |
| -------- | ------------------------------------------------------ | ------------------------------------------------------------------------------------------------------- |
| Complete | Largest distance between a point in A and a point in B | Compact, balanced clusters; ISLR's usual choice                                                         |
| Average  | Mean of all pairwise distances                         | A middle ground, also popular                                                                           |
| Single   | Smallest pairwise distance                             | Long, straggly chains of points                                                                         |
| Centroid | Distance between the two means                         | Can give inversions, merges that happen below an earlier merge, which makes the dendrogram hard to read |
| Ward     | Increase in within-cluster sum of squares              | Similar to k-means; scikit-learn's default                                                              |

Hierarchical clustering stores all pairwise distances, so memory grows with n², which limits it to tens of thousands of points.

**DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** groups points that have enough neighbors within a radius ε, so it finds clusters of any shape, needs no K, and labels isolated points as noise. It struggles when clusters have very different densities.

### Core math

k-means minimizes the total **within-cluster sum of squares** (also called **inertia**), where μ_k is the centroid of cluster C_k:

$$
\min_{C_1,\dots,C_K}\ \sum_{k=1}^{K} \sum_{i \in C_k} \lVert x_i - \mu_k \rVert^2
$$

ISLR writes the same objective with pairwise distances, which shows it really is about how spread out each cluster is:

$$
W(C_k) = \frac{1}{|C_k|}\sum_{i,\,i' \in C_k} \lVert x_i - x_{i'} \rVert^2 = 2\sum_{i \in C_k} \lVert x_i - \mu_k \rVert^2
$$

The **silhouette** of point i compares a(i), its mean distance to points in its own cluster, with b(i), its mean distance to points in the nearest other cluster. It ranges from −1 (probably in the wrong cluster) to 1 (well clustered); the score of a clustering is the mean over all points:

$$
s(i) = \frac{b(i) - a(i)}{\max\{a(i),\, b(i)\}}
$$

One Lloyd iteration costs O(nKp) for n points, K clusters and p features.

### Code

k-means from scratch, then choosing K with inertia and silhouette on data generated with four clusters:

```python
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

X, _ = make_blobs(n_samples=600, centers=4, cluster_std=1.0, random_state=42)

def kmeans(X, k, iters=100, seed=0):
    rng = np.random.default_rng(seed)
    centers = X[rng.choice(len(X), k, replace=False)]           # random initial centroids
    for _ in range(iters):
        labels = np.argmin(((X[:, None, :] - centers[None]) ** 2).sum(-1), axis=1)  # assign
        new = np.array([X[labels == j].mean(0) for j in range(k)])                  # update
        if np.allclose(new, centers):
            break
        centers = new
    inertia = ((X - centers[labels]) ** 2).sum()
    return labels, centers, inertia

print(kmeans(X, 4)[2])   # 1160.3, identical to sklearn's KMeans

for k in range(2, 7):
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
    print(k, round(km.inertia_), round(silhouette_score(X, km.labels_), 3))
# k  inertia  silhouette
# 2  18638    0.591
# 3   4179    0.757
# 4   1160    0.788   <- sharp elbow and the highest silhouette
# 5   1046    0.691
# 6    938    0.559

labels_h = AgglomerativeClustering(n_clusters=4, linkage="ward").fit_predict(X)
```

Inertia always falls as K grows, so never pick the K with the lowest inertia; look for where the drop flattens. Here both methods agree on K = 4, the true number. Real data rarely gives such a clean answer.

### Interview Q&A

**Q31. Walk me through k-means. Does it always converge?** Pick K starting centroids, assign every point to its nearest centroid, recompute each centroid as the mean of its points, and repeat until assignments stop changing. It always converges because neither step can increase the objective and there are finitely many assignments, but only to a local optimum, so use k-means++ and several restarts.

**Q32. How do you choose K?** Plot inertia against K and look for the elbow, compare silhouette scores, check that clusters are stable across random seeds, and ask whether the clusters are useful to the business. Never minimize inertia alone: it reaches zero when K = n.

**Q33. When does k-means fail, and what would you use instead?** On non-spherical shapes (rings, moons), clusters of very different sizes or densities, heavy outliers, and categorical data. Alternatives: DBSCAN for arbitrary shapes and noise, Gaussian mixture models for elliptical clusters with soft assignments, k-modes or k-prototypes for categorical data, or spectral clustering.

**Q34. k-means or hierarchical clustering?** k-means is fast (linear in n per iteration) and scales to millions of points but needs K and assumes spherical clusters. Hierarchical clustering needs no K and gives a dendrogram that shows structure at every level, but costs O(n²) memory or more, so it suits smaller datasets.

**Q35. How do you evaluate a clustering without labels?** Internal metrics such as silhouette score or the Davies-Bouldin index; stability under resampling; and, most importantly, whether the clusters are interpretable and lead to different actions. If a few labels exist, compare against them with the adjusted Rand index.

**Q36. How does a Gaussian mixture model differ from k-means?** A **Gaussian mixture model (GMM)** assumes the data comes from K Gaussian distributions with their own means and covariances, fitted by the **expectation-maximization (EM)** algorithm. It gives each point a probability of belonging to each cluster (soft assignment) and handles elliptical clusters; k-means is the limiting case with hard assignments and equal, spherical clusters.

## 2.7 Principal component analysis

**Principal component analysis (PCA)** replaces many correlated features with a few new ones, the **principal components**, that keep as much of the data's variation as possible. It is the standard tool for visualizing, compressing and de-noising high-dimensional data.

### Intuition

- Picture a cloud of points. The **first principal component** is the direction along which the cloud is most spread out. The second is the direction of greatest remaining spread **at right angles** to the first, and so on. Rotating the axes to these directions and keeping the first few gives a smaller dataset that loses little information.
- Each component is a weighted sum of the original features; the weights are its **loadings**. A point's coordinates in the new axes are its **scores**.
- A second view, from ISLR: the first M components span the M-dimensional surface that is **closest** to the data, so projecting onto them gives the best rank-M approximation in squared error.
- **Standardize first** unless all features share a unit. Otherwise the feature with the largest numbers, say salary in rupees, dominates the first component purely because of its units.
- PCA is **unsupervised**: it never looks at y. The direction with the most variance is not necessarily the one that predicts the label best.
- PCA is **linear**. For curved structure, kernel PCA or autoencoders can help; t-SNE and UMAP make good 2D pictures but are for visualization, not as features.
- **Choosing how many components:** keep enough to explain a target share of variance (often 80 to 95%), look for an elbow in the **scree plot** of variance per component, or treat the number as a hyperparameter and pick it by cross-validation on the downstream task.

### Core math

Center (and usually scale) the n × p data matrix X. The first loading vector φ₁ is the unit vector that maximizes the variance of the projected data:

$$
\phi_1 = \arg\max_{\lVert \phi \rVert = 1}\ \operatorname{Var}(X\phi) = \arg\max_{\lVert \phi \rVert = 1}\ \phi^\top \Sigma\, \phi, \qquad \Sigma = \frac{1}{n-1} X^\top X
$$

The solution is the top **eigenvector** of the covariance matrix Σ, and the variance it captures is the matching **eigenvalue** λ₁. In general Σ = VΛVᵀ: the columns of V are the loadings and the eigenvalues λ₁ ≥ λ₂ ≥ … are the variances along each component. The **proportion of variance explained (PVE)** by component m is:

$$
\text{PVE}_m = \frac{\lambda_m}{\sum_{j=1}^{p} \lambda_j}
$$

In practice PCA is computed with the **singular value decomposition (SVD)** X = UDVᵀ, which is numerically stabler than forming Σ. The loadings are the columns of V, the scores are UD, and λ_j = d_j² / (n − 1). Keeping the first M components gives the best rank-M approximation of X:

$$
X \approx Z_M V_M^\top, \qquad Z_M = X V_M
$$

### Code

PCA on the 30 standardized breast-cancer features, first with scikit-learn and then with a bare SVD:

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

X, _ = load_breast_cancer(return_X_y=True)
X_std = StandardScaler().fit_transform(X)

pca = PCA().fit(X_std)
pve = pca.explained_variance_ratio_
print(pve[:2].round(3))                          # [0.443 0.19 ]: two components keep 63%
print(np.argmax(np.cumsum(pve) >= 0.90) + 1)     # 7 components keep 90% of the variance

# The same thing from the SVD
Xc = X_std - X_std.mean(axis=0)
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
pve_svd = S**2 / np.sum(S**2)                    # equals pca.explained_variance_ratio_
scores_2d = Xc @ Vt[:2].T                        # coordinates for a 2D scatter plot
```

Seven of 30 dimensions hold 90% of the variance, because many of the measurements (radius, perimeter, area) are strongly correlated.

### Interview Q&A

**Q37. Explain PCA to a non-technical stakeholder, then to an engineer.** Stakeholder: it finds the few combined measurements that capture most of the differences between records, so we can summarize 30 numbers with 5. Engineer: it rotates the centered data onto the eigenvectors of its covariance matrix, ordered by eigenvalue, and keeps the top few; in practice computed with an SVD.

**Q38. Why standardize before PCA?** PCA maximizes variance, and variance depends on units. Without scaling, the feature with the largest numeric range dominates the first component regardless of how informative it is.

**Q39. How do you decide how many components to keep?** Cumulative explained variance against a target (say 90%), the elbow of the scree plot, or, when PCA feeds a supervised model, cross-validation on the downstream metric.

**Q40. Can PCA hurt a supervised model?** Yes. It keeps high-variance directions, not predictive ones, so a small-variance direction that separates the classes can be thrown away. Also fit PCA on the training data only; fitting it on all data leaks test information into the features.

**Q41. How are PCA, eigenvectors and the SVD related?** The loadings are the eigenvectors of the covariance matrix, and the eigenvalues are the variances along them. The SVD of the centered data X = UDVᵀ gives the same loadings in V, with eigenvalues d_j² / (n − 1), without ever forming XᵀX.

**Q42. PCA versus feature selection?** Feature selection keeps a subset of the original, interpretable features. PCA builds new features that mix all the originals, which compresses better but makes each component harder to explain.

## 2.8 Other models to recognize

Three more classifiers from ISLR come up as quick questions. You rarely need to derive them, but you should know the idea, the assumption and when each one fits.

| Model                                  | Idea                                                                             | Key assumption                                     | Good for                                               | Weak spot                                                          |
| -------------------------------------- | -------------------------------------------------------------------------------- | -------------------------------------------------- | ------------------------------------------------------ | ------------------------------------------------------------------ |
| **k-nearest neighbours (kNN)**         | Predict the majority class (or mean) of the K closest training points            | Nearby points have similar labels                  | Small, low-dimensional data; a non-parametric baseline | Slow at prediction time; distances lose meaning in high dimensions |
| **Naive Bayes**                        | Apply Bayes' rule, treating features as independent given the class              | Conditional independence of features               | Text classification, very fast training, tiny data     | Probabilities are poorly calibrated when features are correlated   |
| **Linear discriminant analysis (LDA)** | Model each class as a Gaussian with a shared covariance; classify by Bayes' rule | Gaussian features, equal covariance across classes | Well-separated classes, small n, multiclass problems   | Breaks down when the Gaussian assumption is far off                |

All three are built on **Bayes' rule**: the probability of class k given x is proportional to the class's prior share π_k times the density of x within that class, f_k(x). Naive Bayes makes f_k a product of one-feature densities; LDA makes it a Gaussian. **Quadratic discriminant analysis (QDA)** lets each class have its own covariance, which bends the boundary into a curve.

$$
P(y = k \mid x) = \frac{\pi_k\, f_k(x)}{\sum_{l=1}^{K} \pi_l\, f_l(x)}, \qquad \text{naive Bayes: } f_k(x) = \prod_{j=1}^{p} f_{kj}(x_j)
$$

Do not confuse this LDA with **latent Dirichlet allocation**, a topic model for text that shares the acronym.

**Q43. Why is naive Bayes called naive, and why does it work anyway?** It assumes features are independent within each class, which is rarely true (words in an email are correlated). It still classifies well because ranking the classes correctly only needs the right ordering of probabilities, not accurate values, and the simplification cuts the parameters needed so much that it learns from very little data.

**Q44. How do you choose K in kNN, and why does kNN struggle in high dimensions?** Small K gives a jagged, low-bias, high-variance boundary; large K gives a smooth, high-bias one; choose K by cross-validation. In high dimensions, the **curse of dimensionality** makes all points roughly equally far apart, so "nearest" stops being meaningful unless you reduce dimensions first.

**Q45. LDA or logistic regression?** Both give linear boundaries. LDA is more stable when classes are well separated or n is small, provided the Gaussian assumption roughly holds; logistic regression makes fewer assumptions and is safer otherwise.

## 2.9 In system design

In design rounds, these models appear as **baselines**, as **cheap production models**, and as **building blocks** inside larger systems. Proposing one first shows judgment: it tells you how much a complex model actually buys.

- **Logistic regression** is the classic baseline for click-through-rate prediction and risk scoring. It trains on billions of sparse features with feature hashing, serves in microseconds, gives calibrated probabilities that pricing and ranking need, and is explainable to regulators in credit decisions.
- **Linear regression** is the baseline for forecasting and pricing, and it is the workhorse for analyzing experiments, where regression adjustment on pre-experiment data reduces the variance of A/B test results (Chapter 14).
- **Decision trees** become explainable rule sets for policies and fallbacks, and they are the base learners of the gradient-boosted models that dominate tabular ML (Chapter 3).
- **SVMs** still fit small, high-dimensional problems such as text classification on bag-of-words features, but they rarely appear in large-scale systems.
- **k-means** is everywhere inside retrieval: vector databases group embeddings into clusters (an **inverted file index, IVF**) and search only the nearest few clusters, and **product quantization** compresses vectors with k-means codebooks. It also drives customer segmentation, near-duplicate grouping and cold-start cohorts.
- **PCA** compresses embeddings (for example 768 to 128 dimensions) to cut storage and latency, decorrelates features, and gives a 2D view to spot drift between training and production data.

A useful opening line in any design round: "I'd start with a logistic-regression baseline on a handful of strong features, measure it offline, and only add complexity where the error analysis says it pays off."

## 2.10 Cheat sheet and pitfalls

| Model               | Type                                   | Fits by                                        | Boundary or shape                  | Scale features?                               | One-line interview answer                       |
| ------------------- | -------------------------------------- | ---------------------------------------------- | ---------------------------------- | --------------------------------------------- | ----------------------------------------------- |
| Linear regression   | Supervised, regression                 | OLS, closed form (XᵀX)⁻¹Xᵀy                    | Hyperplane                         | Only if regularized or using gradient descent | β_j = change in y per unit of x_j, others fixed |
| Logistic regression | Supervised, classification             | MLE, minimizes log loss; convex                | Linear                             | Yes, if regularized                           | Linear log-odds; e^β is the odds ratio          |
| SVM                 | Supervised, classification             | Hinge loss plus margin; kernels for curves     | Maximum margin, linear or kernel   | Yes, always                                   | Only support vectors define the boundary        |
| Decision tree       | Supervised, both                       | Greedy splits on Gini, entropy or RSS; pruning | Axis-aligned boxes                 | No                                            | Explainable but high variance                   |
| k-means             | Unsupervised, clustering               | Lloyd's algorithm; local optimum               | Spherical clusters around means    | Yes                                           | Assign to nearest centroid, recompute, repeat   |
| Hierarchical        | Unsupervised, clustering               | Repeated merges by linkage                     | Dendrogram, cut at any height      | Yes                                           | No K needed; O(n²) memory                       |
| PCA                 | Unsupervised, dimensionality reduction | Eigenvectors of covariance, via SVD            | Orthogonal max-variance directions | Yes                                           | Keeps variance, not necessarily signal          |

**Remember:**

- Regression predicts numbers, classification predicts categories, clustering groups without labels, PCA compresses without labels.
- Choose models by validation or test error; training error always falls with flexibility.
- Expected error = reducible error + irreducible noise; more flexibility lowers bias and raises variance.
- OLS: normal equations XᵀXβ̂ = Xᵀy; R² = 1 − RSS/TSS; VIF = 1/(1 − R_j²).
- Logistic regression: gradient Xᵀ(p − y); coefficient → odds ratio e^β; separable data needs regularization.
- SVM: margin 2/‖w‖; scikit-learn's C is a penalty, ISLR's C a budget; RBF γ controls locality.
- Trees: Gini = 1 − Σp², entropy = −Σp log p; prune with α|T| chosen by cross-validation.
- k-means: minimizes within-cluster sum of squares; k-means++ and restarts; pick K by elbow and silhouette.
- PCA: standardize, PVE_m = λ_m / Σλ_j, compute with SVD, fit on training data only.

**Pitfalls that cost candidates:**

- Interpreting a regression coefficient without saying "holding other features fixed".
- Saying logistic regression is non-linear, or that it outputs class labels rather than probabilities.
- Using MSE as the loss for logistic regression.
- Forgetting to scale features before SVMs, k-means, kNN or PCA.
- Mixing up the two meanings of C in SVMs.
- Choosing K for k-means by minimizing inertia.
- Treating a large first principal component as "the most important feature for prediction".
- Fitting scalers or PCA on the full dataset before splitting, which leaks test data.
- Jumping to deep learning in a design round without proposing a simple baseline.

## 2.11 Further reading

<ol start="7">
  <li><a href="https://machinelearningmastery.com/a-tour-of-machine-learning-algorithms/">A tour of machine learning algorithms</a> (Machine Learning Mastery)</li>
</ol>

**Supplementary:** Gareth James, Daniela Witten, Trevor Hastie and Robert Tibshirani, [An Introduction to Statistical Learning](https://www.statlearning.com/) (ISLR): Chapter 2 (statistical learning, bias-variance), Chapter 3 (linear regression), Chapter 4 (logistic regression, LDA, QDA, naive Bayes, kNN), Chapter 8 (tree-based methods), Chapter 9 (support vector machines) and Chapter 12 (unsupervised learning: PCA and clustering). Chapter numbers follow the second edition and the 2023 Python edition.
