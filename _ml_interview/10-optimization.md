---
title: "Optimization"
chapter: 10
permalink: "/blog/cracking-the-ml-interview/optimization/"
updated: 2026-11-01
description: "Training a model means searching for the weights that minimize the loss."
---

Training a model means searching for the weights that minimize the loss. Almost all of deep learning, and much of classical ML, does that search with variants of gradient descent. Interviewers expect you to write the update rules from memory, explain what each variant fixes, and know why Adam is the default for transformers while SGD with momentum still wins in some settings.

## 10.1 Intuition

**Gradient descent.** Picture standing on a hilly landscape in fog, where height is the loss. The gradient points uphill, so take a step downhill, proportional to the slope, and repeat. The **learning rate** η is the step size: too small and progress is painfully slow; too large and you overshoot the valley and bounce out of it, or diverge.

**Batch, stochastic and mini-batch** (reference 24):

| Variant                               | Gradient computed on                     | Pros                                                    | Cons                                                       |
| ------------------------------------- | ---------------------------------------- | ------------------------------------------------------- | ---------------------------------------------------------- |
| Batch gradient descent                | The whole training set                   | Exact gradient, smooth progress                         | One step per pass over the data; impossible for large data |
| **SGD (stochastic gradient descent)** | One example                              | Very cheap steps; noise can escape shallow local minima | Very noisy; poor use of parallel hardware                  |
| Mini-batch SGD                        | A batch of 32 to a few thousand examples | Good hardware use; noise reduced but still helpful      | Batch size becomes a hyperparameter                        |

In practice, "SGD" almost always means mini-batch SGD. A pass over the whole training set is an **epoch**; one update is a **step** or **iteration**.

**The problem with plain SGD: ravines and scale.** In many loss landscapes the surface is steep in some directions and nearly flat in others, a long narrow **ravine**. A step size small enough to avoid bouncing across the steep walls is too small to make progress along the flat floor. The ratio between the steepest and flattest curvature, the **condition number**, sets how slow plain gradient descent is. Badly scaled features are a common cause, and so is the structure of deep networks.

**Momentum** (reference 26) keeps a running average of past gradients, a **velocity**, and steps along it. Like a heavy ball rolling downhill, it builds speed along the consistent direction of the ravine floor while the back-and-forth components across the walls cancel out. **Nesterov momentum** looks ahead: it computes the gradient at the point the velocity is about to carry it to, which corrects overshooting sooner.

**Adaptive learning rates** give every parameter its own step size, based on the history of its gradients.

- **AdaGrad** (Adaptive Gradient; reference 25) divides each parameter's step by the square root of the **sum of all its past squared gradients**. Rarely updated parameters, such as embeddings of rare words or sparse features, keep large steps; frequently updated ones get small steps. Its weakness: the sum only grows, so the effective learning rate decays toward zero and training can stall.
- **RMSProp** (Root Mean Square Propagation; reference 27) fixes this by using an **exponential moving average** of squared gradients instead of the full sum, so the scaling tracks recent behaviour and the step size does not vanish.
- **Adam** (Adaptive Moment Estimation) combines momentum (a moving average of gradients, the **first moment**) with RMSProp's scaling (a moving average of squared gradients, the **second moment**), plus a **bias correction** for the averages' start at zero. It is robust to the learning rate and the default for most deep learning.
- **AdamW** is Adam with decoupled weight decay (Chapter 9) and the standard for training transformers.

| Optimizer      | Remembers                                   | Per-parameter step size? | Strengths                                                  | Weaknesses                                                                                                    |
| -------------- | ------------------------------------------- | ------------------------ | ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| SGD            | Nothing                                     | No                       | Simple; often generalizes well                             | Slow in ravines; sensitive to learning rate and feature scale                                                 |
| SGD + momentum | Velocity (average gradient)                 | No                       | Fast along consistent directions; strong for vision models | Extra hyperparameter β; still scale-sensitive                                                                 |
| Nesterov       | Velocity, with look-ahead                   | No                       | Less overshooting than momentum                            | Marginally more complex                                                                                       |
| AdaGrad        | Sum of squared gradients                    | Yes                      | Excellent for sparse features                              | Step size decays to zero                                                                                      |
| RMSProp        | Moving average of squared gradients         | Yes                      | Steady adaptive steps; good for RNNs and RL                | No momentum in its basic form                                                                                 |
| Adam / AdamW   | Averages of gradients and squared gradients | Yes                      | Robust default; fast progress; little tuning               | More memory (two extra numbers per weight); can generalize slightly worse than tuned SGD on some vision tasks |

**Learning-rate schedules** matter as much as the optimizer:

- **Warm-up:** start with a small learning rate and increase it over the first few hundred or thousand steps. Early gradients and Adam's moment estimates are unreliable, and warm-up prevents early divergence, especially in transformers and large-batch training.
- **Decay:** lower the learning rate later so the weights settle into a minimum instead of bouncing around it. Common shapes: **step decay** (divide by 10 at fixed epochs), **cosine decay** (a smooth half-cosine down to near zero), linear decay, and inverse square root.
- **Cyclical and one-cycle schedules** raise and lower the rate during training; the **learning-rate range test** (increase the rate each step until the loss explodes) finds a good maximum quickly.

**Other tools in every training loop:**

- **Gradient clipping** caps the gradient's norm so one bad batch cannot blow up the weights. It is standard for RNNs and transformers.
- **Second-order methods** such as Newton's method and **L-BFGS (limited-memory Broyden–Fletcher–Goldfarb–Shanno)** use curvature to take better steps. They converge in few iterations on small, smooth, full-batch problems (scikit-learn's logistic regression uses L-BFGS by default), but storing or estimating curvature is too costly for networks with billions of weights.
- **Saddle points** are flat regions that slope up in some directions and down in others. In high dimensions they are far more common than bad local minima; gradient noise and momentum help escape them.

## 10.2 Core math

Notation: weights θ, learning rate η, gradient at step t g_t = ∇L(θ_t), and element-wise operations for all squares and square roots (reference 23 derives each rule).

**SGD:**

$$
\theta_{t+1} = \theta_t - \eta\, g_t
$$

**Momentum** (β typically 0.9) and **Nesterov momentum**, which evaluates the gradient at the look-ahead point:

$$
v_{t} = \beta\, v_{t-1} + g_t, \quad \theta_{t+1} = \theta_t - \eta\, v_t \qquad\qquad \text{Nesterov: } v_t = \beta\, v_{t-1} + \nabla L\big(\theta_t - \eta\beta\, v_{t-1}\big)
$$

Unrolling the recursion, v_t is a weighted sum of all past gradients with weights 1, β, β², …, so along a direction where gradients agree, the step grows by up to 1/(1 − β) = 10 times for β = 0.9.

**AdaGrad**, with G_t the running sum of squared gradients and a small ε (around 10⁻⁸) to avoid division by zero:

$$
G_t = G_{t-1} + g_t^2, \qquad \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t} + \epsilon}\, g_t
$$

**RMSProp** (ρ typically 0.9) replaces the sum with an exponential moving average:

$$
E[g^2]_t = \rho\, E[g^2]_{t-1} + (1 - \rho)\, g_t^2, \qquad \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{E[g^2]_t} + \epsilon}\, g_t
$$

**Adam** (defaults β₁ = 0.9, β₂ = 0.999, ε = 10⁻⁸) tracks both moments and corrects their bias:

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)\, g_t, \quad v_t = \beta_2 v_{t-1} + (1-\beta_2)\, g_t^2, \quad \hat m_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat v_t = \frac{v_t}{1 - \beta_2^t}, \quad \theta_{t+1} = \theta_t - \frac{\eta\, \hat m_t}{\sqrt{\hat v_t} + \epsilon}
$$

**Why bias correction?** m and v start at zero, so early averages are pulled toward zero: after one step, m_1 = (1 − β₁)g_1 = 0.1 g_1. Dividing by 1 − β₁^t (here 0.1) restores the right scale. Without it, the second-moment estimate (with 1 − β₂ = 0.001) would be tiny early on, and the first steps would be far too large.

**Why adaptive methods help with badly scaled features.** Dividing by √v gives every parameter a step of roughly η in its own units, regardless of how large its gradients are. This is a cheap diagonal **preconditioner**: it approximates rescaling each feature, which is exactly what standardizing the inputs does for plain SGD.

**Convergence and conditioning.** For a quadratic loss whose curvature (Hessian eigenvalues) ranges from μ to L, with **condition number** κ = L/μ, gradient descent with the best fixed step η = 2/(L + μ) shrinks the error each step by:

$$
\text{GD: } \frac{\kappa - 1}{\kappa + 1}, \qquad \text{heavy-ball momentum (optimally tuned): } \frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1}
$$

With κ = 100 that is 0.980 per step for gradient descent and 0.818 with momentum, so the number of steps grows like κ without momentum but only like √κ with it. The step size must also stay below 2/L, or the steepest direction diverges. On a rotated quadratic with κ = 100, starting from (3, −2), gradient descent needed 461 steps to cut the loss by a factor of 10⁸; optimally tuned momentum needed 71.

**SGD noise and batch size.** A mini-batch gradient is an unbiased but noisy estimate of the full gradient; its variance falls like 1/B for batch size B. So for SGD to converge exactly, the learning rate must eventually decay (the classic conditions are Ση_t = ∞ and Ση_t² < ∞, satisfied by η_t ∝ 1/t). In practice a schedule that decays to near zero does the job. Larger batches allow larger learning rates (Chapter 7's linear scaling rule).

**Cosine decay with linear warm-up**, the most common schedule for large models, with peak rate η_max, warm-up length T_w and total steps T:

$$
\eta_t = \begin{cases} \eta_{\max}\, \dfrac{t}{T_w} & t < T_w \\[6pt] \eta_{\min} + \tfrac{1}{2}(\eta_{\max} - \eta_{\min})\Big(1 + \cos\dfrac{\pi\,(t - T_w)}{T - T_w}\Big) & t \ge T_w \end{cases}
$$

**Gradient clipping by norm** rescales the whole gradient when its norm exceeds a threshold c (often 1.0), keeping its direction:

$$
g \leftarrow g \cdot \min\Big(1, \frac{c}{\|g\|}\Big)
$$

## 10.3 Code

**Five optimizers from scratch** on one problem: logistic regression on 5,000 rows whose 20 features are scaled from 1 to 100, so the loss surface is a badly conditioned ravine. Each optimizer gets the best learning rate from a grid of 13 values, mini-batches of 32 and 30 epochs. A sixth run uses plain SGD on the same data with standardized features.

```python
import numpy as np

rng = np.random.default_rng(0)
n, d = 5000, 20
Z = rng.normal(size=(n, d))                            # well-scaled features
w_true = rng.normal(size=d)
y = (rng.random(n) < 1 / (1 + np.exp(-(Z @ w_true)))).astype(float)
X_bad = Z / np.logspace(0, 2, d)                       # same data, features scaled from 1 to 100

def add_bias(X): return np.c_[X, np.ones(len(X))]

def loss(w, Xb):
    z = Xb @ w
    return np.mean(np.logaddexp(0, z) - y * z)          # stable binary cross-entropy

def sgd(w, g, lr, s, t):
    return w - lr * g

def momentum(w, g, lr, s, t, beta=0.9):
    s["v"] = beta * s.get("v", 0) + g
    return w - lr * s["v"]

def adagrad(w, g, lr, s, t, eps=1e-8):
    s["G"] = s.get("G", 0) + g ** 2                     # sum of squared gradients
    return w - lr * g / (np.sqrt(s["G"]) + eps)

def rmsprop(w, g, lr, s, t, rho=0.9, eps=1e-8):
    s["E"] = rho * s.get("E", 0) + (1 - rho) * g ** 2   # moving average of squared gradients
    return w - lr * g / (np.sqrt(s["E"]) + eps)

def adam(w, g, lr, s, t, b1=0.9, b2=0.999, eps=1e-8):
    s["m"] = b1 * s.get("m", 0) + (1 - b1) * g
    s["v"] = b2 * s.get("v", 0) + (1 - b2) * g ** 2
    m_hat, v_hat = s["m"] / (1 - b1 ** t), s["v"] / (1 - b2 ** t)   # bias correction
    return w - lr * m_hat / (np.sqrt(v_hat) + eps)

def train(step, lr, X, epochs=30, batch=32, seed=0):
    Xb, r = add_bias(X), np.random.default_rng(seed)
    w, state, t = np.zeros(Xb.shape[1]), {}, 0
    history = [loss(w, Xb)]
    for _ in range(epochs):
        for b in np.array_split(r.permutation(n), n // batch):
            t += 1
            p = 1 / (1 + np.exp(-(Xb[b] @ w)))
            w = step(w, Xb[b].T @ (p - y[b]) / len(b), lr, state, t)
            if not np.all(np.isfinite(w)):
                return None                            # diverged at this learning rate
        history.append(loss(w, Xb))
    return np.array(history)

grid = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1, 3, 10, 30, 100]
runs = [("SGD", sgd, X_bad), ("Momentum", momentum, X_bad), ("AdaGrad", adagrad, X_bad),
        ("RMSProp", rmsprop, X_bad), ("Adam", adam, X_bad), ("SGD, standardized", sgd, Z)]
for name, step, X in runs:
    results = [(lr, train(step, lr, X)) for lr in grid]
    lr, h = min(((lr, h) for lr, h in results if h is not None), key=lambda r: r[1][-1])
    print(name, lr, h[1], h[5], h[-1])
```

| Run                        | Best learning rate | Loss after 1 epoch | After 5 | After 30 (optimum 0.2774) |
| -------------------------- | ------------------ | ------------------ | ------- | ------------------------- |
| SGD                        | 3                  | 0.445              | 0.406   | 0.324                     |
| Momentum                   | 1                  | 0.470              | 0.397   | 0.299                     |
| AdaGrad                    | 3                  | 0.292              | 0.286   | 0.279                     |
| RMSProp                    | 0.03               | 0.435              | 0.328   | 0.279                     |
| Adam                       | 0.03               | 0.449              | 0.335   | 0.280                     |
| SGD, standardized features | 0.1                | 0.327              | 0.281   | 0.278                     |

<figure class="figure-pending"><div class="figure-pending__box">diagram coming soon</div><figcaption>experiment in 10.3 · 6 runs, 30 epochs each</figcaption></figure>

**What the experiment teaches:**

1. **Feature scale, not the algorithm's sophistication, was the real problem.** On badly scaled features, plain SGD is still 0.046 above the optimum after 30 epochs; momentum halves that. The three adaptive methods get within 0.003, because their per-parameter scaling undoes the bad scales.
2. **Standardizing the features made plain SGD the best of all** (0.0002 from the optimum). Adaptive optimizers are partly an automatic fix for poor conditioning; when you can fix the conditioning directly, simple methods do very well.
3. **The best learning rates differ by 100 times** across methods (0.03 to 3), so never compare optimizers at one shared learning rate.
4. On convex problems like this one, all methods reach the same optimum eventually; in deep networks, the optimizer also affects _which_ minimum you reach, which is why tuned SGD with momentum sometimes generalizes better than Adam on image classification.

**In PyTorch** the same optimizers are one line each; a typical transformer setup pairs AdamW with warm-up and cosine decay, plus clipping:

```python
import torch

model = torch.nn.Linear(20, 1)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.999), weight_decay=0.1)
warmup, total = 100, 1_000
sched = torch.optim.lr_scheduler.SequentialLR(
    opt,
    [torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, total_iters=warmup),
     torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total - warmup)],
    milestones=[warmup])

for step in range(total):
    x, target = torch.randn(32, 20), torch.randn(32, 1)
    loss = torch.nn.functional.mse_loss(model(x), target)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)   # gradient clipping
    opt.step()
    sched.step()   # learning rate: 3e-6 at the start, 3e-4 after warm-up, 0 at the end
```

## 10.4 In system design

- **Default recipes.** Transformers and most new deep models: AdamW, linear warm-up, cosine or linear decay, gradient clipping at norm 1.0. Image classification with convolutional networks: SGD with momentum 0.9 and a step or cosine schedule remains competitive and sometimes generalizes better. Tabular data: gradient-boosted trees, which have their own learning rate (shrinkage) and early stopping (Chapter 3). Small convex problems: L-BFGS or a closed form.
- **Sparse features and embeddings.** Recommenders with huge embedding tables often use AdaGrad or sparse variants of Adam for embeddings, because rarely seen IDs need large updates when they finally appear, and updating only the rows touched by a batch saves enormous compute.
- **Optimizer memory counts.** Adam stores two extra numbers per weight, which is 8 of the 16 bytes per parameter in mixed-precision training (Chapter 7). Memory-saving optimizers such as Adafactor (factored second moments) and 8-bit Adam exist for very large models.
- **Tune the learning rate first.** It is usually the single most important hyperparameter; search it on a log scale, retune it when the batch size changes, and watch the training curves: a loss that explodes means too high, one that barely moves means too low.
- **Monitor training health:** loss curves, gradient norms, the learning-rate schedule, and the fraction of steps where clipping fired. Spikes in gradient norm often precede divergence and point to bad data batches or a learning rate that is too high.
- **Fine-tuning** a pretrained model typically uses a learning rate 10 to 100 times smaller than pretraining, a short warm-up, and sometimes lower rates for earlier layers (**discriminative learning rates**), so useful pretrained features are not destroyed.
- **Online learning** systems that update continuously from a stream (ads, feeds) use SGD-style updates per batch of fresh data; FTRL (Follow The Regularized Leader) with L1 is a classic choice for sparse logistic regression at scale.

## 10.5 Interview Q&A

**Q1. What is the difference between batch, stochastic and mini-batch gradient descent?** Batch gradient descent uses the whole dataset per step: an exact gradient but slow, memory-hungry steps. Stochastic gradient descent uses one example: cheap, very noisy steps. Mini-batch uses a small batch, balancing noise, speed and hardware efficiency, and is what "SGD" means in practice.

**Q2. Why does momentum help?** It accumulates a velocity from past gradients, so steps grow along directions where gradients consistently agree (up to 1/(1 − β) times) and oscillating components cancel out. In narrow ravines this speeds progress along the floor and damps bouncing between the walls; theoretically, steps needed grow with √κ instead of κ.

**Q3. Explain AdaGrad, and why RMSProp was needed.** AdaGrad divides each parameter's step by the root of its accumulated squared gradients, giving rarely updated parameters larger steps, which is ideal for sparse features. Because the accumulated sum only grows, the effective learning rate shrinks toward zero and training can stall. RMSProp replaces the sum with an exponential moving average, so step sizes stay responsive.

**Q4. Walk through the Adam update. Why the bias correction?** Adam keeps moving averages of the gradient (m) and squared gradient (v), divides the bias-corrected m by the square root of the bias-corrected v, and steps with that. Both averages start at zero and are biased toward zero early on; dividing by 1 − βᵗ corrects that, which matters most for v because β₂ = 0.999 makes it warm up slowly.

**Q5. Adam or SGD: which would you use?** Adam (AdamW) for transformers, NLP, sparse or badly scaled problems, and when you want robust results with little tuning. SGD with momentum when you can tune carefully and generalization matters most, as in many image-classification benchmarks, and when memory is tight. Whatever the choice, tune the learning rate and schedule.

**Q6. What happens if the learning rate is too high or too low?** Too high: the loss oscillates, jumps up, or diverges to NaN, because steps overshoot the minimum. Too low: the loss falls very slowly and training may stall on plateaus. A learning-rate range test or a coarse log-scale search finds a good value quickly.

**Q7. Why use learning-rate warm-up?** At the start, weights are random, gradients can be large and erratic, and Adam's second-moment estimates are unreliable, so a full-size step can push the model into a bad region. Warm-up ramps the rate up over the first steps; it is especially important for transformers and large-batch training.

**Q8. What is gradient clipping, and when do you need it?** It rescales the gradient when its norm exceeds a threshold, keeping its direction. It prevents single bad batches or exploding gradients, common in RNNs and deep transformers, from wrecking the weights. Clipping by norm is preferred to clipping each value, which distorts the direction.

**Q9. Why don't we use Newton's method for deep learning?** Newton's method needs the Hessian, a p × p matrix: for a billion-parameter model that is 10¹⁸ entries, impossible to store or invert. Mini-batch noise also makes curvature estimates unreliable. Adaptive methods are cheap diagonal approximations of this idea; L-BFGS works well for small, smooth, full-batch problems.

**Q10. Your training loss suddenly spikes to NaN after 10,000 steps. How do you debug it?** Check the batch that caused it for bad data (NaN or extreme values), look at gradient norms before the spike, and confirm the learning-rate schedule. Common fixes: lower the peak learning rate or lengthen warm-up, add or tighten gradient clipping, use loss scaling or bf16 in mixed precision, and add ε or log-sum-exp stabilization wherever divisions and logs occur. Resume from the last good checkpoint.

## 10.6 Cheat sheet and pitfalls

**Remember:**

- SGD: θ ← θ − ηg.
- Momentum: v ← βv + g, θ ← θ − ηv; β = 0.9; steps scale like √κ instead of κ.
- AdaGrad: divide by √(sum of g²); great for sparse features; learning rate decays to zero.
- RMSProp: divide by √(moving average of g²); ρ = 0.9.
- Adam: momentum + RMSProp + bias correction; β₁ = 0.9, β₂ = 0.999, ε = 10⁻⁸. AdamW decouples weight decay.
- Stable step size for GD on a quadratic: η < 2/L.
- Schedules: warm-up, then cosine or step decay; clip gradients at norm about 1.0.
- Adaptive methods are diagonal preconditioners; standardizing features is the manual version.

**Pitfalls that cost candidates:**

- Saying Adam "doesn't need a learning rate" or always beats SGD.
- Comparing optimizers at one shared learning rate.
- Forgetting bias correction when writing Adam from memory, or putting the square root in the wrong place.
- Using L2 regularization inside Adam and calling it weight decay.
- Skipping feature standardization and blaming the optimizer.
- Increasing the batch size without retuning the learning rate.

## 10.7 Further reading

<ol start="23">
  <li>Sebastian Ruder, <a href="https://ruder.io/optimizing-gradient-descent/">An overview of gradient descent optimization algorithms</a></li>
  <li><a href="https://en.wikipedia.org/wiki/Stochastic_gradient_descent">Stochastic gradient descent</a> (Wikipedia)</li>
  <li><a href="https://optimization.cbe.cornell.edu/index.php?title=AdaGrad">AdaGrad</a> (Cornell University Computational Optimization Open Textbook)</li>
  <li><a href="https://optimization.cbe.cornell.edu/index.php?title=Momentum">Momentum</a> (Cornell University Computational Optimization Open Textbook)</li>
  <li><a href="https://optimization.cbe.cornell.edu/index.php?title=RMSProp">RMSProp</a> (Cornell University Computational Optimization Open Textbook)</li>
</ol>

**Supplementary:** Kingma and Ba, [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980).
