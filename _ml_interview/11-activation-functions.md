---
title: "Activation Functions"
chapter: 11
permalink: "/blog/cracking-the-ml-interview/activation-functions/"
updated: 2026-11-01
description: "An activation function is the small non-linear function applied to each neuron's output."
---

An activation function is the small non-linear function applied to each neuron's output. It looks like a detail, but it decides whether a deep network can learn at all: the wrong choice makes gradients vanish, blow up, or die. Interviewers ask about activations to test whether you understand backpropagation, not just layer APIs.

## 11.1 Intuition

**Why non-linearity?** A neuron computes z = wᵀx + b and passes it through an activation f. Without f, a stack of layers is a product of matrices, which is just one matrix: a 100-layer linear network can represent nothing a single linear layer cannot. The non-linearity lets each layer bend the space, so the network can represent curved decision boundaries and, with enough units, approximate any continuous function (the **universal approximation theorem**).

**What makes a good hidden-layer activation?**

- **Non-saturating gradients.** During backpropagation, each layer multiplies the gradient by f′(z). If f′ is often tiny, gradients shrink layer by layer and early layers stop learning (**vanishing gradients**).
- **Cheap to compute**, because it runs on every unit of every layer for every example.
- **Zero-centred outputs** help the next layer's updates; outputs that are always positive make all of a neuron's weight gradients share one sign, which zig-zags optimization.
- **Avoiding dead units**: if f′ is zero over a wide range, a unit stuck there never recovers.

**The main activations** (references 28 to 31):

- **Sigmoid**, σ(z) = 1/(1 + e^−z), squashes any input to (0, 1). It was the classic choice but has two problems in hidden layers: its slope is at most 0.25 and nearly zero for |z| beyond about 4 (**saturation**), and its outputs are not zero-centred. Today it is used for **output layers** that need a probability (binary classification, multi-label) and inside **gates** of LSTMs and GRUs.
- **Tanh** squashes to (−1, 1). It is a rescaled sigmoid, tanh(z) = 2σ(2z) − 1, zero-centred with a maximum slope of 1, so it trains better than sigmoid, but it still saturates. Still common in RNN cell updates.
- **ReLU (rectified linear unit)**, max(0, z), passes positive inputs unchanged and zeroes negative ones. Its slope is exactly 1 for every positive input, so gradients pass through many layers without shrinking, and it is extremely cheap. Its outputs are sparse (many zeros). Its weakness is the **dying ReLU** problem: a unit whose input is negative for every example has zero gradient and never learns again, often after a large update pushes its bias far negative.
- **Leaky ReLU** uses a small slope (say 0.01) for negative inputs, and **PReLU (parametric ReLU)** learns that slope, so units can always recover.
- **ELU (exponential linear unit)** (reference 28) is z for positive inputs and α(e^z − 1) for negative ones. It is smooth, its negative outputs push mean activations toward zero, and it saturates gently at −α for very negative inputs. It costs an exponential. **SELU** is a scaled ELU that, with the right initialization, keeps activations normalized in deep fully connected networks.
- **GELU (Gaussian error linear unit)** multiplies the input by the probability that a standard Gaussian is below it, z·Φ(z). It is a smooth ReLU that lets small negative values through; it is the default in transformers such as BERT and GPT.
- **SiLU / Swish**, z·σ(z), is similar in shape to GELU and is used in many vision models and, inside **SwiGLU** gated units, in recent large language models such as LLaMA.
- **Softmax** turns a vector of scores into a probability distribution and is used on the **output** of multi-class classifiers and inside attention, not as a hidden activation.

<figure class="figure-pending"><div class="figure-pending__box">diagram coming soon</div><figcaption>computed from the activation formulas · z from −3 to 3</figcaption></figure>

| Activation         | Range                      | Max slope  | Zero-centred | Saturates                  | Main use today                                     |
| ------------------ | -------------------------- | ---------- | ------------ | -------------------------- | -------------------------------------------------- |
| Sigmoid            | (0, 1)                     | 0.25       | No           | Both sides                 | Binary and multi-label outputs; LSTM/GRU gates     |
| Tanh               | (−1, 1)                    | 1          | Yes          | Both sides                 | RNN cell states; some normalizing outputs          |
| ReLU               | \[0, ∞)                    | 1          | No           | Negative side (zero slope) | Default for CNNs and MLPs                          |
| Leaky ReLU / PReLU | (−∞, ∞)                    | 1          | Roughly      | No                         | When dying ReLUs are a problem; GAN discriminators |
| ELU                | (−α, ∞)                    | 1          | Roughly      | Negative side, gently      | Deeper MLPs and CNNs where smoothness helps        |
| GELU               | about (−0.17, ∞)           | about 1.13 | Roughly      | Negative side, gently      | Transformers (BERT, GPT)                           |
| SiLU / Swish       | about (−0.28, ∞)           | about 1.1  | Roughly      | Negative side, gently      | Vision models; SwiGLU in LLMs                      |
| Softmax            | Probabilities summing to 1 | n/a        | n/a          | When one score dominates   | Multi-class outputs; attention weights             |

**Activation and initialization go together.** Weight initialization is chosen so signals neither shrink nor explode through the layers, and the right scale depends on the activation: **Xavier (Glorot)** initialization for sigmoid and tanh, **He (Kaiming)** initialization for ReLU and its relatives (11.2).

## 11.2 Core math

**Definitions and derivatives.** The derivatives are what backpropagation multiplies by, so they matter more than the functions:

$$
\sigma(z) = \frac{1}{1 + e^{-z}},\ \ \sigma'(z) = \sigma(z)\big(1 - \sigma(z)\big) \le \tfrac{1}{4} \qquad \tanh'(z) = 1 - \tanh^2(z) \le 1
$$

$$
\text{ReLU}(z) = \max(0, z),\ \ \text{ReLU}'(z) = \begin{cases} 1 & z > 0 \\ 0 & z < 0 \end{cases} \qquad \text{ELU}(z) = \begin{cases} z & z > 0 \\ \alpha\,(e^{z} - 1) & z \le 0 \end{cases}
$$

$$
\text{GELU}(z) = z\,\Phi(z) \approx 0.5\,z\Big(1 + \tanh\big[\sqrt{2/\pi}\,(z + 0.044715\,z^3)\big]\Big), \qquad \text{SiLU}(z) = z\,\sigma(z)
$$

The sigmoid derivative is convenient: it can be computed from the output alone. ReLU's derivative at exactly 0 is undefined; frameworks use 0. Φ is the standard Gaussian cumulative distribution function, and the tanh form is the fast approximation used in the original BERT and GPT-2 code.

**Why gradients vanish.** For a network with layers h_l = f(z_l) and z_l = W_l h\_{l−1}, the gradient reaching layer 1 is a product of one term per layer:

$$
\frac{\partial L}{\partial h_1} = \frac{\partial L}{\partial h_n} \prod_{l=2}^{n} W_l^\top\, \operatorname{diag}\big(f'(z_l)\big)
$$

With sigmoid, every f′ factor is at most 0.25, so after 20 layers the activation factors alone can shrink the gradient by 0.25²⁰ ≈ 10⁻¹². If the weights are large instead, the product can explode. ReLU's factor is exactly 1 on active units, so the scale of the gradient is set by the weights, which initialization controls.

**Initialization keeps the variance steady.** For a layer with n_in inputs and n_out outputs, choose the weight variance so activations (forward) and gradients (backward) keep roughly the same variance from layer to layer:

$$
\text{Xavier / Glorot (sigmoid, tanh): } \operatorname{Var}(W) = \frac{2}{n_{\text{in}} + n_{\text{out}}}, \qquad \text{He / Kaiming (ReLU family): } \operatorname{Var}(W) = \frac{2}{n_{\text{in}}}
$$

The extra factor of 2 in He initialization compensates for ReLU zeroing about half its inputs, which halves the variance passed on. **Batch normalization**, **layer normalization** and **residual connections** (which add an identity path, so gradients can skip layers) further protect gradient flow and are why very deep networks train today.

**Softmax and its saturation.** Softmax, p_k = e^{z_k} / Σ_j e^{z_j}, is invariant to adding a constant to all scores, which is why implementations subtract the maximum score (Chapter 8). Dividing scores by a **temperature** T before softmax controls sharpness: T < 1 makes the distribution peakier, T > 1 flatter. Attention divides scores by √d for the same reason: large dot products would saturate the softmax and shrink its gradients.

## 11.3 Code

**Gradient flow through 20 layers.** Four 20-layer networks (64 units per layer) with the recommended initialization for each activation: Xavier for sigmoid and tanh, He for ReLU and GELU. One backward pass on the same batch; compare each layer's weight-gradient norm.

```python
import torch, torch.nn as nn

def make_mlp(act, depth=20, width=64, init="he"):
    layers = []
    for i in range(depth):
        lin = nn.Linear(2 if i == 0 else width, width)
        if init == "xavier":
            nn.init.xavier_normal_(lin.weight)
        else:
            nn.init.kaiming_normal_(lin.weight, nonlinearity="relu")
        nn.init.zeros_(lin.bias)
        layers += [lin, act()]
    return nn.Sequential(*layers, nn.Linear(width, 1))

torch.manual_seed(0)
X = torch.randn(256, 2)
y = (X[:, 0] * X[:, 1] > 0).float().unsqueeze(1)          # XOR-like target

for name, act, init in [("Sigmoid", nn.Sigmoid, "xavier"), ("Tanh", nn.Tanh, "xavier"),
                        ("ReLU", nn.ReLU, "he"), ("GELU", nn.GELU, "he")]:
    torch.manual_seed(0)
    model = make_mlp(act, init=init)
    nn.functional.binary_cross_entropy_with_logits(model(X), y).backward()
    norms = [m.weight.grad.norm().item() for m in model if isinstance(m, nn.Linear)]
    print(name, norms[0], norms[19])                     # first vs last hidden layer
```

| Activation (initialization) | Gradient norm, layer 1 | Layer 20   | Layer 1 ÷ layer 20 |
| --------------------------- | ---------------------- | ---------- | ------------------ |
| Sigmoid (Xavier)            | 2.3 × 10⁻¹⁴            | 1.2 × 10⁻² | 1.9 × 10⁻¹²        |
| Tanh (Xavier)               | 3.8 × 10⁻²             | 2.5 × 10⁻² | 1.5                |
| ReLU (He)                   | 3.3 × 10⁻²             | 8.4 × 10⁻² | 0.40               |
| GELU (He)                   | 7.5 × 10⁻³             | 6.5 × 10⁻² | 0.12               |

The sigmoid network's first layer receives a gradient about a trillion times smaller than its last: it effectively cannot learn. Tanh, ReLU and GELU keep every layer within roughly one order of magnitude. Training 10-layer networks with Adam for 500 steps on two-moons data confirms it: sigmoid reaches 86.5% training accuracy and a loss of 0.289, while tanh, ReLU, ELU and GELU all reach about 98% with losses between 0.047 and 0.069.

**A dead ReLU, and why Leaky ReLU recovers.** A single unit whose bias a bad update has pushed to −20:

```python
torch.manual_seed(0)
x = torch.randn(1000, 5)
w = torch.randn(5, requires_grad=True)
b = torch.tensor(-20.0, requires_grad=True)              # bias pushed far negative

for name, act in [("ReLU", torch.relu),
                  ("Leaky ReLU", lambda z: nn.functional.leaky_relu(z, 0.01))]:
    w.grad, b.grad = None, None
    act(x @ w + b).sum().backward()
    print(name, (x @ w + b > 0).sum().item(), w.grad.norm().item(), b.grad.item())
# ReLU        0 inputs active   weight gradient 0.0    bias gradient 0.0   -> dead forever
# Leaky ReLU  0 inputs active   weight gradient 0.708  bias gradient 10.0  -> can recover
```

The ReLU unit is negative for all 1,000 inputs, so every gradient is exactly zero and no amount of training can revive it. The leaky version still passes 1% of the gradient, enough to move the bias back. In practice, dying ReLUs are kept rare by sensible learning rates, He initialization and normalization layers; if many units are dead, monitor the fraction of units that never activate on a batch.

## 11.4 In system design

- **Output activations follow the task**, and are part of the design: sigmoid for binary or multi-label outputs (several labels can be true), softmax for mutually exclusive classes, no activation (linear) for regression, softplus or exp for strictly positive quantities like durations or counts. Getting this wrong is a common design-round slip.
- **Hidden activations follow the architecture's defaults**: ReLU for CNNs and simple MLPs, GELU or SwiGLU in transformers, tanh and sigmoid inside recurrent cells. Changing them rarely pays off compared with data, architecture or tuning, so do not spend design-round time on it unless asked.
- **Inference cost.** ReLU is a single comparison; GELU and SiLU need an exponential or error function, usually approximated or fused into the preceding matrix multiply by inference compilers. On edge devices and in integer-only quantized models (Chapter 13), ReLU and ReLU6 (ReLU capped at 6) are much easier to run fast, which is why mobile networks such as MobileNet use them.
- **Numerical stability in serving.** Compute sigmoid and softmax in a stable way, and do not apply them twice: if the model already outputs probabilities, applying a fused "with logits" loss or another softmax downstream silently corrupts scores.
- **Calibration and temperature.** A softmax temperature fitted on validation data (**temperature scaling**) is a cheap, standard way to calibrate a classifier's probabilities after training, and sampling temperature controls how adventurous an LLM's generations are.
- **Monitoring training health**: track activation statistics per layer (mean, variance, fraction of zeros) and gradient norms; exploding or vanishing values flag initialization, learning-rate or normalization problems early.

## 11.5 Interview Q&A

**Q1. Why do neural networks need non-linear activation functions?** Without them, every layer is a linear map and a stack of linear maps is one linear map, so a deep network would be no more expressive than logistic or linear regression. Non-linearities let the network represent curved boundaries and complex functions.

**Q2. What is the vanishing gradient problem, and how does the activation affect it?** Backpropagation multiplies the gradient by each layer's activation derivative; if those are often below 1, the gradient shrinks exponentially with depth and early layers stop learning. Sigmoid's derivative is at most 0.25 and near zero when saturated, which makes it worst; ReLU's derivative is 1 for active units, which keeps gradients alive. Initialization, normalization and residual connections also help.

**Q3. Why is ReLU so popular?** It does not saturate for positive inputs, so gradients flow through deep networks; it is extremely cheap; and it produces sparse activations. Networks with ReLU trained much faster than with sigmoid or tanh, which helped make deep CNNs practical.

**Q4. What is the dying ReLU problem, and how do you fix it?** A unit whose input is negative for every example outputs zero with zero gradient, so it can never update and stays dead. It is often caused by a large learning rate or a bad update pushing biases negative. Fixes: lower learning rates, He initialization, normalization, or variants with a non-zero negative slope: Leaky ReLU, PReLU, ELU, GELU.

**Q5. Sigmoid versus tanh: which is better in hidden layers, and why?** Tanh: its outputs are zero-centred and its maximum slope is 1 rather than 0.25, so gradients are larger and optimization zig-zags less. Both still saturate, so neither is a default in deep feed-forward networks today.

**Q6. Why do transformers use GELU instead of ReLU?** GELU is a smooth version of ReLU that lets small negative values through, weighted by how likely they are under a Gaussian. It has non-zero gradients for negative inputs and a smooth curve, which empirically gave slightly better results in BERT and GPT; later models often use SwiGLU, a gated variant built on SiLU.

**Q7. Where would you still use sigmoid?** On the output for binary and multi-label classification, to produce a probability per label; inside LSTM and GRU gates, which need values between 0 and 1 to act as soft switches; and in attention-style gating mechanisms.

**Q8. How does the choice of activation affect weight initialization?** Initialization aims to keep activation and gradient variance stable across layers. Xavier initialization (variance 2/(n_in + n_out)) suits symmetric activations like tanh; He initialization (variance 2/n_in) doubles the variance to compensate for ReLU zeroing half its inputs.

**Q9. What is the derivative of the sigmoid, and why is it convenient?** σ′(z) = σ(z)(1 − σ(z)). It depends only on the output, which is already computed in the forward pass, so backpropagation needs no extra exponentials. Its maximum is 0.25, at z = 0.

**Q10. Softmax or sigmoid outputs for a classifier that tags photos with several objects?** Sigmoid per label, with binary cross-entropy for each: labels are not mutually exclusive, so each needs its own independent probability. Softmax forces the probabilities to compete and sum to 1, which suits exactly one correct class.

## 11.6 Cheat sheet and pitfalls

**Remember:**

- No non-linearity → a deep network collapses to one linear layer.
- Sigmoid: range (0, 1), σ′ = σ(1 − σ) ≤ 0.25, saturates, not zero-centred; use for outputs and gates.
- Tanh: range (−1, 1), zero-centred, max slope 1, still saturates.
- ReLU: max(0, z), slope 1 for z > 0, cheap, sparse; can die. Leaky ReLU, PReLU and ELU fix dying.
- GELU = z·Φ(z), SiLU = z·σ(z): smooth ReLUs used in transformers; SwiGLU in many LLMs.
- Xavier initialization for tanh and sigmoid; He initialization for the ReLU family.
- Outputs: sigmoid (binary, multi-label), softmax (one class), linear (regression).

**Pitfalls that cost candidates:**

- Saying ReLU "solves" vanishing gradients entirely; initialization, normalization and residuals matter too.
- Using softmax for multi-label problems.
- Applying sigmoid or softmax before a loss that expects logits.
- Forgetting that ReLU's gradient is zero for negative inputs, the cause of dead units.
- Pairing ReLU with Xavier initialization in very deep plain networks.

## 11.7 Further reading

<ol start="28">
  <li><a href="https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html#elu">ELU</a> (ML Glossary, activation functions)</li>
  <li><a href="https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html#relu">ReLU</a> (ML Glossary)</li>
  <li><a href="https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html#tanh">Tanh</a> (ML Glossary)</li>
  <li><a href="https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html#sigmoid">Sigmoid</a> (ML Glossary)</li>
</ol>

**Supplementary:** Glorot and Bengio, [Understanding the difficulty of training deep feedforward neural networks](https://proceedings.mlr.press/v9/glorot10a.html) (Xavier initialization); He et al., [Delving Deep into Rectifiers](https://arxiv.org/abs/1502.01852) (He initialization and PReLU); Hendrycks and Gimpel, [Gaussian Error Linear Units (GELUs)](https://arxiv.org/abs/1606.08415).
