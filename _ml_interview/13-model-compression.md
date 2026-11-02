---
title: "Model Compression"
chapter: 13
permalink: "/blog/cracking-the-ml-interview/model-compression/"
updated: 2026-11-01
description: "The model that wins offline is often too big, too slow or too expensive to serve."
---

The model that wins offline is often too big, too slow or too expensive to serve. Compression makes it smaller and faster while keeping most of its accuracy, and it is how large models reach phones, browsers and tight latency budgets. Interviewers expect you to know the four main families, what each one actually saves, and the trade-offs in production.

## 13.1 Intuition

**Why compress?**

- **Latency:** a ranking model may have 10 ms; a keyboard suggestion even less.
- **Memory and cost:** a 7B-parameter model needs 14 GB in 16-bit precision, more than many GPUs serving several models at once, and serving cost scales with hardware.
- **Edge deployment:** phones, cars and cameras have limited memory, compute and battery, and may have no network.
- **Throughput:** smaller models serve more requests per GPU.

Compression works because trained networks are **over-parameterized**: many weights are near zero or redundant, and full 32-bit precision is far more than the model needs. The survey in reference 40 groups the methods into four families.

<figure class="figure-pending"><div class="figure-pending__box">diagram coming soon</div><figcaption>model compression techniques · grouped after the survey in reference 40</figcaption></figure>

**Quantization** represents numbers with fewer bits: 32-bit floats become 8-bit integers (int8) or even 4-bit ones (int4). Each value is mapped to an integer grid with a **scale** (and sometimes a **zero point**) so the grid covers the value range. Int8 makes weights 4 times smaller than float32, and integer arithmetic is faster and more energy-efficient on supported hardware.

- **Post-training quantization (PTQ)** quantizes a trained model directly. **Dynamic quantization** quantizes weights ahead of time and activations on the fly; **static quantization** also fixes activation scales in advance, using a small **calibration** set. PTQ is fast and needs no retraining, and int8 PTQ usually costs little accuracy.
- **Quantization-aware training (QAT)** (reference 39) simulates quantization during training (**fake quantization**): the forward pass uses rounded weights so the network learns to tolerate the rounding. It recovers accuracy when PTQ loses too much, typically at 4 bits or below, or for sensitive models.
- **Granularity matters:** one scale for a whole tensor (**per-tensor**) is simple but a single large weight forces a coarse grid for all the others; one scale per output channel (**per-channel**) is far more accurate for weights.
- For LLMs, **weight-only** methods such as GPTQ and AWQ quantize weights to 4 bits after training while keeping activations in 16 bits, because LLM activations contain rare, very large outliers that are hard to quantize.

**Pruning** removes parts of the network that contribute little.

- **Unstructured pruning** zeroes individual weights, usually those with the smallest magnitude. It can remove 80 to 90% of weights with little accuracy loss, but the result is a sparse matrix that is only faster on hardware or kernels that exploit sparsity.
- **Structured pruning** removes whole neurons, channels, attention heads or layers. It saves less accuracy for the same parameter count but produces smaller dense layers that are faster everywhere.
- Pruning is usually **iterative**: prune a little, fine-tune to recover, repeat. The **lottery ticket hypothesis** observes that dense networks contain small subnetworks that, trained from the right initialization, match the full network.

**Low-rank factorization** replaces a large weight matrix with the product of two thin matrices, using the singular value decomposition (SVD). It works when the matrix has a few dominant directions. **LoRA (low-rank adaptation)** uses the same idea for fine-tuning: freeze the big weights and learn only a low-rank update.

**Knowledge distillation** trains a small **student** to reproduce a large **teacher**'s outputs, not just the hard labels. The teacher's full probability distribution (with a softening **temperature**) carries **dark knowledge**: which wrong classes are close to the right one (a 7 that looks like a 1). DistilBERT, for example, kept about 97% of BERT's language-understanding performance with 40% fewer parameters while running 60% faster. Distillation is also how large LLMs label data to train smaller, cheaper ones.

| Technique              | What shrinks                     | Typical saving          | Needs retraining?          | Speed-up depends on                    |
| ---------------------- | -------------------------------- | ----------------------- | -------------------------- | -------------------------------------- |
| PTQ (int8)             | Bits per weight and activation   | 4× smaller than float32 | No (calibration only)      | Integer kernels on the target hardware |
| QAT                    | Same as PTQ, at lower bit widths | 4× to 8× or more        | Yes (fine-tuning)          | Same                                   |
| Unstructured pruning   | Number of non-zero weights       | 5× to 10× fewer weights | Usually fine-tuning        | Sparse kernels or hardware support     |
| Structured pruning     | Neurons, channels, heads, layers | 1.5× to 3×              | Fine-tuning                | Works everywhere (dense)               |
| Low-rank factorization | Parameters in large matrices     | Depends on rank         | Often fine-tuning          | Fewer FLOPs if the rank is low enough  |
| Distillation           | The whole architecture           | Any student size        | Yes (training the student) | Student design                         |

**Combine them.** Production systems stack techniques: distill a large model into a smaller architecture, prune it, then quantize it. Each step costs a little accuracy, so measure after each one on the metrics and slices that matter.

## 13.2 Core math

**Affine (asymmetric) quantization** maps a real value x to a b-bit integer q with a scale s and an integer zero point z, so that real zero is represented exactly:

$$
q = \operatorname{clamp}\Big(\operatorname{round}\big(\tfrac{x}{s}\big) + z,\ q_{\min},\ q_{\max}\Big), \qquad \hat x = s\,(q - z), \qquad s = \frac{x_{\max} - x_{\min}}{q_{\max} - q_{\min}}
$$

For 8 bits, q runs from 0 to 255 (unsigned) or −128 to 127 (signed). **Symmetric** quantization fixes z = 0 and uses s = max|x| / (2^(b−1) − 1); it is standard for weights, which are centred on zero. Asymmetric quantization suits activations such as ReLU outputs, which are all non-negative. The rounding error per value is at most s/2, so a large range (one outlier) makes s large and the error large for every other value. Per-channel quantization gives each output channel its own s.

**Integer matrix multiplication.** For y = Wx with symmetric weights (scale s_w) and activations (scale s_x), the product is computed entirely in integers and rescaled once:

$$
y \approx s_w\, s_x \sum_j q_{w,j}\, q_{x,j}
$$

The int8 × int8 products are accumulated in 32-bit integers, then multiplied by the combined scale and requantized for the next layer.

**Fake quantization and the straight-through estimator (STE).** Rounding has zero gradient almost everywhere, so QAT cannot backpropagate through it directly. QAT computes the forward pass with quantized-then-dequantized weights, Q(w), and in the backward pass treats Q as the identity (inside the clipping range):

$$
\text{forward: } y = f\big(Q(w)\big), \qquad \text{backward: } \frac{\partial L}{\partial w} \approx \frac{\partial L}{\partial Q(w)}
$$

In code this is the one-line trick w + (Q(w) − w).detach(): its value is Q(w), but its gradient flows to w unchanged. The full-precision weights keep accumulating small updates that rounding alone would erase.

**Magnitude pruning.** Keep weight w_i only if |w_i| exceeds a threshold τ chosen so a target fraction (the **sparsity**) is removed, either per layer or globally across layers:

$$
m_i = \mathbb{1}\big[\,|w_i| > \tau\,\big], \qquad w \leftarrow m \odot w
$$

The mask m is reapplied after every fine-tuning step so pruned weights stay at zero. Storing a sparse matrix needs the values plus their indices, so unstructured sparsity saves memory only above roughly 50 to 70% sparsity, and saves time only with sparse kernels; structured pruning removes whole rows or channels, shrinking the dense matrices directly.

**Knowledge distillation loss.** With teacher logits z_t, student logits z_s, temperature T and mixing weight α:

$$
L = \alpha\, T^2\, D_{\text{KL}}\Big(\operatorname{softmax}\big(\tfrac{z_t}{T}\big)\ \Big\|\ \operatorname{softmax}\big(\tfrac{z_s}{T}\big)\Big) + (1 - \alpha)\,\text{CE}\big(y,\ \operatorname{softmax}(z_s)\big)
$$

A temperature above 1 flattens both distributions so the small probabilities on wrong classes (the dark knowledge) carry weight. The T² factor keeps the soft-target gradients on the same scale as the hard-label term as T changes.

**Low-rank factorization.** The SVD writes W = UΣVᵀ. Keeping the top r singular values gives the best rank-r approximation in squared error (the Eckart–Young theorem), and replaces one m × n layer with two layers of sizes m × r and r × n:

$$
W \approx \underbrace{U_r \Sigma_r}_{m \times r}\, \underbrace{V_r^\top}_{r \times n}, \qquad \text{parameters: } mn \;\to\; r(m + n)
$$

This saves parameters and FLOPs whenever r < mn/(m + n); for a 256 × 256 layer, any rank below 128.

**LoRA** applies the same idea to fine-tuning: keep W frozen and learn W + BA with B (m × r) and A (r × n) at a small rank such as 8 or 16, which trains well under 1% of the parameters of a large layer.

## 13.3 Code

One 85,002-parameter network (64 → 256 → 256 → 10) trained on the scikit-learn digits dataset (1,797 8×8 images; 70/30 split) reaches 97.4% test accuracy. Every technique is applied to that same model.

```python
import copy, io
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

X, y = load_digits(return_X_y=True)                              # 8x8 digit images, 10 classes
X = (X / 16.0).astype(np.float32)
X_tr, X_te, y_tr, y_te = map(torch.tensor, train_test_split(X, y, test_size=0.3, stratify=y, random_state=0))

def mlp(h=256):
    return nn.Sequential(nn.Linear(64, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 10))

def train(model, epochs=60, lr=1e-3, seed=0, after_step=None):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        perm = torch.randperm(len(X_tr))
        for i in range(0, len(X_tr), 64):
            b = perm[i:i + 64]
            opt.zero_grad()
            F.cross_entropy(model(X_tr[b]), y_tr[b]).backward()
            opt.step()
            if after_step: after_step()
    return model

def accuracy(model):
    with torch.no_grad():
        return (model(X_te).argmax(1) == y_te).float().mean().item()

def size_kb(model):
    buf = io.BytesIO(); torch.save(model.state_dict(), buf); return buf.getbuffer().nbytes / 1024

base = train(mlp())                                              # 97.4% accuracy, 334.6 KB

# 1. Symmetric quantization, per-tensor or per-channel
def quantize(w, bits=8, per_channel=False):
    qmax = 2 ** (bits - 1) - 1                                   # 127 for int8
    scale = (w.abs().amax(dim=1, keepdim=True) if per_channel else w.abs().max()) / qmax
    return torch.clamp(torch.round(w / scale), -qmax - 1, qmax) * scale   # quantize, then dequantize

W = base[2].weight.data.clone()
W_outlier = W.clone(); W_outlier[0] *= 20                        # one output channel with large weights
for name, w in [("normal", W), ("outlier channel", W_outlier)]:
    for pc in [False, True]:
        print(name, pc, ((quantize(w, 8, pc) - w).norm() / w.norm()).item())

# 2. Post-training quantization of all weights at several bit widths
def ptq(model, bits):
    m = copy.deepcopy(model)
    for layer in m:
        if isinstance(layer, nn.Linear):
            layer.weight.data = quantize(layer.weight.data, bits, per_channel=True)
    return m
for bits in [8, 4, 3, 2]:
    print("PTQ", bits, accuracy(ptq(base, bits)))

# 3. Quantization-aware training: fake-quantize in the forward pass, straight-through gradients
class QLinear(nn.Linear):
    def __init__(self, *args, bits=4, **kw):
        super().__init__(*args, **kw); self.bits = bits
    def forward(self, x):
        wq = quantize(self.weight, self.bits, per_channel=True)
        w = self.weight + (wq - self.weight).detach()           # forward: wq; backward: identity
        return F.linear(x, w, self.bias)

for bits in [4, 3, 2]:
    q = nn.Sequential(QLinear(64, 256, bits=bits), nn.ReLU(), QLinear(256, 256, bits=bits), nn.ReLU(),
                      QLinear(256, 10, bits=bits))
    q.load_state_dict(base.state_dict())                          # start from the trained model
    print("QAT", bits, accuracy(train(q, epochs=20, lr=3e-4)))

# 4. PyTorch's built-in dynamic int8 quantization
from torch.ao.quantization import quantize_dynamic
qd = quantize_dynamic(copy.deepcopy(base), {nn.Linear}, dtype=torch.qint8)
print("dynamic int8", accuracy(qd), size_kb(qd))

# 5. Global magnitude pruning, then fine-tuning with the mask held fixed
def prune(model, sparsity):
    m = copy.deepcopy(model)
    lins = [l for l in m if isinstance(l, nn.Linear)]
    threshold = torch.quantile(torch.cat([l.weight.data.abs().flatten() for l in lins]), sparsity)
    masks = [(l.weight.data.abs() > threshold).float() for l in lins]
    for l, mask in zip(lins, masks):
        l.weight.data *= mask
    def reapply():
        for l, mask in zip(lins, masks):
            l.weight.data *= mask
    return m, reapply

for s in [0.5, 0.8, 0.9, 0.95, 0.98]:
    m, reapply = prune(base, s)
    before = accuracy(m)
    after = accuracy(train(m, epochs=10, lr=3e-4, after_step=reapply))
    print(f"prune {s:.0%}", before, after)

# 6. Low-rank factorization of the 256 x 256 middle layer
U, S, Vh = torch.linalg.svd(base[2].weight.data, full_matrices=False)
for r in [32, 16, 8, 4]:
    m = copy.deepcopy(base)
    m[2].weight.data = (U[:, :r] * S[:r]) @ Vh[:r]
    print("rank", r, r * (256 + 256), accuracy(m))
```

**Quantization results:**

| Experiment                                 | Result                                                 |
| ------------------------------------------ | ------------------------------------------------------ |
| int8 error, normal layer                   | 1.37% per-tensor, 0.80% per-channel (relative error)   |
| int8 error, one outlier channel (×20)      | 9.5% per-tensor, 0.74% per-channel                     |
| PTQ, weights at 8 / 4 / 3 / 2 bits         | 97.4% / 97.2% / 97.0% / 82.4% accuracy                 |
| QAT, weights at 4 / 3 / 2 bits (20 epochs) | 97.4% / 97.4% / 97.6% accuracy                         |
| PyTorch dynamic int8                       | 97.4% accuracy; file 334.6 KB → 89.1 KB (3.8× smaller) |

One outlier channel made per-tensor int8 error 7 times worse, while per-channel scaling was unaffected. Post-training quantization held up down to 3 bits on this small, easy model but lost 15 points at 2 bits; quantization-aware training recovered all of it. Large models are usually more sensitive, which is why 4-bit LLM quantization needs careful methods.

**Pruning results:**

<figure class="figure-pending"><div class="figure-pending__box">diagram coming soon</div><figcaption>experiment in 13.3, part 5 · 5 pruning levels</figcaption></figure>

Without fine-tuning, accuracy falls quickly past 50% sparsity. Ten epochs of fine-tuning with the mask fixed bring 80% sparsity back to the baseline and 90% to within 2 points; at 95% it recovers only partly, and at 98% the network no longer learns.

**Low-rank results.** Replacing the middle layer (65,536 weights) by rank 32 (16,384 weights) or rank 16 (8,192) left accuracy unchanged at 97.4%; rank 8 cost 0.4 points and rank 4 dropped it to 77.4%, all without any fine-tuning.

**Distillation: when it helps and when it doesn't.** Two experiments with the same 97.4% teacher, using the loss in 13.2:

1. **Same labeled data.** Small students (16 and 8 hidden units) trained on the full labeled training set did _worse_ with distillation (T = 4, α = 0.7) than with hard labels: 92.3% versus 95.2%, and 83.8% versus 89.7%, averaged over 5 seeds. Lower temperatures only closed the gap. The teacher was 100% accurate on its own training data, with an average top-class probability of 99.8%, so its soft labels there were almost one-hot and added little information.
2. **Teacher labels extra unlabeled data.** A 32-unit student (3,466 parameters) with only 100 labeled examples reached 87.4%. Trained instead on the teacher's soft labels (T = 4) for all 1,257 training images, with their true labels hidden, it reached 91.0%.

The lesson: distillation transfers knowledge best through data the student could not otherwise learn from, and soft labels are most informative when the teacher is not overconfident on that data. In practice teams distill on large unlabeled or augmented sets, hold out data from the teacher's training, and tune T and α.

```python
def distill_student(h, seed, T=4.0, epochs=40):
    """Student learns only from the teacher's softened outputs on every training image."""
    with torch.no_grad():
        teacher_logits = base(X_tr)
    torch.manual_seed(seed)
    student = mlp(h)
    opt = torch.optim.Adam(student.parameters(), lr=1e-3)
    for _ in range(epochs):
        perm = torch.randperm(len(X_tr))
        for i in range(0, len(perm), 128):
            b = perm[i:i + 128]
            opt.zero_grad()
            loss = F.kl_div(F.log_softmax(student(X_tr[b]) / T, dim=1),
                            F.softmax(teacher_logits[b] / T, dim=1),
                            reduction="batchmean") * T * T
            loss.backward()
            opt.step()
    return student

print(np.mean([accuracy(distill_student(32, s)) for s in range(5)]))   # 0.910
```

PyTorch is moving its quantization tools from `torch.ao.quantization` to the separate **torchao** library; recent versions print a deprecation notice for `quantize_dynamic`, so check the current documentation (reference 39) for the recommended API.

## 13.4 In system design

- **Start from the budget.** State the latency target (p50 and p99), memory limit, hardware and cost per request first; then pick the cheapest technique that meets them. Often the first win is not compression at all: batching, caching, a smaller off-the-shelf model, or exporting to an optimized runtime (ONNX Runtime, TensorRT, OpenVINO, TFLite, Core ML).
- **Match the technique to the hardware.** Int8 helps only where integer kernels exist (most server CPUs, NVIDIA tensor cores, phone NPUs). Unstructured sparsity rarely speeds up commodity hardware; structured pruning and distillation shrink dense compute everywhere. Measure real latency on the target device, not parameter counts.
- **LLM serving** typically uses 16-bit (bf16 or fp16) weights, 8-bit or 4-bit weight-only quantization (GPTQ, AWQ, bitsandbytes NF4) to fit bigger models per GPU, quantized **KV caches** to serve longer contexts, distillation into smaller models for high-volume tasks, and LoRA adapters so many fine-tuned variants share one base model.
- **Edge deployment** (keyboards, cameras, wearables) combines a compact architecture, int8 quantization and often distillation; the model must also fit the app's download size and battery budget.
- **Validate the compressed model like a new model.** Compression can hurt rare classes, long inputs or particular user groups more than the average metric shows, so compare per-slice metrics, check calibration, and shadow-test or A/B test before replacing the original (Chapter 14).
- **Keep the pipeline reproducible.** Store the calibration set, quantization configuration and pruning masks with the model version, so a retrained model can be compressed the same way automatically.

## 13.5 Interview Q&A

**Q1. What is the difference between post-training quantization and quantization-aware training?** PTQ quantizes a trained model directly, using a calibration set to choose activation ranges; it is fast and needs no retraining, and works well at 8 bits. QAT inserts fake quantization during training so the weights adapt to rounding, using the straight-through estimator for gradients; it costs a training run but recovers accuracy at low bit widths or for sensitive models.

**Q2. How does int8 quantization work?** Each tensor (or channel) gets a scale, and optionally a zero point, mapping its real range onto 256 integer levels: q = round(x/s) + z. Matrix multiplications run on int8 values with 32-bit accumulation and are rescaled once. Weights become 4 times smaller than float32 and integer math is faster on supported hardware.

**Q3. Why is per-channel quantization more accurate than per-tensor?** With one scale per tensor, a few large weights force a coarse grid for every other weight. Per-channel scales adapt to each output channel's range. In the example, one outlier channel made per-tensor int8 error 7 times larger while per-channel error barely changed.

**Q4. What is the straight-through estimator?** Rounding has zero gradient almost everywhere, so QAT uses quantized weights in the forward pass but treats quantization as the identity in the backward pass. In code: w + (quantize(w) − w).detach(). The full-precision weights keep receiving useful gradients.

**Q5. Structured or unstructured pruning?** Unstructured pruning removes individual weights and reaches higher sparsity at the same accuracy, but produces sparse matrices that are only faster with special kernels or hardware. Structured pruning removes whole neurons, channels or heads, giving smaller dense layers that are faster on any hardware, at some accuracy cost. Choose structured when latency on standard hardware is the goal.

**Q6. Explain knowledge distillation. Why use a temperature?** A small student is trained to match a large teacher's output distribution as well as the true labels. Dividing logits by a temperature above 1 softens the teacher's distribution, exposing the relative probabilities of wrong classes, which carry information about class similarity. The soft-target loss is multiplied by T² to keep gradient scales comparable.

**Q7. When might distillation not help?** When the teacher is overconfident on the training data (its soft labels are nearly one-hot), when the student is too small to represent the teacher's function, or when temperature and weighting are poorly tuned. It helps most when the teacher labels data the student could not otherwise learn from, such as unlabeled or augmented examples.

**Q8. How does low-rank factorization reduce parameters?** It approximates an m × n weight matrix by the product of m × r and r × n matrices from its truncated SVD, reducing parameters from mn to r(m + n). It works when the matrix's singular values decay quickly; LoRA uses the same idea to fine-tune large models cheaply.

**Q9. You need to cut a model's latency by 4× without losing more than 1 point of accuracy. Plan it.** Profile first to see where time goes. Try runtime optimizations (an optimized engine, operator fusion, batching), then int8 PTQ on the target hardware and measure. If not enough, distill into a smaller architecture or structurally prune, fine-tune, and quantize again, with QAT if needed. Check per-slice accuracy and calibration at each step, then validate online.

**Q10. How are large language models compressed for serving?** Weight-only 8-bit or 4-bit quantization (GPTQ, AWQ, NF4) to cut memory, since generation is mostly memory-bandwidth-bound; quantized KV caches for long contexts; distillation into smaller models for specific tasks; pruning of attention heads or layers in some cases; and LoRA adapters so many fine-tuned variants share one base model.

## 13.6 Cheat sheet and pitfalls

**Remember:**

- Four families: quantization, pruning, low-rank factorization, distillation; plus compact architectures.
- Quantization: q = round(x/s) + z; symmetric (z = 0) for weights; per-channel beats per-tensor.
- int8 = 4× smaller than float32; int4 = 8×. PTQ is cheap; QAT (fake quantization + STE) recovers low-bit accuracy.
- Magnitude pruning: drop the smallest |w|, fine-tune with the mask fixed, iterate.
- Unstructured = sparse, needs sparse kernels; structured = smaller dense layers, faster everywhere.
- Distillation loss: αT² KL(teacher ‖ student at temperature T) + (1 − α) cross-entropy.
- Low rank: mn → r(m + n) parameters; LoRA for fine-tuning.
- Measure latency on the target hardware and accuracy per slice.

**Pitfalls that cost candidates:**

- Claiming that removing 90% of weights makes a model 10× faster.
- Quantizing without a representative calibration set, or with per-tensor scales on outlier-heavy layers.
- Forgetting to fine-tune after pruning.
- Assuming distillation always beats training the small model on labels.
- Judging a compressed model only by its average accuracy.

## 13.7 Further reading

<ol start="39">
  <li><a href="https://pytorch.org/docs/stable/quantization.html">Quantization</a> (PyTorch documentation)</li>
  <li>Cheng et al., <a href="https://arxiv.org/pdf/1710.09282.pdf">A Survey of Model Compression and Acceleration for Deep Neural Networks</a> (2017)</li>
</ol>

**Supplementary:** Hinton, Vinyals and Dean, [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531); Han, Mao and Dally, [Deep Compression](https://arxiv.org/abs/1510.00149) (pruning, quantization and Huffman coding); Sanh et al., [DistilBERT](https://arxiv.org/abs/1910.01108); Hu et al., [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685).
