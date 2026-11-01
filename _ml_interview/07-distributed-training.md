---
title: "Distributed Training"
chapter: 7
permalink: "/blog/cracking-the-ml-interview/distributed-training/"
updated: 2026-11-01
description: "Why modern models are trained on many GPUs, how data, tensor and pipeline parallelism split the work, and what each costs in memory and communication."
---

Modern models are trained on hundreds or thousands of **GPUs (graphics processing units)** or **TPUs (tensor processing units, Google's ML accelerators)**, because one device has neither the memory to hold them nor the speed to train them in reasonable time. ML engineer and LLM roles expect you to explain how work is split across devices, what each split costs in memory and communication, and how to choose between them.

## 7.1 Intuition

**Two reasons to distribute.**

- **Too slow on one device.** A useful rule of thumb puts the compute for training a transformer at about 6 × parameters × training tokens floating-point operations (**FLOPs**). A 7-billion-parameter model on 2 trillion tokens needs about 6 × 7×10⁹ × 2×10¹² ≈ 8.4×10²² FLOPs. If one GPU sustains 400 TFLOP/s (4×10¹⁴ per second), that is about 2.1×10⁸ seconds, nearly 7 years; 1,024 GPUs at the same efficiency bring it to about 2.4 days.
- **Too big for one device.** Training with the Adam optimizer in **mixed precision** (16-bit weights and gradients for speed, 32-bit copies for accuracy) needs about **16 bytes per parameter** before counting activations (7.2 breaks this down). A 7B model needs about 112 GB; a 70B model about 1.1 TB. A top data-centre GPU has 80 to 192 GB.

**Data parallelism** (reference 13) is the default. Every GPU holds a full copy of the model and processes a different slice of each batch. After the backward pass, the GPUs **all-reduce** their gradients: every GPU ends up with the average gradient across all of them. Every copy then takes the same optimizer step and stays identical.

- **Synchronous** data parallelism waits for all GPUs every step. It is mathematically the same as training on the combined batch (7.3 checks this), but the slowest GPU, the **straggler**, sets the pace.
- **Asynchronous** data parallelism, the older **parameter server** design, lets workers push gradients to central servers and pull fresh weights whenever they are ready. No one waits, but workers compute on **stale** weights, which hurts convergence. Synchronous all-reduce is the modern default.
- The catch: the full model, gradients and optimizer state must still fit on each GPU, and communication grows with model size.

**Model parallelism** (reference 14) splits the model itself across devices:

- **Pipeline parallelism (inter-layer)** puts consecutive groups of layers, called **stages**, on different GPUs. Activations flow forward from stage to stage and gradients flow back. Run naively, only one GPU works at a time; splitting each batch into **micro-batches** lets stages work on different micro-batches at once, leaving idle **bubbles** only at the start and end of each step.
- **Tensor parallelism (intra-layer)** splits individual weight matrices across GPUs, so every GPU computes part of every layer and they exchange partial results inside each layer. It needs very fast links, so it usually stays within one server (a **node**).
- **Sharded data parallelism** (DeepSpeed's **ZeRO, Zero Redundancy Optimizer**, and PyTorch's **FSDP, Fully Sharded Data Parallel**) keeps data parallelism's simplicity but stops every GPU from storing a full copy of the optimizer state, gradients and even the weights. Each GPU stores a 1/N slice and gathers the rest just in time.
- **Expert parallelism** places the different experts of a **mixture-of-experts (MoE)** model on different GPUs and routes each token to the GPUs holding its chosen experts. **Sequence (context) parallelism** splits very long sequences across GPUs.

<figure class="figure-pending"><div class="figure-pending__box">diagram coming soon</div><figcaption>data, tensor and pipeline parallelism · a 32-layer model on 4 GPUs</figcaption></figure>

| Strategy          | What is split                                       | Memory per GPU                            | Communication                                | Main cost                              |
| ----------------- | --------------------------------------------------- | ----------------------------------------- | -------------------------------------------- | -------------------------------------- |
| Data parallel     | The batch                                           | Full model, gradients and optimizer state | All-reduce of all gradients, once per step   | Model must fit on one GPU              |
| ZeRO / FSDP       | Batch, plus sharded states (and weights in stage 3) | As low as 1/N of model state              | About 1.5× data-parallel traffic in stage 3  | More communication, more engineering   |
| Pipeline parallel | Layers, into stages                                 | About 1/N of the layers                   | Activations between neighbouring stages only | Bubbles of idle time; balancing stages |
| Tensor parallel   | Each layer's matrices                               | About 1/N of each layer                   | Several all-reduces inside every layer       | Needs fast links within a node         |
| Expert parallel   | Experts of an MoE layer                             | 1/N of the experts                        | All-to-all token exchange                    | Load balancing across experts          |

**Combining them: 3D parallelism.** Large language models use all three at once, matched to the hardware. Within a node, GPUs talk over NVLink at roughly 900 GB/s per GPU on recent NVIDIA hardware; between nodes, a 400 Gb/s InfiniBand link carries about 50 GB/s. So tensor parallelism stays inside a node (say 8 GPUs), pipeline parallelism spans a few nodes, and data parallelism (often sharded) replicates that whole group many times. For example, 1,024 GPUs = 8-way tensor × 8-way pipeline × 16-way data parallelism.

**Tools that trade compute for memory**, used alongside every strategy:

- **Activation checkpointing (rematerialization):** store only some layers' activations in the forward pass and recompute the rest during the backward pass. It costs roughly one extra forward pass (about 30% more compute) and saves most activation memory.
- **Gradient accumulation:** run several small micro-batches, add up their gradients, and only then step the optimizer. It gives a large effective batch on few GPUs.
- **Mixed precision** (16-bit compute, 32-bit master weights) halves activation memory and speeds up math on tensor cores.
- **Offloading** moves optimizer state or parameters to CPU memory or NVMe disks when GPU memory runs out, at a large speed cost.

## 7.2 Core math

**Why data parallelism is exact.** With a mean loss over a batch B split into N equal shards B_k, the full-batch gradient is the average of the shard gradients. Each GPU computes one g_k, the all-reduce averages them, and every GPU applies the same update:

$$
g = \frac{1}{|B|}\sum_{i \in B} \nabla_\theta\, \ell_i = \frac{1}{N}\sum_{k=1}^{N} \underbrace{\frac{1}{|B_k|}\sum_{i \in B_k} \nabla_\theta\, \ell_i}_{g_k}, \qquad \theta \leftarrow \theta - \eta\, g
$$

Two things can break this equality: shards of unequal size (then the average must be weighted), and layers that compute statistics over the batch, such as batch normalization, which sees only its own shard unless you use synchronized batch norm.

**Training memory per parameter.** With Adam in mixed precision, each parameter needs 2 bytes for 16-bit weights, 2 for 16-bit gradients, and 12 for optimizer state (a 32-bit master copy of the weight, plus Adam's first and second moments, 4 bytes each):

$$
M_{\text{model states}} = (2 + 2 + 12)\,\Psi = 16\,\Psi \ \text{bytes}
$$

for Ψ parameters. A 7B model needs 112 GB before activations. Activations add memory that grows with batch size × sequence length × hidden size × layers, which is what activation checkpointing reduces.

**ZeRO's three stages.** With N_d data-parallel GPUs, ZeRO partitions progressively more of the model states, each GPU keeping only its slice:

$$
\text{Stage 1: } \Big(4 + \frac{12}{N_d}\Big)\Psi, \qquad \text{Stage 2: } \Big(2 + \frac{14}{N_d}\Big)\Psi, \qquad \text{Stage 3: } \frac{16\,\Psi}{N_d}
$$

| Setup (7.5B parameters, 64 GPUs)    | What each GPU stores    | Memory per GPU |
| ----------------------------------- | ----------------------- | -------------- |
| Plain data parallelism              | Everything              | 120 GB         |
| ZeRO stage 1                        | Optimizer state sharded | 31.4 GB        |
| ZeRO stage 2                        | + gradients sharded     | 16.6 GB        |
| ZeRO stage 3 (= FSDP full sharding) | + weights sharded       | 1.9 GB         |

Stages 1 and 2 cost no extra communication: a reduce-scatter plus an all-gather moves the same volume as an all-reduce. Stage 3 must gather each layer's weights just before using them in the forward and backward passes, so it moves about 1.5 times the data of plain data parallelism.

**All-reduce cost.** In a **ring all-reduce**, N GPUs are arranged in a ring; the gradient (S bytes) is cut into N chunks that travel around the ring twice, once to sum (**reduce-scatter**) and once to share the result (**all-gather**). With per-link bandwidth b and per-message latency α:

$$
T_{\text{ring}} \approx 2(N-1)\,\alpha + \frac{2(N-1)}{N}\cdot\frac{S}{b}
$$

The bandwidth term approaches 2S/b however many GPUs join (the factor is 1.75 at N = 8), which is why ring all-reduce scales well. The latency term grows with N, so very large clusters use tree or hierarchical variants. Example: a 1B-parameter model has 2 GB of 16-bit gradients; at 50 GB/s that is about 2 × 2/50 = 80 ms per step, which frameworks hide by overlapping communication with the backward pass.

**Large-batch training.** N GPUs with a per-GPU batch of b give a global batch of N · b. More data per step means fewer steps, but the learning rate must change with it. The **linear scaling rule** (Goyal et al., 2017) multiplies the learning rate by the same factor as the batch, with a **warm-up** that ramps the rate up over the first few epochs:

$$
\eta_{\text{new}} = \eta_{\text{base}} \cdot \frac{B_{\text{new}}}{B_{\text{base}}}
$$

It let ResNet-50 train on ImageNet with a batch of 8,192 in one hour without losing accuracy. Beyond a problem-specific **critical batch size**, larger batches stop speeding up training. Adaptive optimizers often follow a gentler square-root rule, and layer-wise methods (LARS, LAMB) push batch sizes further.

**Pipeline bubble.** With p pipeline stages and m micro-batches per step, the GPipe schedule leaves each GPU idle for a fraction of the step:

$$
\text{bubble fraction} = \frac{p - 1}{m + p - 1}
$$

<figure class="figure-pending"><div class="figure-pending__box">diagram coming soon</div><figcaption>GPipe schedule · 4 stages, 4 micro-batches, 14 time slots</figcaption></figure>

With p = 4: 1 micro-batch idles 75% of the time, 4 micro-batches 43%, 16 micro-batches 16%, 32 micro-batches 8.6%. So use many more micro-batches than stages (m ≥ 4p is a common rule). The **1F1B (one forward, one backward)** schedule interleaves passes so each stage holds activations for at most p micro-batches instead of m, and interleaved schedules shrink the bubble further.

**Tensor parallelism in a transformer MLP** (Megatron-LM). The block computes Y = GeLU(XA), then Z = YB. Split A by **columns** across GPUs: each computes its own slice of Y with no communication, because GeLU acts element by element. Split B by **rows**: each GPU computes a partial Z, and one all-reduce sums them:

$$
A = [A_1, A_2], \quad Y_i = \text{GeLU}(X A_i), \quad B = \begin{bmatrix} B_1 \\ B_2 \end{bmatrix}, \quad Z = Y_1 B_1 + Y_2 B_2
$$

That is one all-reduce in the forward pass and one in the backward pass per MLP block, and the same for attention, where heads are split across GPUs. Activations, not gradients, cross the links, many times per step, which is why tensor parallelism needs NVLink-class bandwidth.

## 7.3 Code

**Averaged shard gradients equal the full-batch gradient** (linear regression, 64 rows, 4 "GPUs"):

```python
import numpy as np

rng = np.random.default_rng(0)
X, y, w = rng.normal(size=(64, 5)), rng.normal(size=64), np.zeros(5)
grad = lambda Xb, yb, w: 2 * Xb.T @ (Xb @ w - yb) / len(yb)     # gradient of mean squared error

full = grad(X, y, w)
shards = np.array_split(np.arange(64), 4)
averaged = np.mean([grad(X[s], y[s], w) for s in shards], axis=0)  # what all-reduce computes
print(np.allclose(full, averaged))                                  # True
```

**Real data parallelism with PyTorch DDP (DistributedDataParallel).** This script starts two processes on a CPU with the `gloo` backend (on GPUs you would use `nccl` and launch with `torchrun`). Each process trains on its own half of the batch; the result is compared with one process training on the full batch.

```python
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

def make_model():
    torch.manual_seed(0)                                   # same initial weights everywhere
    return torch.nn.Sequential(torch.nn.Linear(10, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1))

def make_data():
    g = torch.Generator().manual_seed(1)
    return torch.randn(64, 10, generator=g), torch.randn(64, 1, generator=g)

def worker(rank, world_size, steps, out):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29512")
    dist.init_process_group("gloo", rank=rank, world_size=world_size)  # "nccl" on GPUs
    model = DDP(make_model())                              # broadcasts rank 0's weights to all
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    X, y = make_data()
    shard = X.shape[0] // world_size                       # each rank gets its own slice
    Xr, yr = X[rank * shard:(rank + 1) * shard], y[rank * shard:(rank + 1) * shard]
    for _ in range(steps):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(Xr), yr)
        loss.backward()                                    # DDP all-reduces (averages) gradients here
        opt.step()
    if rank == 0:
        torch.save(model.module.state_dict(), out)
    dist.destroy_process_group()

if __name__ == "__main__":
    steps, out = 20, "ddp_weights.pt"
    mp.spawn(worker, args=(2, steps, out), nprocs=2)

    ref = make_model()                                     # one process, full batch
    opt = torch.optim.SGD(ref.parameters(), lr=0.1)
    X, y = make_data()
    for _ in range(steps):
        opt.zero_grad()
        torch.nn.functional.mse_loss(ref(X), y).backward()
        opt.step()

    ddp_state = torch.load(out)
    diff = max((ddp_state[k] - v).abs().max().item() for k, v in ref.state_dict().items())
    print(f"max weight difference after {steps} steps: {diff:.2e}")   # 2.98e-08
```

After 20 steps the weights differ by about 3×10⁻⁸, which is floating-point rounding: synchronous data parallelism is the same algorithm as large-batch training. In real jobs, a `DistributedSampler` gives each rank its own slice of the dataset, and you call `sampler.set_epoch(epoch)` so the shuffle changes every epoch. DDP also buckets gradients and starts all-reducing each bucket as soon as its gradients are ready, overlapping communication with the rest of the backward pass.

**A training-memory calculator** for the model states (excluding activations):

```python
def train_memory_gb(params_billion, n_gpus=1, zero_stage=0):
    """Adam + mixed precision: 2 B weights + 2 B grads + 12 B optimizer state per parameter."""
    P = params_billion * 1e9
    weights, grads, optim = 2 * P, 2 * P, 12 * P
    if zero_stage >= 1: optim /= n_gpus
    if zero_stage >= 2: grads /= n_gpus
    if zero_stage >= 3: weights /= n_gpus
    return (weights + grads + optim) / 1e9

print([round(train_memory_gb(7.5, 64, s), 1) for s in range(4)])   # [120.0, 31.4, 16.6, 1.9]
print(train_memory_gb(7), train_memory_gb(70))                      # 112.0 1120.0
```

In PyTorch, ZeRO stage 3 is available as `torch.distributed.fsdp.FullyShardedDataParallel`, and DeepSpeed exposes all three stages through a configuration file.

## 7.4 In system design

- **Pick the simplest strategy that fits.** If the model, gradients and optimizer state fit on one GPU, use plain data parallelism (DDP). If they don't, shard them (FSDP or ZeRO) before reaching for pipeline or tensor parallelism, which need model changes and careful tuning. Use tensor and pipeline parallelism only for the largest models, and match them to the hardware: tensor within a node, pipeline across nodes, data parallelism across the rest.
- **Measure efficiency, not just GPU count.** **MFU (model FLOPs utilization)** is the share of the hardware's peak FLOPs that goes into useful model computation; 40 to 55% is good for large transformer training. Low MFU points to communication, input pipelines or bubbles. **Scaling efficiency** compares throughput on N GPUs with N times one GPU's.
- **Keep the input pipeline ahead of the GPUs.** Data loading, decoding and augmentation must keep up; prefetch, cache preprocessed shards, and give each worker its own data shard.
- **Plan for failure.** On thousands of GPUs, a hardware failure every few hours is normal. Checkpoint model and optimizer state regularly to shared storage, make jobs resumable, and use elastic training that can restart with fewer nodes.
- **Communication tricks:** overlap gradient all-reduce with the backward pass, use gradient accumulation to all-reduce less often, and compress gradients (16-bit or lower) when bandwidth is the bottleneck.
- **Reproducibility.** Seeds, data order, and the number of GPUs (which changes the effective batch) all affect results. Record the global batch size and learning-rate schedule, not just per-GPU settings.
- **Fine-tuning rather than pretraining**, the common case in applied roles, often fits on one node: **parameter-efficient fine-tuning (PEFT)** such as **LoRA (low-rank adaptation)** trains small adapter matrices instead of all weights, cutting gradient and optimizer memory dramatically, and combines with FSDP when the base model is large.

## 7.5 Interview Q&A

**Q1. What is the difference between data parallelism and model parallelism?** Data parallelism copies the whole model to every GPU and splits the batch, averaging gradients with an all-reduce each step; it speeds up training but needs the model to fit on one GPU. Model parallelism splits the model itself (by layers in pipeline parallelism, within layers in tensor parallelism) so that models too large for one device can be trained, at the cost of more communication and idle time.

**Q2. Is synchronous data-parallel training the same as single-GPU training?** With equal shards and a mean loss, yes: the averaged gradient equals the full-batch gradient, so N GPUs with batch b each behave like one GPU with batch N · b. Differences come from batch normalization statistics computed per GPU, floating-point summation order, and the larger effective batch, which usually requires a higher learning rate.

**Q3. Synchronous or asynchronous training?** Synchronous all-reduce gives deterministic, well-understood convergence but waits for the slowest worker. Asynchronous parameter-server training avoids waiting but updates with stale gradients, which can slow or destabilize convergence. Modern GPU clusters with fast interconnects almost always use synchronous training.

**Q4. How does ring all-reduce work, and why does it scale?** GPUs form a ring and the gradient is cut into N chunks. In N − 1 reduce-scatter steps, each chunk is summed as it travels round the ring; in N − 1 all-gather steps, the summed chunks are shared. Each GPU sends about 2(N − 1)/N × S bytes, nearly independent of N, so adding GPUs barely increases per-GPU traffic.

**Q5. You double the number of GPUs. What happens to the learning rate?** If the per-GPU batch stays the same, the global batch doubles. Under the linear scaling rule, double the learning rate and use a warm-up. Check validation curves, because beyond the critical batch size larger batches stop helping and can hurt generalization.

**Q6. How much memory does it take to train a 7B model with Adam?** About 16 bytes per parameter for weights, gradients and optimizer state in mixed precision, so roughly 112 GB, plus activations. That exceeds an 80 GB GPU, so you need ZeRO/FSDP sharding, offloading, or a parameter-efficient method such as LoRA.

**Q7. Explain ZeRO's three stages.** Stage 1 shards the optimizer state across data-parallel GPUs, stage 2 also shards gradients, and stage 3 also shards the weights, gathering each layer's weights just in time. Memory per GPU falls from 16Ψ to about 16Ψ/N; stages 1 and 2 add no communication, and stage 3 adds about 50%.

**Q8. What is a pipeline bubble, and how do you shrink it?** Idle time at the start and end of each step while the pipeline fills and drains; its fraction is (p − 1)/(m + p − 1) for p stages and m micro-batches. Shrink it with more micro-batches, the 1F1B or interleaved schedules, fewer stages, and evenly balanced stages.

**Q9. Why is tensor parallelism usually kept inside one node?** It exchanges activations through all-reduces inside every layer, several times per step, so it is bandwidth- and latency-bound. Within a node, NVLink is roughly an order of magnitude faster than the network between nodes.

**Q10. Training is 40% slower than expected on 64 GPUs. How do you debug it?** Profile one step: check GPU utilization, time in communication versus compute, and data-loader wait time. Look for stragglers, all-reduce not overlapping with the backward pass, small per-GPU batches, pipeline bubbles, or a network bottleneck between nodes. Compare scaling from 8 to 16 to 64 GPUs to see where efficiency drops.

## 7.6 Cheat sheet and pitfalls

**Remember:**

- Training compute ≈ 6 × parameters × tokens FLOPs.
- Mixed-precision Adam: 16 bytes per parameter (2 weights + 2 gradients + 12 optimizer state), before activations.
- Data parallel: full copy per GPU, split batch, all-reduce gradients; equals large-batch training.
- ZeRO 1/2/3 shards optimizer state, then gradients, then weights; FSDP = ZeRO-3 in PyTorch.
- Pipeline: split layers into stages; bubble = (p − 1)/(m + p − 1); use m ≥ 4p micro-batches.
- Tensor: split matrices (column-parallel then row-parallel); one all-reduce per block per pass; keep inside a node.
- Ring all-reduce traffic per GPU ≈ 2(N − 1)/N × gradient size.
- Linear scaling rule: learning rate × k when the batch × k, with warm-up.
- Save memory with activation checkpointing, gradient accumulation, mixed precision, offloading.

**Pitfalls that cost candidates:**

- Saying data parallelism lets you train a model too large for one GPU (without sharding, it does not).
- Forgetting optimizer state when estimating memory: weights alone are only 2 of the 16 bytes per parameter.
- Scaling the batch without adjusting the learning rate.
- Using plain batch norm across GPUs and getting per-GPU statistics.
- Spreading tensor parallelism across nodes over a slow network.
- Forgetting `sampler.set_epoch`, so every epoch sees the same shuffle.

## 7.7 Further reading

<ol start="13">
  <li><a href="https://www.telesens.co/2017/12/25/understanding-data-parallelism-in-machine-learning/">Understanding data parallelism in machine learning</a> (Telesens)</li>
  <li><a href="https://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel-intro.html">Introduction to model parallelism</a> (Amazon SageMaker documentation)</li>
</ol>

**Supplementary:** Rajbhandari et al., [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054); Shoeybi et al., [Megatron-LM](https://arxiv.org/abs/1909.08053) (tensor parallelism); Huang et al., [GPipe](https://arxiv.org/abs/1811.06965) (pipeline parallelism); Goyal et al., [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677) (linear scaling rule).
