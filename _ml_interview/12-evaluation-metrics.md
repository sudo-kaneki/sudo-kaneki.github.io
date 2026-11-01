---
title: "Evaluation Metrics for Generative Models"
chapter: 12
permalink: "/blog/cracking-the-ml-interview/evaluation-metrics/"
updated: 2026-11-01
description: "Classification has one right answer per example; generation does not."
---

Classification has one right answer per example; generation does not. There are many good translations of a sentence, many good captions for a photo and infinitely many realistic faces. This chapter covers the classic automatic metrics for generated images (FID and Inception Score) and generated text (BLEU, METEOR, ROUGE, CIDEr and SPICE): what each one measures, how it is computed, and where it fails. Classification metrics (precision, recall, F1, AUC) are covered in Chapter 6.

## 12.1 Intuition

**Why generation is hard to score.** A good automatic metric must agree with human judgment, be cheap and repeatable, and be hard to game. Every metric here trades these off. Most compare the output with human-written **references** (text) or with real data (images), so they reward resemblance to what was seen, not usefulness, truth or creativity.

**Image generation metrics.** Both use **Inception-v3**, a classifier pretrained on ImageNet, as a fixed judge.

- **Inception Score (IS)** (reference 33) runs each generated image through the classifier. Good samples should be **confident** (each image clearly shows one object, so its class distribution is peaked) and the set should be **diverse** (all images together cover many classes, so the average distribution is flat). IS is high when both hold. It never looks at real images, it is tied to ImageNet's 1,000 classes, and it cannot tell whether the variety within a class is realistic. Higher is better; the maximum is the number of classes.
- **FID (Fréchet Inception Distance)** (reference 32) passes both real and generated images through the network, takes the 2,048-dimensional features from its last pooling layer, fits a Gaussian to each set, and measures the distance between the two Gaussians. It compares to real data, captures both quality (means) and variety (covariances), and agrees with human judgment better than IS, so it is the standard. Lower is better.

<figure class="figure-pending"><div class="figure-pending__box">diagram coming soon</div><figcaption>how FID is computed · Heusel et al., 2017</figcaption></figure>

**Text generation metrics** compare a candidate text with one or more human references by overlapping words.

- **BLEU (Bilingual Evaluation Understudy)** (reference 34), the machine-translation standard since 2002, is **precision-oriented**: what share of the candidate's 1- to 4-word sequences (**n-grams**) appear in the references? Counts are **clipped** so repeating a correct word does not help, and a **brevity penalty** stops very short outputs from scoring high precision. It is designed for a whole test corpus; on single sentences it is noisy.
- **METEOR (Metric for Evaluation of Translation with Explicit ORdering)** (reference 35) aligns words between candidate and reference, matching exact words, then stems ("running" with "runs"), then synonyms from WordNet. It weights **recall** more than precision and penalizes fragmented, out-of-order matches. It correlates with human judgment better than BLEU at the sentence level.
- **ROUGE (Recall-Oriented Understudy for Gisting Evaluation)** (reference 36), the summarization standard, is **recall-oriented**: what share of the reference's n-grams appear in the candidate? **ROUGE-N** counts n-gram overlap (ROUGE-1 for words, ROUGE-2 for word pairs); **ROUGE-L** uses the **longest common subsequence (LCS)**, the longest sequence of words appearing in both texts in the same order but not necessarily adjacent. Reports usually give the F1 version.
- **CIDEr (Consensus-based Image Description Evaluation)** (reference 37) was built for image captioning with many references per image. It weights each n-gram by **TF-IDF (term frequency–inverse document frequency)**: n-grams common in this image's references count more, and n-grams common across all captions in the dataset ("a picture of") count less. It measures consensus with what most humans said.
- **SPICE (Semantic Propositional Image Caption Evaluation)** (reference 38) parses captions into a **scene graph** of objects, attributes and relations ("girl", "girl – young", "girl – standing on – sand") and computes an F1 score over these tuples. It checks meaning rather than wording, so "a young girl stands on the beach" and "a little girl standing on sand" match well, but it ignores fluency.

| Metric          | Task                | Compares                                   | Oriented to                   | Higher or lower is better                | Main blind spot                                                    |
| --------------- | ------------------- | ------------------------------------------ | ----------------------------- | ---------------------------------------- | ------------------------------------------------------------------ |
| Inception Score | Image generation    | Generated images only                      | Confidence and class variety  | Higher (max = number of classes)         | Ignores real data; ImageNet-specific; misses within-class collapse |
| FID             | Image generation    | Generated vs real feature statistics       | Quality and variety           | Lower (0 = identical)                    | Needs many samples; biased at small n; ImageNet features           |
| BLEU            | Machine translation | n-grams vs references                      | Precision                     | Higher (0 to 1, often reported ×100)     | Synonyms, paraphrases, meaning; noisy per sentence                 |
| METEOR          | Machine translation | Aligned words, stems, synonyms             | Recall-weighted F             | Higher (0 to 1)                          | Needs language resources; slower                                   |
| ROUGE           | Summarization       | n-grams or LCS vs references               | Recall (F1 commonly reported) | Higher (0 to 1)                          | Rewards copying; ignores fluency and factuality                    |
| CIDEr           | Image captioning    | TF-IDF-weighted n-grams vs many references | Consensus                     | Higher (CIDEr-D scaled to about 0 to 10) | Needs many references; still surface-level                         |
| SPICE           | Image captioning    | Scene-graph tuples                         | Semantic F1                   | Higher (0 to 1)                          | Ignores grammar and fluency; depends on the parser                 |

**All of them are proxies.** None checks that a translation is faithful, a summary is factually correct, or an image matches its prompt. Modern systems add learned metrics (BERTScore, COMET, CLIPScore), model-based judges, and above all human evaluation (12.4).

## 12.2 Core math

**Inception Score.** For generated images x, p(y | x) is the classifier's distribution over classes and p(y) is its average over all generated images. IS is the exponential of the average KL divergence between them:

$$
\text{IS} = \exp\Big(\mathbb{E}_{x}\big[D_{\text{KL}}\big(p(y \mid x)\,\|\,p(y)\big)\big]\Big), \qquad p(y) = \mathbb{E}_x\big[p(y \mid x)\big]
$$

The KL term is large when each p(y | x) is peaked (confident) and the marginal p(y) is spread out (diverse). The score lies between 1 and the number of classes K.

**FID.** With μ_r, Σ_r the mean and covariance of real-image features and μ_g, Σ_g those of generated images:

$$
\text{FID} = \|\mu_r - \mu_g\|_2^2 + \operatorname{Tr}\Big(\Sigma_r + \Sigma_g - 2\,\big(\Sigma_r \Sigma_g\big)^{1/2}\Big)
$$

The first term measures whether the generated features sit in the right place (quality); the trace term measures whether they have the right spread and correlations (variety). This is the squared **2-Wasserstein** (Fréchet) distance between two Gaussians. Estimating a 2,048 × 2,048 covariance needs many samples, and FID estimates are **biased upward** at small sample sizes, so only compare FIDs computed with the same number of samples and the same implementation (the standard is 50,000).

**BLEU.** For n = 1 to N (usually 4), the **modified n-gram precision** clips each candidate n-gram count at the most times it appears in any single reference:

$$
p_n = \frac{\sum_{g \in \text{n-grams}(c)} \min\big(\text{count}_c(g),\ \max_{r} \text{count}_r(g)\big)}{\sum_{g \in \text{n-grams}(c)} \text{count}_c(g)}
$$

BLEU combines them with a geometric mean and multiplies by the **brevity penalty** BP, where c is the candidate length and r the reference length (the closest one, for sentence BLEU; summed over the corpus, for corpus BLEU):

$$
\text{BLEU} = \text{BP}\cdot\exp\Big(\sum_{n=1}^{N} w_n \log p_n\Big), \qquad \text{BP} = \begin{cases} 1 & c > r \\ e^{\,1 - r/c} & c \le r \end{cases}, \qquad w_n = \tfrac{1}{N}
$$

The geometric mean means that if any p_n is zero, BLEU is zero, which is why sentence-level BLEU needs **smoothing** and why corpus-level BLEU, which pools counts over all sentences before dividing, is the standard.

**METEOR.** After aligning m unigrams between a candidate (length c) and a reference (length r), precision P = m/c and recall R = m/r are combined into a recall-weighted harmonic mean, then reduced by a **fragmentation penalty** based on the number of **chunks**, runs of matches that are adjacent and in the same order in both texts:

$$
F_{\text{mean}} = \frac{10\,P R}{R + 9P}, \qquad \text{Penalty} = 0.5\Big(\frac{\text{chunks}}{m}\Big)^{3}, \qquad \text{METEOR} = F_{\text{mean}}\,(1 - \text{Penalty})
$$

A perfectly ordered match has one chunk and almost no penalty; scattered matches can lose up to half the score. (Later versions tune these constants per language.)

**ROUGE.** ROUGE-N recall is the share of the reference's n-grams found in the candidate. ROUGE-L uses the length of the longest common subsequence:

$$
\text{ROUGE-N}_{\text{recall}} = \frac{\sum_{g} \min\big(\text{count}_c(g), \text{count}_r(g)\big)}{\sum_{g} \text{count}_r(g)}, \qquad R_{\text{LCS}} = \frac{\text{LCS}(c, r)}{|r|},\ \ P_{\text{LCS}} = \frac{\text{LCS}(c, r)}{|c|}
$$

The matching precision divides by the candidate's n-gram count, and the reported F1 is their harmonic mean. LCS is computed by dynamic programming in O(|c|·|r|) time.

**CIDEr.** Each sentence becomes a vector g^n of TF-IDF weights over its n-grams. For candidate c and m references s_j, CIDEr_n is the average cosine similarity, and CIDEr averages n = 1 to 4:

$$
\text{CIDEr}_n(c, S) = \frac{1}{m}\sum_{j=1}^{m} \frac{g^n(c)\cdot g^n(s_j)}{\|g^n(c)\|\,\|g^n(s_j)\|}, \qquad \text{CIDEr}(c, S) = \frac{1}{4}\sum_{n=1}^{4} \text{CIDEr}_n(c, S)
$$

The IDF part, log(number of images ÷ number of images whose references contain the n-gram), down-weights phrases that appear in captions of every image. The widely used **CIDEr-D** variant clips n-gram counts, adds a Gaussian penalty for length differences and multiplies by 10.

**SPICE.** With T(c) the set of scene-graph tuples (objects, object–attribute pairs, object–relation–object triples) parsed from the candidate and T(S) those from the references, matched with WordNet synonyms:

$$
P = \frac{|T(c) \otimes T(S)|}{|T(c)|}, \qquad R = \frac{|T(c) \otimes T(S)|}{|T(S)|}, \qquad \text{SPICE} = F_1 = \frac{2PR}{P + R}
$$

where ⊗ counts matching tuples.

## 12.3 Code

**BLEU from scratch**, checked against NLTK's implementation:

```python
import math
from collections import Counter

def ngrams(tokens, n):
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))

def bleu(candidate, references, max_n=4):
    """Sentence BLEU: uniform weights, brevity penalty, no smoothing."""
    log_p = 0.0
    for n in range(1, max_n + 1):
        cand = ngrams(candidate, n)
        max_ref = Counter()
        for ref in references:
            max_ref |= ngrams(ref, n)                 # max count of each n-gram in any reference
        clipped = sum(min(c, max_ref[g]) for g, c in cand.items())
        if clipped == 0:
            return 0.0                                # geometric mean collapses to zero
        log_p += math.log(clipped / sum(cand.values())) / max_n
    c = len(candidate)
    r = min((len(ref) for ref in references), key=lambda L: (abs(L - c), L))  # closest length
    bp = 1.0 if c > r else math.exp(1 - r / c)
    return bp * math.exp(log_p)

ref = "the cat is sitting on the mat".split()
print(bleu("the cat is sitting on a mat".split(), [ref]))   # 0.6435, same as nltk sentence_bleu
```

| Candidate (reference: "the cat is sitting on the mat") | BLEU-4 | Exact-match METEOR |
| ------------------------------------------------------ | ------ | ------------------ |
| the cat is sitting on the mat                          | 1.000  | 0.999              |
| the cat is sitting on a mat                            | 0.644  | 0.841              |
| on the mat the cat is sitting                          | 0.595  | 0.818              |
| the cat is                                             | 0.000  | 0.446              |
| the the the the the the the                            | 0.000  | n/a                |
| a kitten rests upon the rug                            | 0.000  | n/a                |

Three lessons. A 3-word output has no 4-grams, so unsmoothed BLEU-4 is exactly 0. Clipping matters: "the" × 7 has unigram precision 7/7 without clipping but 2/7 with it, because the reference contains "the" only twice. And a correct paraphrase ("a kitten rests upon the rug") scores 0 on BLEU because it shares almost no wording; METEOR's synonym matching or a learned metric would recognize it.

**ROUGE-N and ROUGE-L from scratch**, matching Google's `rouge-score` package exactly:

```python
def rouge_n(candidate, reference, n):
    c, r = ngrams(candidate, n), ngrams(reference, n)
    overlap = sum((c & r).values())                    # clipped matches
    p, rec = overlap / sum(c.values()), overlap / sum(r.values())
    return p, rec, (0 if overlap == 0 else 2 * p * rec / (p + rec))

def lcs_len(a, b):
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            dp[i][j] = dp[i-1][j-1] + 1 if a[i-1] == b[j-1] else max(dp[i-1][j], dp[i][j-1])
    return dp[-1][-1]

def rouge_l(candidate, reference):
    l = lcs_len(candidate, reference)
    p, r = l / len(candidate), l / len(reference)
    return p, r, (0 if l == 0 else 2 * p * r / (p + r))

reference = "police killed the gunman after a long standoff".split()
candidate = "the gunman was killed by police after a standoff".split()
# ROUGE-1 (P, R, F) = (0.778, 0.875, 0.824)
# ROUGE-2           = (0.250, 0.286, 0.267)
# ROUGE-L           = (0.556, 0.625, 0.588)
```

Seven of the reference's eight words appear in the summary (ROUGE-1 recall 0.875), but reordering the sentence breaks most word pairs (ROUGE-2 recall 0.286). For real evaluations use the packages (`sacrebleu` for BLEU, which also standardizes tokenization; `rouge-score` for ROUGE), because tokenization and smoothing choices change the numbers.

**FID from scratch** on synthetic 64-dimensional "features", so the answer can be checked against theory:

```python
import numpy as np
from scipy import linalg

def fid(mu1, s1, mu2, s2):
    covmean = linalg.sqrtm(s1 @ s2).real              # matrix square root
    return float(np.sum((mu1 - mu2) ** 2) + np.trace(s1 + s2 - 2 * covmean))

rng = np.random.default_rng(0)
d = 64
real = rng.normal(0, 1, size=(5000, d))
stats = lambda x: (x.mean(0), np.cov(x, rowvar=False))

for name, fake in [("same distribution", rng.normal(0, 1, size=(5000, d))),
                   ("mean shifted by 0.5", rng.normal(0.5, 1, size=(5000, d))),
                   ("too little variety (std 0.5)", rng.normal(0, 0.5, size=(5000, d)))]:
    print(name, fid(*stats(real), *stats(fake)))
# same distribution             0.47
# mean shifted by 0.5          16.43   (theory: 64 x 0.25 = 16)
# too little variety (std 0.5) 16.40   (theory: 64 x (1 + 0.25 - 2 x 0.5) = 16)
```

FID penalizes wrong content (a shifted mean) and missing variety (a collapsed spread) alike. Its small-sample bias is large: two independent samples from the _same_ distribution score FID 22.8 with 100 samples each, 4.6 with 500, 1.1 with 2,000 and 0.2 with 10,000.

**Inception Score from class probabilities:**

```python
def inception_score(p_yx, eps=1e-12):
    p_y = p_yx.mean(0, keepdims=True)                  # marginal class distribution
    kl = (p_yx * (np.log(p_yx + eps) - np.log(p_y + eps))).sum(1)
    return float(np.exp(kl.mean()))

K = 10
confident_diverse = np.eye(K)[rng.integers(0, K, 1000)] * 0.99 + 0.01 / K
confident_one_class = np.eye(K)[np.zeros(1000, dtype=int)] * 0.99 + 0.01 / K
uncertain = np.full((1000, K), 1 / K)
# IS: 9.26 (confident, all 10 classes), 1.00 (one class only), 1.00 (uncertain about every image)
```

IS reaches nearly its maximum of 10 only when images are both confidently classified and spread across classes; a generator that produces only one class (**mode collapse**), or blurry images the classifier cannot recognize, scores 1.

## 12.4 In system design

- **Offline metrics are a filter, not a verdict.** Use BLEU, ROUGE or FID to catch regressions cheaply on every model change, then confirm real improvements with human evaluation and online tests (Chapter 14). Report the metric's version and settings (sacreBLEU signatures exist for this reason).
- **Learned metrics correlate better with humans.** **BERTScore** matches candidate and reference tokens by contextual-embedding similarity, so paraphrases score well; **COMET** is a model trained on human translation ratings; **CLIPScore** measures how well an image matches its text prompt, which FID cannot. For images, FID computed with stronger feature extractors, and precision/recall for generative models (separating quality from coverage), address some of FID's blind spots.
- **Evaluating LLM outputs.** Open-ended answers have no single reference, so teams combine several methods:
  - **Task-specific checks:** exact match or unit tests for code and math, citation checks for RAG answers, and schema validation for structured outputs.
  - **LLM-as-a-judge:** a strong model grades answers against a rubric, or picks the better of two answers. It is cheap and scalable, but has known biases toward longer answers, the first-presented option and its own style. Calibrate it against human labels, swap answer order, and use clear rubrics.
  - **Human evaluation** on a fixed, versioned set of prompts that represents real traffic, with side-by-side preference ratings and agreement checks between raters.
  - **Online signals:** thumbs up or down, edits, retries and task completion.
- **Golden sets and slices.** Keep a curated evaluation set per use case (languages, domains, difficult inputs), versioned like code, and report metrics per slice; an average can hide a regression in one language or customer segment.
- **Guard against gaming and contamination.** Models tuned hard on one metric learn its quirks (ROUGE rewards copying; BLEU rewards matching reference phrasing). For LLMs, check that public benchmark data did not leak into training data before trusting scores.
- **Metric compute cost.** FID needs tens of thousands of generated samples, which is expensive for slow diffusion models; teams track a cheaper proxy (FID on 5,000 or 10,000 samples) during training and the full FID at release, never comparing across the two.

## 12.5 Interview Q&A

**Q1. What does FID measure, and why is it preferred over Inception Score?** FID fits Gaussians to Inception features of real and generated images and computes the Fréchet distance between them, capturing both how realistic samples are (means) and how varied (covariances). Unlike IS, it compares against real data, detects within-class mode collapse, and correlates better with human judgment. Lower is better.

**Q2. What are the limitations of FID?** It assumes Gaussian features; it depends on an ImageNet-trained network, so it can mislead on domains far from ImageNet (medical scans, faces); it is biased at small sample sizes, so numbers are only comparable with the same sample count and implementation; and it says nothing about whether an image matches its text prompt.

**Q3. Explain how Inception Score works and how it can be fooled.** IS = exp(average KL divergence between each image's class distribution and the marginal class distribution). It rewards confident, class-diverse samples. A generator that produces one perfect image per ImageNet class would score near the maximum while having no variety within classes, and IS ignores whether images look like the real training data.

**Q4. Walk through the BLEU calculation.** Compute modified (clipped) n-gram precisions for n = 1 to 4, take their geometric mean, and multiply by a brevity penalty exp(1 − r/c) when the candidate is shorter than the reference. Clipping stops repetition from inflating precision, and the brevity penalty stops a short, precise output from scoring high.

**Q5. Why is BLEU precision-based and ROUGE recall-based?** In translation, adding words that are not in any reference is the main error, so BLEU checks that what you produced is supported (precision), with a brevity penalty to avoid dropping content. In summarization, missing key content is the main error, so ROUGE checks how much of the reference you covered (recall).

**Q6. How does METEOR improve on BLEU?** It matches stems and synonyms, not just exact words; it weights recall more heavily; and it penalizes fragmented word order through its chunk penalty. It correlates better with human judgments, especially at the sentence level.

**Q7. What is ROUGE-L?** ROUGE based on the longest common subsequence: the longest sequence of words appearing in both texts in the same order, not necessarily adjacent. It rewards preserving sentence-level word order without requiring exact consecutive matches.

**Q8. How do CIDEr and SPICE differ for image captioning?** CIDEr compares TF-IDF-weighted n-grams with many reference captions, rewarding consensus wording and down-weighting generic phrases. SPICE parses captions into scene graphs of objects, attributes and relations and scores the overlap, so it rewards correct meaning even with different wording but ignores fluency.

**Q9. Your summarization model's ROUGE went up but users say summaries got worse. What happened?** The model may be copying more source text verbatim, getting longer, or including reference-like phrasing without improving coherence or factual accuracy, all of which ROUGE rewards. Check length, copy rate and factual consistency, run human or LLM-judge evaluations, and examine examples where ROUGE and human ratings disagree.

**Q10. How would you evaluate a chatbot built on an LLM?** Define what good means for the product (correctness, groundedness in sources, helpfulness, safety, tone), build a versioned evaluation set from real and adversarial prompts, and score it with automatic checks where possible, a calibrated LLM judge for subjective criteria, and periodic human review. Then confirm with online metrics and A/B tests.

## 12.6 Cheat sheet and pitfalls

**Remember:**

- IS = exp(E\[KL(p(y|x) ‖ p(y))\]); higher is better; ignores real data.
- FID = ‖μ_r − μ_g‖² + Tr(Σ_r + Σ_g − 2(Σ_rΣ_g)^½); lower is better; use 50,000 samples and one implementation.
- BLEU: clipped n-gram precision, geometric mean over n = 1 to 4, brevity penalty; corpus-level.
- METEOR: F = 10PR/(R + 9P) times (1 − 0.5(chunks/matches)³); stems and synonyms.
- ROUGE: recall of reference n-grams (ROUGE-N) or LCS (ROUGE-L); report F1.
- CIDEr: TF-IDF n-gram cosine against many references. SPICE: F1 over scene-graph tuples.
- All are proxies; confirm with humans, learned metrics and online tests.

**Pitfalls that cost candidates:**

- Comparing FIDs computed with different sample sizes or implementations.
- Quoting sentence-level BLEU as if it were reliable, or comparing BLEU across tokenizations.
- Treating a higher ROUGE as proof of a better or more factual summary.
- Saying FID "measures image quality" without mentioning variety and its reliance on ImageNet features.
- Using only reference-based metrics for open-ended LLM outputs.

## 12.7 Further reading

<ol start="32">
  <li><a href="https://en.wikipedia.org/wiki/Fr%C3%A9chet_inception_distance">Fréchet inception distance</a> (Wikipedia)</li>
  <li><a href="https://en.wikipedia.org/wiki/Inception_score">Inception score</a> (Wikipedia)</li>
  <li><a href="https://en.wikipedia.org/wiki/BLEU">BLEU</a> (Wikipedia)</li>
  <li><a href="https://en.wikipedia.org/wiki/METEOR">METEOR</a> (Wikipedia)</li>
  <li><a href="https://en.wikipedia.org/wiki/ROUGE_(metric)">ROUGE (metric)</a> (Wikipedia)</li>
  <li>Vedantam et al., <a href="https://arxiv.org/pdf/1411.5726.pdf">CIDEr: Consensus-based Image Description Evaluation</a> (CVPR 2015)</li>
  <li>Anderson et al., <a href="https://arxiv.org/pdf/1607.08822.pdf">SPICE: Semantic Propositional Image Caption Evaluation</a> (ECCV 2016)</li>
</ol>
