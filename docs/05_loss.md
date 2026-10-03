# Quality-weighted language-model loss for SEA-Rater

This note explains the proposed objective, how it changes model updates, why document length and normalization matter, and how to implement and evaluate it. The numerical values below are teaching examples, not experimental findings.

## 1. The main idea

Ordinary language-model training gives every valid target token equal weight in the loss. Our proposal gives each token a weight determined by the quality score of the document it belongs to.

Higher-quality text receives a larger coefficient in the training objective. Lower-quality text still contributes, but with a smaller coefficient per token.

**Our goal is to prioritize learning from useful text. We do not aim to make the model deliberately bad at predicting low-quality text.**

There are two different models involved:

- **SEA-Rater:** predicts document-quality scores.
- **Language model (LM):** learns to predict the next token. The scores modify its training loss; they do not become its prediction targets.

Keep the rater frozen during this baseline experiment. Its own training objective does not change.

## 2. Where this proposal comes from

This particular formulation was proposed in our discussion as a baseline for SEA-Rater. It was not copied from a specific paper.

| Component | Origin/status |
|---|---|
| Next-token cross-entropy | Standard LM training objective |
| Multiplying a training loss by an importance weight | Established loss-reweighting principle |
| Dividing a weighted sum by the sum of weights | Standard mathematical weighted average |
| Using SEA-Rater document scores as token-loss weights | Proposed application for this project |
| Mapping a 0–5 score to weight using `0.5 + score / 5` | Illustrative baseline suggested in our discussion; not a published optimum |
| Normalizing average weights within each language | Proposed control for language-level weight differences |

The formula itself should not be claimed as a new loss-reweighting invention. The research contribution would need to come from the quality signals, multilingual design, and experimental evidence. Related papers mentioned earlier should not be cited as sources of this exact equation or mapping without checking their actual objectives.

## 3. Ordinary next-token training

### 3.1 One token produces one prediction loss

Given preceding tokens, the model predicts a probability distribution over the vocabulary. For the actual next token:

$$
\ell_{d,t}(\theta)=-\log P_\theta(x_{d,t}\mid x_{d,<t}).
$$

| Symbol | Meaning |
|---|---|
| $d$ | Document |
| $t$ | Target-token position |
| $\theta$ | LM parameters |
| $x_{d,t}$ | Actual token to predict |
| $x_{d,<t}$ | Preceding context in the document |
| $\ell_{d,t}$ | Cross-entropy for that prediction |

For example, a correct-token probability of 0.8 gives loss about 0.223; probability 0.1 gives loss about 2.303, using natural logarithms.

High loss means the model assigns low probability to the observed target. It can reflect unfamiliar information, unusual vocabulary, ambiguity, or noise. **It is not a direct measure of document quality or knowledge gained.**

The notation above assumes within-document context. If packed documents can attend across boundaries, the actual conditioning context includes those permitted earlier tokens. Keep that attention policy identical across experiments.

### 3.2 Many token losses become one training objective

Let $T_d$ contain the valid prediction positions from document $d$ in the effective batch, and let $n_d=|T_d|$. Standard token averaging is:

$$
L_{\mathrm{ordinary}}=
\frac{\sum_d\sum_{t\in T_d}\ell_{d,t}}{\sum_d n_d}.
$$

There is no division by token count in an individual token's cross-entropy. The division happens when combining the losses from many prediction positions.

In common implementations, this averaging is built into the cross-entropy function. PyTorch's default `reduction="mean"`, with integer targets and no class weights, averages over non-ignored targets. Do not divide an already averaged loss by the token count again.

“Valid tokens” means targets that actually participate in the loss after shifting, truncation, and masking. Padding must be marked as ignored; an attention mask alone does not necessarily exclude a position from the loss.

## 4. The proposed quality-weighted loss

Assign a fixed positive weight $w_d$ to document $d$. Every valid target token from that document inherits the same weight:

$$
\boxed{
L_{\mathrm{quality}}(\theta)=
\frac{\sum_d w_d\sum_{t\in T_d}\ell_{d,t}(\theta)}
{\sum_d w_d n_d}
}
$$

| Component | Meaning |
|---|---|
| $w_d$ | Document-quality weight, independent of LM parameters |
| $w_d\ell_{d,t}$ | Weighted loss of one token |
| Numerator | Sum of weighted token losses |
| Denominator | Sum of weights over valid target tokens |

If every weight is one, this exactly reduces to ordinary token-averaged loss for the same batch and masking.

Each token has its own prediction loss. In this proposal, tokens do **not** receive individually estimated quality scores: they inherit their parent document's weight.

## 5. Complete training flow

### Step 1: Score each document

Run the frozen rater on each training document. Store its ID, language, quality scores, and final loss weight. A document weight is metadata used during LM training.

### Step 2: Tokenize and form the batch

Tokenize using the LM's tokenizer. Carry the document weight alongside each token. If multiple documents are packed into one sequence, preserve the origin of every target token.

The rater and LM can use different tokenizers: the score belongs to the document and is subsequently attached to its LM tokens.

### Step 3: Run the forward pass

The LM produces logits for next-token prediction. During ordinary causal-LM training, a causal mask allows many positions to be processed in parallel while preventing access to future tokens.

Training typically uses the actual preceding tokens as context, rather than generating an entire document one token at a time before calculating loss.

### Step 4: Compute individual token losses

Align each prediction with its next-token target. Compute cross-entropy with no reduction, obtaining one loss per target position. Exclude ignored targets.

### Step 5: Apply quality weights and normalize

Multiply each valid token loss by its inherited document weight. Sum those values and divide by the total weight of valid target tokens.

### Step 6: Backpropagate

Backpropagation differentiates the scalar objective with respect to the model parameters. Although the loss is one number, its computation retains the connections to all token predictions and their weights.

For fixed weights:

$$
\nabla_\theta L_{\mathrm{quality}}=
\frac{\sum_d w_d\sum_{t\in T_d}\nabla_\theta\ell_{d,t}}
{\sum_d w_d n_d}.
$$

Thus, quality weights scale the gradients before they are combined.

### Step 7: Update parameters

For basic gradient descent:

$$
\theta_{\mathrm{new}}=\theta_{\mathrm{old}}-\eta\nabla_\theta L.
$$

The gradient points toward increasing loss locally, so gradient descent subtracts it. Adam and similar optimizers further transform the combined gradient using running statistics.

The optimizer updates **parameters**, not the loss value itself. The next forward pass calculates a new loss using the updated parameters. A finite update need not reduce every document's loss.

### Step 8: Repeat

Clear gradients at the appropriate optimizer-step boundary and process subsequent batches. With gradient accumulation, several microbatches contribute to one optimizer update.

## 6. Worked example: A has 100 tokens, B has 900

Assume both documents are represented in the same effective batch:

| Document | Valid tokens | Quality weight per token | Mean token loss |
|---|---:|---:|---:|
| A: higher quality | 100 | 3 | 2 |
| B: lower quality | 900 | 1 | 4 |

The weights 3 and 1 are selected for easy arithmetic. They are not the same numerical mapping as the later 0.5–1.5 baseline. The losses 2 and 4 are hypothetical; low quality does not necessarily imply high loss.

### Ordinary loss

$$
L_{\mathrm{ordinary}}=\frac{100(2)+900(4)}{1000}=3.8.
$$

Define $L_A$ and $L_B$ as each document's mean token loss. Then:

$$
L_{\mathrm{ordinary}}=0.1L_A+0.9L_B.
$$

### Quality-weighted loss

$$
L_{\mathrm{quality}}=\frac{3(100)(2)+1(900)(4)}{3(100)+1(900)}
=\frac{4200}{1200}=3.5.
$$

Equivalently:

$$
L_{\mathrm{quality}}=0.25L_A+0.75L_B.
$$

Every A token has coefficient $3/1200$; every B token has coefficient $1/1200$. A token therefore receives three times the coefficient of a B token.

However, B contributes nine times as many tokens. Its document-average loss still receives a larger total coefficient. Quality weighting increases A's share from 10% to 25%; it does not automatically make a short document dominate a much longer document.

**The decrease from 3.8 to 3.5 does not demonstrate learning.** The model predictions have not changed; we only changed the objective used to aggregate them.

## 7. Why the denominator matters

The denominator makes the objective a weighted average. It controls its scale while preserving the relative token weights.

Using the same example:

| Quantity | Weights 3 and 1 | Weights 30 and 10 |
|---|---:|---:|
| Weighted loss sum | 4,200 | 42,000 |
| Sum of token weights | 1,200 | 12,000 |
| Weighted average | 3.5 | 3.5 |

Multiplying all weights by ten does not change their relative importance. The normalized objective stays the same. Without normalization, both the loss sum and its gradient become ten times larger.

Duplicating all examples inside the same batch similarly doubles the numerator and denominator, leaving their ratio unchanged. This statement concerns the same objective evaluation, not running twice as many optimizer steps.

The denominator is not mathematically mandatory. These are distinct choices:

| Objective | Scale behavior |
|---|---|
| $\sum_{d,t}w_d\ell_{d,t}$ | Depends on token count and overall weight scale |
| $\frac{1}{N}\sum_{d,t}w_d\ell_{d,t}$ | Depends on mean weight, with $N$ valid tokens |
| $\frac{\sum_{d,t}w_d\ell_{d,t}}{\sum_d w_dn_d}$ | Normalizes by total token weight |

Our proposal uses the third choice as a baseline. Batchwise normalization also means relative coefficients depend on batch composition. A ratio computed separately in each small batch is not generally an unbiased estimate of the globally normalized corpus objective. This is a design consideration, not proof that one normalization is universally best.

## 8. How gradient directions combine

For equal-length documents with quality weights 3 and 1:

$$
g=0.75g_A+0.25g_B,
$$

where $g_A$ and $g_B$ are gradients of their mean token losses.

For one parameter:

| $g_A$ | $g_B$ | Combined gradient | Interpretation |
|---:|---:|---:|---|
| +2 | +1 | +1.75 | Both support the same gradient direction |
| +1 | −1 | +0.5 | Partial cancellation |
| +1 | −3 | 0 | Complete cancellation |

If these numbers are gradients, subtract the result times the learning rate. If describing suggested parameter changes instead, be explicit that the negative-gradient sign has already been applied.

For unequal lengths of 100 and 900 in our worked example, use $g=0.25g_A+0.75g_B$ instead.

Coefficients are not percentages of knowledge learned. Gradient sizes and directions differ across documents and parameters. A larger quality weight does not guarantee that a document dominates every update.

## 9. Document length: two different objectives

Length already matters in ordinary token averaging because longer documents supply more prediction targets. Quality weighting does not introduce that fact.

### Option 1: Weight each token by document quality

$$
L_{\mathrm{token}}=\frac{\sum_d w_dn_dL_d}{\sum_d w_dn_d}.
$$

Each higher-quality token receives more weight. Longer documents contribute more targets. This is our proposed first experiment because uniform weights recover the existing token-averaged baseline.

### Option 2: Weight each document average by quality

$$
L_{\mathrm{document}}=\frac{\sum_d w_dL_d}{\sum_d w_d},
\qquad L_d=\frac{1}{n_d}\sum_t\ell_{d,t}.
$$

Each document's total coefficient depends on quality, regardless of its length. In the A/B example this gives $0.75L_A+0.25L_B$.

At the token level, Option 2 effectively introduces a factor of $1/n_d$. Thus, among equally rated documents, each token in a short document receives greater weight than each token in a long document.

Neither is universally superior. Option 1 isolates quality weighting relative to standard token averaging. If testing Option 2, include an unweighted document-average baseline to separate the effects of quality and length normalization. For split documents, explicitly define whether the unit is a complete document or a chunk; chunk averaging is not automatically document averaging.

## 10. Constructing positive quality weights

Assume predicted dimension scores are bounded to 0–5. If your actual scale differs, adjust the mapping.

First combine dimensions:

$$
q_d=\sum_{k=1}^{K}a_ks_{d,k},\qquad a_k\geq0,\qquad\sum_ka_k=1.
$$

- $s_{d,k}$: predicted score for dimension $k$.
- $a_k$: weight used to combine dimensions.
- $q_d$: combined quality score.

Then an illustrative positive mapping is:

$$
u_d=0.5+\frac{q_d}{5}.
$$

| Quality score | Weight |
|---:|---:|
| 0 | 0.5 |
| 2.5 | 1.0 |
| 5 | 1.5 |

This is a moderate baseline, not an optimized mapping. The dimension weights $a_k$ and the loss weights $u_d$ have different roles. Test individual dimensions as well as combinations; averaging five dimensions is not automatically the best definition of useful training text.

Do not use raw z-scores as loss weights. A below-average score has a negative z-score. Multiplying a loss by a negative weight reverses that token's learning signal and can reward increasing its prediction loss. A z-score can be transformed into a positive weight, but the transformation introduces another design choice.

### Optional language normalization

To avoid larger average quality scores automatically assigning a language more total weight, estimate on training data:

$$
m_\lambda=\frac{\sum_{d:\operatorname{lang}(d)=\lambda}n_du_d}
{\sum_{d:\operatorname{lang}(d)=\lambda}n_d},
\qquad w_d=\frac{u_d}{m_{\operatorname{lang}(d)}}.
$$

This gives mean token weight one for each language on the data used to estimate these statistics. It does not guarantee equal gradient magnitudes, fix rater bias, or ensure identical language contributions in every batch. Control language token budgets separately. Final normalized weights need not remain in the initial 0.5–1.5 range.

## 11. Implementation details

The following illustrates one batch on one process. It assumes unshifted labels and weights aligned with input-token positions, with ignored labels equal to −100.

```python
import torch
import torch.nn.functional as F

def weighted_causal_loss(logits, labels, token_weights):
    # logits: [batch, sequence_length, vocabulary_size]
    # labels: [batch, sequence_length]
    # token_weights: same shape as labels; parent-document weights
    predictions = logits[:, :-1, :].contiguous()
    targets = labels[:, 1:].contiguous()
    weights = token_weights[:, 1:].detach().float()

    valid = targets.ne(-100)
    if not valid.any().item():
        raise ValueError("No valid target tokens in this batch")
    valid_weights = weights[valid]
    if not torch.isfinite(valid_weights).all().item():
        raise ValueError("Weights must be finite")
    if (valid_weights <= 0).any().item():
        raise ValueError("This baseline requires positive weights")

    per_token = F.cross_entropy(
        predictions.float().reshape(-1, predictions.size(-1)),
        targets.reshape(-1),
        reduction="none",
        ignore_index=-100,
    ).reshape_as(targets)

    weights = weights.masked_fill(~valid, 0.0)
    numerator = (per_token * weights).sum()
    denominator = weights.sum()
    return numerator / denominator
```

This is explanatory code, not a complete distributed training integration. Expanding all logits to float32 can increase memory use; production implementations may use fused loss kernels.

Important implementation points:

1. The weight belongs to the **target being predicted**, so shift weights consistently with labels.
2. Avoid applying a causal shift twice if the surrounding training code already shifts targets.
3. Do not use `CrossEntropyLoss(weight=...)` for document weights; that argument weights vocabulary classes.
4. Keep EOS handling, cross-document attention, and boundary masking identical across comparisons.
5. Do not multiply an already averaged model loss by one average document weight; that loses the token-to-document correspondence.

### Multiple microbatches and devices

For microbatch $j$, define weighted loss sum $S_j$ and total token weight $Z_j$. The intended effective-batch objective is:

$$
L=\frac{\sum_jS_j}{\sum_jZ_j}.
$$

It is generally different from averaging $S_j/Z_j$ across microbatches. Normalize against the effective-batch total, accounting for the framework's gradient-accumulation and distributed reduction conventions.

Because weights and masks are fixed, the effective denominator can be counted before backpropagation. Backpropagating each $S_j/Z_{\mathrm{total}}$ and summing gradients gives the intended result on one process, provided the trainer does not introduce an additional averaging factor. Distributed wrappers may average gradients across devices, requiring corresponding scale handling.

### Why one document per separately normalized update is problematic

For one document:

$$
\frac{w_d\sum_t\ell_{d,t}}{w_dn_d}=L_d.
$$

Its weight cancels. The same happens whenever all valid tokens in a separately normalized batch have the same weight. Accumulating several documents works only if normalization combines their weighted sums over the effective batch. Sequential single-document updates do not eliminate conflicting learning signals.

## 12. Weighted loss versus weighted sampling

| Method | What changes? | Practical consequence |
|---|---|---|
| Quality-weighted loss | Coefficient of each processed token loss | Can retain the exact baseline training stream |
| Quality-weighted sampling | Frequency with which examples are processed | Changes exposure, repetition, and coverage |
| Top-score filtering | Which documents are eligible | Removes some documents from the training pool |

Loss weighting still spends compute on low-weight tokens. Sampling can spend less compute on them. Do not initially apply the same preference through both sampling and loss weighting: this compounds it.

## 13. Evaluation plan

Start with the same random training stream under ordinary loss and quality-weighted token loss. Match model initialization, tokenization, document order, context length, packing, optimizer settings, language token budgets, and processed-token count.

Evaluate both models with **ordinary unweighted loss on identical held-out data**. Weighted training loss is not directly comparable to ordinary training loss.

Report:

- Held-out token-averaged loss and perplexity per language.
- Macro-average language loss, clearly distinguished from pooling all languages' tokens.
- Suitable downstream tasks, with an evaluation setup appropriate to the model's capabilities.
- Independent human-rated quality subsets, alongside the original full held-out evaluation.
- Runtime, unique-data coverage, length/domain distributions, and variation across matched seeds.

Use validation data to choose score combinations and weighting strength. Reserve an untouched test set for final reporting. Ensure diagnostic human-rated evaluation data is independent of LM training and, when evaluating rater-defined selection, preferably held out from rater training/tuning as well.

Useful controls include individual-dimension weights, shuffled weights within language, and document-average objectives with and without quality weights. If length or domain correlates strongly with quality, stratified shuffling or matched analyses can help disentangle those effects.

## 14. A short explanation you can say aloud

“A language model predicts the next token and gets a cross-entropy loss for each prediction. Normal training averages those token losses equally. We use our document-quality rater to assign a positive weight to each document, and every target token inherits that weight. We multiply each token loss by its weight, sum the results, and divide by the total token weight. Backpropagation therefore gives higher-quality tokens larger coefficients in the combined gradient. Longer documents still contribute more tokens, just as in ordinary training. We test whether this preference improves unweighted held-out performance and downstream tasks.”

If asked why we divide:

“We divide by the total token weight to form a weighted average. Changing all weights from 3 and 1 to 30 and 10 should not change the objective, because the relative preference is identical.”

If asked whether B dominates because it is longer:

“B can have a larger document-level coefficient because it provides many more tokens. Our proposal prioritizes each good token; it does not guarantee that every good document outweighs every longer, lower-quality document.”

If asked whether this formula comes from a paper:

“It is a standard weighted-average construction applied to token cross-entropy. This particular SEA-Rater mapping is our proposed experimental baseline, not a formula copied from a specific paper or an established optimum.”

## 15. Implementation references

These sources support the reduction behavior and causal-LM implementation details, not the novelty or effectiveness of the proposed SEA-Rater method:

- [PyTorch CrossEntropyLoss documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html): reductions, ignored targets, and class weights.
- [Hugging Face Transformers loss utilities](https://github.com/huggingface/transformers/blob/main/src/transformers/loss/loss_utils.py): causal target alignment and mean or count-normalized summed cross-entropy. The main branch can change; check the version installed in your training environment.
