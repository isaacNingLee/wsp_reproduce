# Reading Weights, Fixing Failures: Weight-Space Programming with Astra, DeepSeek, GLM, and Qwen

*Results snapshot: November 3, 2026*

What can an LLM learn by inspecting another model’s weights? And when it finds a failure, can it repair the weights directly?

We investigated these questions using Astra and DeepSeek V4.1 Flash 1017, GLM 5.3 Flash, and Qwen 3.8 27B. We evaluated a released Qwen weight edit, ran new editing experiments, and extended the idea to vision transformers. The results are encouraging—but the failures are just as informative as the successes.

**The clearest result is that targeted edits can recover useful behavior when a model already has the relevant capability.** On our math subset, the released edit raises Qwen3.5-0.8B from **5.0% to 54.5%** on MATH-500. A new local Astra edit reaches **64.0%**. Completed edits from GLM 5.3 Flash, Qwen 3.8 27B, and DeepSeek V4.1 Flash 1017 reach **53.5%, 24.5%, and 22.5%**, respectively, on the same MATH-500 subset. All four editors identify our anonymous vision checkpoints correctly. But repairing fine-grained bird classification is less reliable: an edit that reaches **96.7% on practice images** achieves only **60.0% on held-out images**, and fails our overall-accuracy constraint.

In an additional merging experiment, Astra and DeepSeek both reach **80.6% mean accuracy** across CIFAR-100 and CUB, versus **79.4% for plain averaging**. Qwen reaches **80.5%**; GLM finds a useful candidate but saves the wrong mathematical object.

The saved editor reasoning traces help explain this gap. They show useful mathematical deductions, sensible calibration strategies, and also a validation procedure that inadvertently reuses information from the example it claims to hold out.

## What we tested

This investigation follows [*Sparks of Weight-Space Programming*](https://rikiyat.github.io/blog/weight-space-programming/), which studies LLMs inspecting and editing network weights directly. Its Qwen experiment reports the following results:

| Published Qwen3.5-0.8B condition | MATH-500 (%) | GSM8K (%) | IFEval (%) | MATH answers reaching the token limit (%) |
|---|---:|---:|---:|---:|
| Base | 7.9 | 13.8 | 21.0 | 93.0 |
| Edited, 10 forward passes | 12.3 | 25.7 | 49.0 | 87.2 |
| Edited, 25 forward passes | 14.1 | 25.1 | 39.9 | 86.4 |
| Edited, 50 forward passes | 48.1 | 67.1 | 62.3 | 33.1 |

These are the authors’ reported values, not measurements from our runs. Their evaluation uses full benchmarks and repeated samples; our math evaluation uses 100 problems per benchmark and two samples per problem. [Source: original post, Experiment 8](https://rikiyat.github.io/blog/weight-space-programming/).

Our work has four parts:

1. **Read:** identify ImageNet, CIFAR-100, and CUB vision checkpoints from middle-layer weights, without forward passes.
2. **Repair:** edit a vision classifier to improve a confused class pair, while limiting the overall accuracy loss.
3. **Improve:** evaluate the released Qwen edit, compare controls, and test locally produced math edits.
4. **Merge:** combine two separately fine-tuned vision backbones while retaining both task-specific heads—a task not evaluated in the original post.

The local vision experiments are extensions of the idea, not reproductions of the blog’s tasks. There is no published CIFAR/CUB score to put beside them.

### The editors

| Editor model | Reasoning effort used in these experiments |
|---|---|
| DeepSeek V4.1 Flash 1017 | Maximum |
| GLM 5.3 Flash | Maximum |
| Qwen 3.8 27B | Extra high |
| GPT-6 Astra | Separate execution; effort was not normalized against the other editors |

These are the model names recorded for the experiments. Throughout this post, **the editor** is the model proposing weight changes; **Qwen3.5-0.8B** is the smaller model whose math performance is evaluated. The math tables measure the edited Qwen3.5-0.8B, not the editor’s own ability to solve math problems.

We compile one collection of saved experiments, using corrected scoring records where available.

## Experiment 1: Which dataset produced these weights?

We gave each editor anonymous checkpoints A, B, and C. All three use ViT-B/16: one is an ImageNet model, and the other two are full fine-tunes of that same model on CIFAR-100 and CUB-200.

The editor sees only encoder blocks 4–7. It cannot inspect the input embedding or classification head, see images, or run the network. The prompt does disclose the candidate datasets, their sizes, and the fine-tuning recipe.

| Editor | Trials with all three datasets identified correctly | Individual checkpoints correctly identified |
|---|---:|---:|
| DeepSeek V4.1 Flash 1017 | 2/2 | 6/6 |
| GLM 5.3 Flash | 2/2 | 6/6 |
| Qwen 3.8 27B | 2/2 | 6/6 |
| GPT-6 Astra | 6/6 | 18/18 |

A random complete assignment has probability 1/6. However, these trials permute the **same three checkpoints**; they are not independent training seeds. The table demonstrates consistency under renaming, not a measured success rate across arbitrary networks. Earlier shortcut runs are excluded.

### Why it works: training leaves measurable traces

In GLM trial 0, the model identifies B as the parent checkpoint using the geometry of the weight differences. The two differences from B have cosine similarity **0.041**, while the alternative candidate origins give much larger angles. A has approximately **1.46 times** C’s residual drift from B.

GLM then combines this measurement with the disclosed training recipe. CIFAR-100 has more optimizer steps and a larger input-resolution mismatch with ImageNet. Greater drift and stronger radial shrinkage support assigning A to CIFAR-100 and C to CUB.

This is a useful inference about training history. It does not establish that GLM has decoded bird features or recovered the visual computation of an individual head.

Astra finds an especially interpretable signal in attention’s **key biases**. For query $q_i$ and token $x_j$,

$$
q_i^\top(W_Kx_j+b_K)=q_i^\top W_Kx_j+q_i^\top b_K.
$$

The bias contributes the same offset to every entry in an attention row, so softmax cancels it. In exact arithmetic it has no task gradient. If AdamW applies decay to this parameter, its systematic shrinkage can reveal cumulative optimizer exposure.

In Astra trial 0, the pooled key-bias scales relative to checkpoint C are **0.960674 for A** and **0.981537 for B**. Their inferred decay-exposure ratio is **2.153**, close to the **2.080** ratio of disclosed backbone step counts. The assignment is correct.

This mechanism also identifies a limitation of our task: a model can succeed by reading a training fingerprint. Matching optimizer exposure, withholding recipe details, and introducing independently trained checkpoints would make a stronger test of behavioral interpretation.

## Experiment 2: Can we repair a confused vision classifier?

For each dataset, we split test images by class into practice and held-out halves, using seed 0. We select the most-confused class pair using only the practice half.

The editor receives weights, practice labels, and up to ten forward calls returning logits and requested class-token features. It may compute statistics and save weight replacements. Gradients, optimizers, iterative loss fitting, and regression fitting are prohibited by the task prompt.

Success requires both:

- Held-out accuracy on images from the target pair increases.
- Held-out overall accuracy falls by no more than **0.3 percentage points**.

“Pair accuracy” still uses the full multiclass prediction: a maple image classified as pine is wrong, even though pine is outside the target pair. Also, the permitted use of labeled means and covariances makes these data-dependent edits; “no gradient descent” does not mean “no statistical estimation.”

### CIFAR-100: maple versus oak

There are 100 pair images in each half, and 5,000 total images per half. Baseline pair accuracy is **70.0% on practice** and **77.0% on holdout**; held-out overall accuracy is **88.46%**.

| Editor | Practice accuracy (%) | Unseen-image accuracy (%) | Target-pair gain (percentage points) | Overall change (percentage points) | Weight entries changed | Success? |
|---|---:|---:|---:|---:|---:|---|
| DeepSeek V4.1 Flash 1017 | 79.0 | 83.0 | +6.0 | −0.100 | 1,538 | Yes |
| GLM 5.3 Flash | 80.0 | 79.0 | +2.0 | −0.060 | 3,076 | Yes |
| Qwen 3.8 27B | 75.0 | 79.0 | +2.0 | −0.060 | 1,538 | Yes |
| GPT-6 Astra | 79.0 | 79.0 | +2.0 | −0.280 | 1,538 | Yes |

The two accuracy columns refer to the target pair; overall change includes every class. All changes are measured on unseen images. All four pass. These are small absolute gains: two to six additional correct pair images. Astra’s overall change is close to the permitted boundary.

### A successful trace: preserve a useful direction and recalibrate it

DeepSeek starts from the linear classifier equation,

$$
z_c(x)=w_c^\top x+b_c.
$$

Its saved trace considers rotating the maple–oak separation direction, but reports that covariance-based rotations perform worse under cross-validation than a simpler correction.

The final edit adds the same feature direction $g$ to both target rows:

$$
w_{47}'=w_{47}+g,\qquad w_{52}'=w_{52}+g.
$$

The shared shift changes the pair’s competition with other classes while preserving the **weight-vector difference** $w_{47}-w_{52}$. A separate bias adjustment recalibrates the maple–oak threshold.

This decomposition matters. Some “pair errors” are trees losing to other classes, so strengthening shared tree evidence can help without replacing an already useful species-separation direction. The held-out score rises from **77/100 to 83/100**. That supports the effectiveness of this edit on this split; it does not prove every step of the model’s explanation.

### CUB: California Gull versus Western Gull

CUB is much less forgiving. There are only **15 images per species** in each half. Baseline pair accuracy is **43.3% on practice** and **53.3% on holdout**. Overall held-out accuracy is **83.225%** across 2,915 images.

| Editor / trial | Practice accuracy (%) | Unseen-image accuracy (%) | Target-pair gain (percentage points) | Overall change (percentage points) | Weight entries changed | Success? |
|---|---:|---:|---:|---:|---:|---|
| DeepSeek V4.1 Flash 1017 | 90.0 | 53.3 | 0.0 | −0.377 | 1,538 | No |
| GLM 5.3 Flash | 96.7 | 60.0 | +6.7 | −0.309 | 1,538 | No |
| Qwen 3.8 27B | 70.0 | 63.3 | +10.0 | −0.103 | 1,538 | Yes |
| GPT-6 Astra, trial 0, rescored | 63.3 | 60.0 | +6.7 | +0.034 | 8,853,363 | Yes |
| GPT-6 Astra, trial 1 | 66.7 | 40.0 | −13.3 | −0.206 | 1,538 | No |

Again, accuracy refers to the target pair; overall change includes every class, measured on unseen images. GLM misses the overall-loss threshold narrowly: nine net errors over 2,915 images are a **0.309-point** loss. DeepSeek’s large practice gain does not transfer. Astra’s second trial actively harms the pair. We report both Astra trials rather than selecting the successful one.

Astra trial 0 originally scored as an empty edit because of a nested relative-path launcher bug. Its existing `result_rescored.json` evaluates the recovered saved tensors. The correction changes the outcome to success, but the edit is broad: it modifies late encoder blocks as well as the head.

### A failed trace: a convincing validation result with leaked information

GLM’s CUB trace is particularly revealing. It builds a partially whitened class-mean direction from a covariance eigendecomposition:

$$
v\propto U\,\mathrm{diag}(\lambda^{-p})U^\top(\mu_{58}-\mu_{65}).
$$

The selected edit uses $p=0.2$ and symmetric head adjustments. During development, the model reports **zero leave-one-out errors out of 30** for nearby whitening settings and treats that as evidence of generalization.

But the executed validation code computes `U` and `ev` **before** the leave-one-out loop, using all practice images. Within each fold it recomputes the two means, while retaining that full-data covariance basis. The held-out image therefore still influences the transformation used to classify it.

The reasoning trace explicitly notices the issue, describing reuse of the full-data covariance as “slight leakage,” then continues to rely on the result. This is not a clean leave-one-out estimate.

There is a second warning sign: a 768-dimensional covariance estimated from a small collection of bird images is rank-deficient. Very small eigenvalues are clamped to `1e-9`; inverse powers amplify directions with little measured within-class variation. Such directions can separate the practice set remarkably well without representing a stable species distinction.

The measured outcome matches that concern: **29/30 correct on practice becomes 18/30 on holdout**. The code establishes validation leakage; small-sample estimation and adaptive candidate selection are plausible additional contributors. We cannot isolate their causal contributions from this run alone.

### Why changing only two rows can still hurt other classes

Qwen’s CUB explanation claims that preserving the sum of the two target logits preserves their collective pull on other classes. The guarantee is too strong.

For symmetric changes $z_a'=z_a+\delta$ and $z_b'=z_b-\delta$, the sum stays fixed, but their maximum need not. For example, pair logits **(3, 3)** become **(5, 1)**: an unrelated class with logit **4** loses its previous win.

Qwen’s edit succeeds under our aggregate criterion, yet the held-out overall score still drops slightly. This is a useful distinction between a successful intervention and a fully correct explanation of its safety.

## Experiment 3: Making Qwen finish its math answers

Our math evaluation uses fixed seed-0 subsets of **100 MATH-500 problems and 100 GSM8K problems**, with **two samples each**. All thinking-mode rows below use temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.5, and a 32,768-token generation cap.

Accuracy is per generated answer, not pass@2. “Cut” means generation reached its token limit; it is an observable proxy for nontermination, not a direct loop detector.

All rows evaluate **Qwen3.5-0.8B after the named intervention**. All four local editors now have completed MATH-500 and GSM8K evaluations. The three newly completed runs each had a nominal budget of ten generation/activation-read calls; Astra had 25. These budgets are not identical amounts of computation: calls can inspect different numbers of examples, and failed calls are refunded.

| Who produced the weight edit? | MATH-500 accuracy (%) | GSM8K accuracy (%) | Evaluation status |
|---|---:|---:|---|
| No editor: original Qwen3.5-0.8B | 5.0 | 12.5 | Completed |
| GPT-6 Astra: authors’ publicly released edit | 54.5 | 75.0 | Completed local evaluation of the released checkpoint |
| GPT-6 Astra: our new edit | 64.0 | 72.5 | Completed; 25-call budget |
| DeepSeek V4.1 Flash 1017: newly completed edit | 22.5 | 35.0 | Completed; 10-call budget |
| GLM 5.3 Flash | 53.5 | 56.0 | Completed; 10-call budget |
| Qwen 3.8 27B | 24.5 | 40.0 | Completed; 10-call budget |

Each accuracy is calculated from 200 generated answers to 100 problems. Relative to the original model, the new DeepSeek, GLM, and Qwen edits improve MATH-500 by **17.5, 48.5, and 19.5 percentage points**, respectively, and GSM8K by **22.5, 43.5, and 27.5 points**.

**Historical result, kept separate:** an earlier DeepSeek-associated checkpoint scored **41.0% on MATH-500 and 57.5% on GSM8K**. That evaluation was associated with an archived run affected by a harness bug. It is not the newly completed four-layer edit above. Both evaluations recorded the same checkpoint-directory path at different times, so the path alone cannot identify the checkpoint version. The old records remain in the evidence archive; they are not substituted for the new run’s scores or treated as a second clean trial.

**Termination behavior for the evaluated checkpoints:**

| Evaluated version of Qwen3.5-0.8B | MATH-500 answers reaching the token limit (%) | GSM8K answers reaching the token limit (%) |
|---|---:|---:|
| Original model | 95.0 | 90.0 |
| Authors’ released Astra edit | 32.0 | 3.5 |
| Our new Astra edit | 7.5 | 0.0 |
| DeepSeek V4.1 Flash 1017: newly completed edit | 78.5 | 64.0 |
| GLM 5.3 Flash edit | 21.0 | 3.0 |
| Qwen 3.8 27B edit | 53.5 | 29.0 |

GLM’s edit substantially reduces truncation on both benchmarks. DeepSeek and Qwen also improve on the base model, but many answers still reach the cap. Lower truncation alone does not guarantee higher accuracy: GLM has fewer truncated GSM8K answers than the released Astra edit, yet its GSM8K accuracy is **56.0% versus 75.0%**. Reaching an answer and reaching the right answer are separate requirements.

**Control experiments: what happens if we alter the released edit?**

| Intervention applied to the original Qwen3.5-0.8B | MATH-500 accuracy (%) | GSM8K accuracy (%) |
|---|---:|---:|
| Remove the selected weight-direction components completely, rather than reversing and amplifying them | 13.5 | 27.5 |
| Replace the selected directions with random directions, preserving the total edit magnitude; one random seed | 9.5 | 16.0 |
| Apply the selected directions to 64 randomly selected columns per layer; one random seed | 6.0 | 14.0 |

Our base and released-edit scores show the same broad improvement as the published results. Numerical differences should not be read as contradictions or new full-benchmark records: the sample counts, selected problems, and local evaluation implementation differ. We did not evaluate IFEval.

The local Astra edit has the highest MATH point estimate among these checkpoints. GLM comes close to the released edit on MATH-500 (**53.5% versus 54.5%**) but falls further behind on GSM8K (**56.0% versus 75.0%**). These are single editor runs on small subsets, with unequal budgets, not evidence of a general editor ranking. Two answers to the same problem are also correlated, so treating all 200 samples as independent would overstate precision.

### What changed in the weights?

Our audit finds that the released checkpoint differs in the MLP output projections of **layers 15 and 19**, with **64 affected columns per layer**. The numerical deltas are almost rank one: over **99.999%** of their squared singular-value energy lies in the leading direction.

The nominal support is 131,072 entries; the audit counts **130,188 numerically changed entries**. Finite-precision storage can leave some entries in an edited support unchanged. These are different counting conventions.

The reconstruction fits the update

$$
W_{:,S}'\approx W_{:,S}-c\,u(u^\top W_{:,S}),\qquad c\approx6.
$$

That coefficient matters. At $c=1$, this operation removes the projection along $u$. At $c\approx6$, it reverses and amplifies that component, approximately multiplying it by −5. The saved reconstruction matches over **99.8%** of entries in each full tensor bit-for-bit, with a maximum absolute discrepancy of **0.00048828125**.

The controls show why a generic “small edit” explanation is insufficient. Literal projection removal helps somewhat, but much less than the released edit. A norm-matched random direction also performs much worse. Random columns perform poorly, although that control has a smaller total perturbation norm, so it does not isolate column choice from edit magnitude. Only one random seed was evaluated.

### Four local edits, four ways of steering the model

The completed results let us compare the actual submitted edits. Layer numbers below are zero-indexed, and changed-entry counts come from each saved editing result.

| Editor | MLP output projections edited | Numerically changed weight entries | Intended intervention |
|---|---|---:|---|
| GPT-6 Astra | Layer 23 | 2,157,312 | Rank-two update favoring thinking termination and suppressing repeated “Wait” transitions |
| DeepSeek V4.1 Flash 1017 | Layers 16, 18, 20, 22 | 9,576,531 | Rank-one update per layer, steering from repeated reconsideration toward computation |
| GLM 5.3 Flash | One column each in layers 9, 13, 17, 21 | 4,094 | Persistent residual-stream bias toward the thinking-close token |
| Qwen 3.8 27B | Layer 23 | 2,457,706 | Rank-one close-versus-“Wait” direction, gated by recent numeric content |

Edit size does not predict the result here. GLM changes far fewer entries than the other local editors and outperforms DeepSeek and Qwen on both benchmarks. This comparison is descriptive; the runs do not control for direction choice, layer choice, search effort, or edit magnitude.

Astra’s practice generation calls used an 8,192-token cap. Its final 128-problem check described in the editor response is a reserve from the training-data practice pool, not the MATH-500 or GSM8K test evaluation. The main table uses separate evaluation at 32,768 tokens for every checkpoint.

### A successful math trace: a small bias can unlock completion

In its [completed math-edit transcript](logs/math-edit-completed/glm_t0/transcript.jsonl), GLM searches for MLP features that remain positive across many tokens. It selects one column in each of four MLP output matrices and adds a direction based on the final-normalization-scaled embedding of the thinking-close token. When an approximately constant positive feature multiplies that column, it adds an approximately persistent residual-stream shift.

The intended effect is to make closing the thinking block more likely without changing the embeddings or output layer. The direction is motivated by the output geometry, but the resulting probability change remains state-dependent because of normalization and subsequent computation; it is not an exact constant addition to the final logit.

The trace reports a useful dose-response on its 128-problem practice set: accuracy rises from **7.8%** to **50.0%** at the selected steering strength, then falls to **28.1%** at a stronger setting. The independent benchmark results—**53.5% MATH-500 and 56.0% GSM8K**—support a real benefit beyond that practice set. The four selected columns have a nominal support of 4,096 entries; 4,094 entries differ numerically after storage.

There is an execution caveat: GLM rerouted generation to another GPU after resource failures. This changed environment plumbing, while the submitted artifact remained a weight edit and the final benchmark was scored separately. We report the run as completed, without claiming that its infrastructure was identical to every other run.

### Why more steering is not always better

DeepSeek’s [new trace](logs/math-edit-completed/deepseek_t0/transcript.jsonl) takes a broader approach. It contrasts activations near arithmetic, numbers, and answer construction with activations near reconsideration phrases, then writes that direction into four MLPs. The result improves both benchmarks, but leaves **78.5% of MATH answers** and **64.0% of GSM8K answers** truncated.

The trace also tests a stronger perturbation: on its 96-problem practice set, increasing the relative steering strength from **12% to 20%** of the estimated residual norm reduces accuracy from **27.1% to 15.6%**. The submitted edit uses the weaker setting. These practice generations used a 4,096-token cap after an 8,192-token attempt encountered memory limits, so their absolute scores should not be compared directly with the final 32,768-token table.

Qwen’s [completed run](logs/math-edit-completed/qwen38_t0/transcript.jsonl) uses a more conditional idea: steer toward closing the thinking block and away from “Wait” after the model has recently written a number. The trace itself identifies the risk: writing a number does not mean the calculation is finished or correct, and some “Wait” transitions precede useful self-correction.

On its practice set, Qwen reports that the number of responses finishing rises from **11 to 73**, while accuracy among those finished responses falls from **100% to 61.6%**. Those are different selected subsets, so this is not a paired estimate of lost reasoning ability. It does illustrate the tradeoff: encouraging completion admits many more answers, including wrong ones. The final scores improve to **24.5% MATH-500 and 40.0% GSM8K**, but remain well below GLM and Astra.

Taken together, the traces support an intervention with a limited useful range: too little steering leaves the model looping; too much can interrupt needed reasoning. Fewer truncated answers are helpful, but they are not a sufficient optimization objective.

### Archived implementation failure versus a completed retry

The earlier archived Qwen attempt correctly identified repetition but never saved its proposed edit: a variable shadowed the weights dictionary. Its result records zero changed weights. The new completed Qwen run is a separate attempt and does save an edit, with the evaluated scores above. We retain the archived failure as an implementation lesson rather than using it to describe the current result.

The completed runs’ budget files record **10 charged calls for DeepSeek, 8 for GLM, and 5 for Qwen**, under a maximum of ten each. Failed attempts also appear in the logs, so log-entry counts are not the same as charged calls. Tool-call budgets, editor turns, and computation spent inspecting cached activations should not be conflated.

### Does the edit add mathematical ability?

A useful control is to disable thinking:

| Local condition | MATH-500 (%) | GSM8K (%) |
|---|---:|---:|
| Base, thinking disabled | 52.5 | 62.0 |
| Released edit, thinking disabled | 51.5 | 57.5 |

The unedited model already answers many problems correctly in this mode. The released edit does not improve these point estimates. The no-thinking profile also changes temperature, top-p, and the token cap, so this is not an isolated toggle experiment.

Nevertheless, together with the large decline in truncated thinking-mode answers, these observations support a narrower interpretation: **the edits recover access to existing competence by improving the generation trajectory**. They do not establish that the model learned new mathematical procedures.

## Experiment 4: Merging two fine-tuned models into one backbone

Can the editor combine two adaptations without retraining? This experiment extends the investigation to a task not tested in the original post.

We start with the shared ImageNet ViT-B/16 parent and its CIFAR-100 and CUB fine-tunes. The editor must produce **one shared backbone**. The original CIFAR-100 and CUB classification heads stay fixed, and the appropriate head is used for each dataset. This is not a single unified classifier or an ensemble of two backbones.

Each editor has a budget of **eight practice evaluations**. Practice contains 5,000 CIFAR-100 images and 2,875 CUB images; scoring uses 5,000 and 2,915 held-out images, respectively. The score is the **unweighted mean of the two dataset accuracies**, not accuracy pooled over all images. The logged prompt incorrectly says 2,895 CUB practice images; the saved split contains 2,875.

### Results: similar scores, slightly different blends

| Method or editor | CIFAR-100 held-out accuracy (%) | CUB held-out accuracy (%) | Mean held-out accuracy (%) |
|---|---:|---:|---:|
| Plain equal averaging of the two fine-tuned backbones | 85.32 | 73.52 | 79.42 |
| Task arithmetic, shared strength selected on practice | 84.76 | 74.96 | 79.86 |
| GPT-6 Astra | 84.36 | 76.78 | **80.57** |
| DeepSeek V4.1 Flash 1017 | 83.06 | 78.04 | **80.55** |
| Qwen 3.8 27B | 83.04 | 78.01 | **80.53** |
| GLM 5.3 Flash: incorrectly saved backbone | 1.00 | 0.65 | **0.83 — submission failure** |
| Reference: retain both separate task-specific backbones | 88.46 | 83.22 | 85.84 |

The final row uses two different backbones and therefore does not satisfy the merging task. It is a reference for how much accuracy the shared model gives up, not a proven upper bound on what a merged model could achieve.

Astra and DeepSeek both round to **80.6%**, while Qwen rounds to **80.5%**. Their exact mean scores differ by less than **0.05 percentage points**. This supports “similar performance on this experiment,” not a general model ranking. The small differences between DeepSeek and Qwen also occur despite nominally identical formulas; their code uses different floating-point arithmetic, and these runs do not isolate numerical effects from evaluation variation.

The best result improves on plain averaging by **1.15 percentage points**, or on tuned task arithmetic by **0.71 points**. The gap from plain averaging to the separate-backbone reference is **6.42 points**, so the best merge recovers about **18%** of that gap. Most of the gap remains.

### What the successful editors actually saved

Let $P$ be the parent backbone, $C$ the CIFAR fine-tune, and $B$ the bird fine-tune. The task vectors are $Delta_C=C-P$ and $Delta_B=B-P$.

Plain averaging uses $(C+B)/2$. The task-arithmetic baseline searches a single shared strength in

$$
P+lambda(Delta_C+Delta_B).
$$

It tests strengths 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, and 1.0, selecting **0.4 on practice accuracy** before reporting the held-out score.

DeepSeek and Qwen instead save an asymmetric blend:

$$
P+0.40Delta_C+0.60Delta_B=0.40C+0.60B.
$$

Astra favors the bird adaptation too, but also shrinks the update toward the parent:

$$
P+0.40Delta_C+0.50Delta_B=0.10P+0.40C+0.50B.
$$

Thus, the three successful editors converge on a **similar CUB-favoring family**, not exactly the same merge. A read-only [audit of the saved weights](results/merging/saved-weight-audit.json) checks all 150 backbone tensors per submission. Each successful artifact matches its stated formula with a maximum absolute discrepancy below **0.00000006**.

### Why an asymmetric blend helps

The [DeepSeek](logs/vit_merge/deepseek_t0/transcript.jsonl), [Qwen](logs/vit_merge/qwen38_t0/transcript.jsonl), and [Astra](logs/vit_merge/astra_t0/transcript.jsonl) logs share a useful observation: the CIFAR task vector is roughly **1.5 times larger** than the CUB task vector across much of the network. Equal scalar weights therefore do not mean equal-sized perturbations. The practice results also show that equal averaging sacrifices substantially more bird accuracy than CIFAR accuracy.

Those measurements motivate reducing the CIFAR contribution. The successful merges trade some CIFAR accuracy for a larger CUB gain, improving the mean. Their candidate evaluations—not the norm comparison alone—establish that this trade is beneficial on this split.

More elaborate proposals do not necessarily help. Qwen’s full sum of both task vectors reaches only **59.88% mean practice accuracy**, and its sign-based conflict-resolution candidate reaches **54.18%**. Near-orthogonal parameter changes do not guarantee independent functional effects: both updates still pass through the same nonlinear network and fixed heads. Astra likewise tests head-aware corrections, depth-varying coefficients, and sparse sign-consistent updates, then keeps the simpler parent-shrunk blend.

This is a useful result for an editor that can inspect weights and choose among candidates. It is not evidence that the LLM discovered a new merging algorithm. A direct search over two independent mixing coefficients is an important missing baseline; outperforming equal averaging and a shared-strength task-arithmetic grid does not establish superiority over well-tuned asymmetric interpolation.

### GLM’s failure: a good candidate is not a valid saved model

The [GLM trace](logs/vit_merge/glm_t0/transcript.jsonl) evaluates a correct 40% CIFAR / 60% CUB blend at **80.37% mean practice accuracy**. Its final prose claims that this candidate was submitted.

The save code tells a different story. It first constructs the blended **delta**, then subtracts the parent before passing that tensor dictionary to a helper that adds the parent. The two parent terms cancel. What reaches disk is

$$
0.40Delta_C+0.60Delta_B,
$$

rather than the full backbone $P+0.40Delta_C+0.60Delta_B$.

The saved-weight audit confirms this across every backbone tensor. The resulting artifact scores **0.83% mean held-out accuracy**, close to chance. Shape and finite-value checks pass because a delta has exactly the same tensor shapes as a model; they cannot detect the missing parent.

This failure is especially informative because the candidate-selection reasoning was broadly sound. The broken link was between the evaluated candidate and the submitted artifact. A robust workflow must reload the saved backbone and evaluate that exact object before declaring success. We report the actual failed submission and do not silently replace it with the candidate described in the final answer.

### Limits of this merging experiment

There is only one checkpoint pair and one split. The trials have separate run directories; earlier open-filesystem attempts are excluded. However, the sandbox’s allowlist includes the baseline JSON containing held-out baseline scores and cache/split files containing held-out information. DeepSeek also inspects evaluator internals and searches for data in its trace. The intended scoring split is held out from the normal practice-evaluation tool, but these artifacts do **not** establish strict evaluation blindness or complete run isolation.

The recorded scores and saved-weight formulas remain reproducible evidence about the submitted models. A stronger comparison should expose only practice data and practice baseline summaries, isolate scoring in a separate process, repeat the experiment across seeds, and include a tuned asymmetric-blend baseline. We omit cost comparisons because the experiment artifacts contain token usage but no verified billing record.

## What the traces teach us

These experiments suggest three conditions under which direct weight editing can be useful.

First, **there must be a readable signal**. Shared checkpoint ancestry and optimizer-induced shrinkage provide unusually strong clues. They are easier to interpret than an arbitrary distributed behavior.

Second, **the intervention should match the failure**. A calibration problem can respond to a small head edit. Persistent nontermination can respond to a directional change that encourages finishing. Neither observation implies that every classification or reasoning error has a similarly compact repair.

Third, **the validation procedure must survive scrutiny**. GLM’s CUB trace contains sophisticated linear algebra and a confident generalization story, yet its leave-one-out estimate retains information from the omitted example. Qwen’s sum-preserving argument misses changes in multiclass competition. An agent’s explanation is evidence about its proposed mechanism, not a substitute for held-out measurement.

The next experiments should repeat these tasks across training seeds, rebuild every data-dependent statistic inside validation folds, compare against fixed calibration baselines, and evaluate math edits on full benchmarks. For vision, cached features also permit many candidate head evaluations after one forward call: the forward-call budget does not measure the full amount of adaptive search.

Our evidence supports targeted repair as a promising use of weight-space programming. Its present limits are equally concrete: small validation sets, training fingerprints, collateral changes in predictions, and failures to turn a proposed intervention into a correctly saved artifact.

## Evidence and reproduction notes

This article compiles existing artifacts; it does not report newly rerun model evaluations. Scores come from structured evaluation records, not the editors’ claimed final-answer scores. Trace passages above are brief quotations or explanatory summaries; raw reasoning is not required to read this post.

The companion [results snapshot](results-snapshot.json) contains public-facing numerical records, including all three newly completed math runs. The [published logs](logs/README.md) and [per-answer evaluations](results/math/) retain the evidence behind the tables. The [reproduction package](code/README.md) contains portable experiment code, setup instructions, and tests.

The base revision is `2fc06364715b967f1860aea9cf38778875588b17`; the evaluated released-edit revision is `d1671a178705bcab7644806d23c535517f35723b`. Earlier shortcut, truncated, killed, and harness-failure runs are not counted as clean completed trials. Some exported `reasoning.md` files are incomplete; the full saved transcript is used when available.

To publish, keep this article, `results-snapshot.json`, and the `code/` folder together. The reproduction instructions distinguish publicly downloadable math checkpoints from vision checkpoints and new local edits that must be supplied separately.
