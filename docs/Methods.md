Below is a structured “analysis protocol” you can almost drop into the Methods / Supplement of a Q1 paper.
I assume you have:

* `class_embeddings_trajectory.pt`
* `probe_features_epoch_XXXX.npz` for a probe set, with `{sample_id, class_id, epoch, timestep, layer_name, branch, feature}`
* `clustering_metrics.pt` (silhouette, Calinski–Harabasz, Davies–Bouldin)
* Per-epoch/per-class FID (or you will compute it)
* Downstream classification experiments (real vs generated/augmented).

I organize the plan into **latent-space geometry** and **task performance**, then explicitly connect them.

---

## 0. Dataset and training bias summary

**Goal**
Quantify the intrinsic class imbalance of PathMNIST and make clear that all analyses are conditioned on this realism constraint.

**Input data**

* PathMNIST label counts (train/val/test).([GitHub][1])
* Training log: effective batch composition if you use any re-weighting / oversampling.

**Output**

* Table: class label, absolute counts, percentage, minority/majority flag.
* Bar plot: class frequencies.

**Rationale**
Reviewers need to see that the dataset itself is imbalanced, that you *do not* artificially rebalance it during training (unless you do), and that any per-class performance disparities must be interpreted in this context.

**Interpretation**
This sets the baseline: e.g. “class 8 constitutes 3.2 % of training data vs 27.6 % for class 0”. Any later claim that the model handles minorities must be read against these frequencies.

---

## 1. Global structure of the learned class embedding space

### 1.1 Static geometry at convergence

**Goal**
Show that the **parametric class embeddings** learned by the CFG-DDPM form a structured space (not arbitrary codes), despite label imbalance.

**Input**

* Final epoch slice from `class_embeddings_trajectory.pt`: matrix (E \in\mathbb{R}^{C\times d}).

**Output**

1. PCA / UMAP 2-D scatter plot of class embeddings (one point per class).
2. Heatmap of pairwise cosine similarities (\cos(e_i,e_j)).
3. Table of summary stats: intra-minority / intra-majority / inter-class distances.

**Rationale**

* Classifier-free guidance mixes conditional and unconditional scores using these embeddings as the only semantic signal.([arXiv][2])
* If embeddings form a roughly spherical, well-separated configuration (no degenerate collapse, no extreme outliers for minority classes), it supports the claim that the generative model alone (no external classifier) carries usable class semantics.

**Interpretation**

* “Nice” result: classes form a roughly isotropic simplex-like configuration; majority vs minority classes do not form separate subspaces.
* Problematic signs: extremely small angles between some classes (embedding collapse), or minorities pushed to the periphery with much larger norms.

---

### 1.2 Trajectory and stability of class embeddings across epochs

**Goal**
Demonstrate that class embeddings converge to a stable geometry (sample and structural stability in Mabadeje & Pyrcz’s sense).([arXiv][3])

**Input**

* `class_embeddings_trajectory.pt` across all epochs.

**Output**

1. For each class (c): line plot of (|e_c^{(t)} - e_c^{(T)}|_2) vs epoch (t) (drift towards final embedding).
2. Heatmap of pairwise cosine similarity between epochs for each class (epoch × epoch).
3. Optional: Procrustes-aligned PCA plots of embeddings at early/mid/late epochs (so reviewers see qualitatively that the geometry “locks in”).

**Rationale**

* Stability of parametric embeddings is part of the **minimum acceptance criteria** for using a latent space downstream.([arXiv][3])
* Large late-epoch drifts or frequent “swaps” of relative positions could indicate that the model hasn’t converged to a robust class representation, making CFG brittle.

**Interpretation**

* You want curves that flatten after some epoch and similarity heatmaps that show high similarity among late epochs.
* You can phrase this as: “After epoch 30, the average embedding drift falls below X and the class embedding geometry is structurally stable.”

---

## 2. Feature-space geometry of the U-Net (probe features)

This is where you use the `probe_features_epoch_XXXX.npz` and the anisotropy visualizations.

### 2.1 Class-conditional geometry at selected layers and timesteps

**Goal**
Show that **mid-block and other key layers** learn class-structured manifolds, particularly in the conditional branch, and that this structure is consistent across epochs and timesteps.

**Input**

* For each chosen epoch (t\in{t_{\text{early}},t_{\text{mid}},t_{\text{late}}}), timestep (k\in{100,500,900}) and layer (ℓ) (e.g. early block, mid block, late block):

  * Conditional probe features (h(x,y,k,ℓ)) for the balanced probe set, from `probe_features_epoch_XXXX.npz` (branch=`cond`).

**Output**

1. 2-D PCA plots with KDE + 95 % contour + local anisotropy ellipses and points coloured by class (your existing grids).
2. For each (layer, timestep, epoch):

   * cluster validity scores (silhouette, Calinski–Harabasz, Davies–Bouldin) from `clustering_metrics.pt`.([ScienceDirect][4])
   * table of (\beta_{\text{local}}) per dense lobe.

**Rationale**

* This is essentially applying Mabadeje’s **latent stability workflow** to your diffusion U-Net: geometric anisotropy, cluster quality, and their evolution across epochs.([arXiv][3])
* At high noise timesteps the conditional branch should carve **class-dependent basins in feature space**, while the unconditional branch remains more isotropic, consistent with the theory of classifier-free guidance.([arXiv][2])

**Interpretation (what you want to show reviewers)**

* At high noise (t=900):

  * uncond features form a single mixed Gaussian-like blob,
  * cond features form multiple islands with high class purity and moderate anisotropy → **the generative model itself organises noise space into class-specific basins**, no external classifier needed.
* At mid/low noise: cond vs uncond closer, but cond still exhibits class-structured anisotropy.
* Cluster validity indices improve and stabilise over epochs, evidencing **sample and structural stability** of the latent representations.

---

### 2.2 Alignment of class embeddings with feature clusters

**Goal**
Connect the **parametric class embeddings** to the **empirical feature manifolds**.

**Input**

* Final epoch class embeddings (E).
* Final epoch probe features for some (timestep, layer, branch).

**Output**

1. PCA computed on probe features only, then project both features and class embeddings into that PCA space.
2. Plot: feature cloud + per-class centroid + class embedding (e.g. centroid as triangle, embedding as star).
3. Table: distance between embedding (e_c) and feature centroid (\mu_c) per class.

**Rationale**

* This addresses a potential reviewer concern: “Do your class embeddings actually correspond to the feature geometry, or are they just arbitrary vectors that happen to make FID good?”
* Small distances (|e_c-\mu_c|) indicate that embeddings are anchored near the centre of their class manifolds.

**Interpretation**

* You want to report that for all classes (including minorities) the embedding lies close to the empirical centroid of its features (possibly within one standard deviation).
* Outlier classes would warrant discussion and could correlate with bad per-class FID or classification performance later.

---

### 2.3 Minority vs majority classes: intra- vs inter-class spread

**Goal**
Test whether minority classes are geometrically disadvantaged (e.g. collapsed manifolds, high overlap with majority classes) compared to majority classes.

**Input**

* Final-epoch probe features for selected (timestep, layer, cond branch).

**Output**

1. For each class (c):

   * intra-class covariance determinant (\det(\Sigma_c)) or log-volume in latent space,
   * average distance to nearest neighbours (intra-class) vs nearest neighbours of other classes.
2. Bar plots:

   * intra-class spread vs class frequency,
   * inter-class margin (distance between class centroids) vs class frequency.

**Rationale**

* On an imbalanced dataset, a reviewer will ask whether minority classes are underrepresented in latent space.
* Wide intra-class spread with good inter-class margins for minorities suggests the model has *not* collapsed them.

**Interpretation**

* You want to show that minority classes have intra-class spreads and centroid distances comparable to majority classes (within some factor), reinforcing that imbalance does not strongly distort latent geometry.

---

## 3. Generative quality and per-class FID

### 3.1 Global and per-class FID (with caveats)

**Goal**
Quantify generative quality overall and per class, while being honest about FID’s limitations, especially for small per-class sample sizes and diffusion models.([arXiv][5])

**Input**

* Generated samples conditioned on each class (fixed number per class, e.g. ≥10k if feasible).
* Real PathMNIST test images per class.
* Inception (or a more medical-appropriate encoder if you choose to follow “Rethinking FID”).

**Output**

1. Table:

   * overall FID,
   * per-class FID with bootstrap confidence intervals.
2. Optional: per-class KID or PRD (precision/recall for generative models), to address known weaknesses of FID.([NeurIPS Proceedings][6])

**Rationale**

* FID is expected by reviewers, but recent work shows it can be biased, especially for diffusion models and small sample sizes.([NeurIPS Proceedings][6])
* Per-class FID shows whether minority classes suffer mode collapse or over-smoothing.

**Interpretation**

* Report both the global FID and the distribution of per-class FIDs.
* Highlight that minority classes do **not** systematically have worse FID than majority classes, or if they do, connect that to latent geometry (Section 2.3) and classification.

---

## 4. Downstream classification performance

### 4.1 Baselines on PathMNIST

**Goal**
Demonstrate that your generative model, used for augmentation, competes with or improves upon standard discriminative baselines on PathMNIST.([Nature][7])

**Input**

* Real PathMNIST train/val/test splits.
* Discriminative architectures (e.g. ResNet-18, wider CNN) trained:

  1. on real data only (baseline),
  2. with traditional augmentations only,
  3. with traditional aug + ccDDPM synthetic augmentation (balanced per class),
  4. optionally on synthetic-only for sanity.

**Output**

* Table: global accuracy, macro-F1, balanced accuracy, AUC (macro & micro) for each training regime.
* Per-class sensitivity/specificity/F1.

**Rationale**

* Shows that your model is not just “good FID” but **useful** for downstream tasks.
* Macro-F1 and balanced accuracy directly address class imbalance.

**Interpretation**

* Ideally, the augmented models (3) improve minority-class sensitivity and macro-F1 compared to (1) and (2), without hurting majority classes much.
* If synthetic-only (4) is competitive, it supports that ccDDPM alone learns meaningful class structure.

---

### 4.2 Linking per-class FID to classification performance

**Goal**
Demonstrate that per-class generative quality correlates with downstream discriminative performance, tying geometry + FID to practical impact.

**Input**

* Per-class FID (Section 3).
* Per-class sensitivity/F1 from augmented classifiers (Section 4.1).

**Output**

* Scatter plots: per-class FID vs per-class F1 (or sensitivity).
* Correlation coefficients (Spearman/Pearson) with p-values.

**Rationale**

* If reviewers doubt FID, showing that lower FID is associated with better downstream performance (especially for minorities) strengthens its relevance *in this context*, despite global criticisms.([arXiv][5])

**Interpretation**

* You do not need perfect correlation, but a clear trend (better FID → better F1) supports the claim that class-balanced generative quality translates into more robust classifiers.

---

## 5. Joint analysis: latent geometry ↔ performance

### 5.1 Correlating cluster metrics with per-class FID & F1

**Goal**
Close the loop: show that **latent-space geometry** (cluster separation, anisotropy, centroid distances) predicts both per-class FID and per-class classification performance.

**Input**

For each class (c):

* Latent metrics from Section 2 (for selected layer/timestep, final epoch):

  * silhouette contribution, distance between centroid and other centroids, intra-class variance, (\beta_{\text{local}}).
* Per-class FID.
* Per-class F1/sensitivity.

**Output**

* Table with all these quantities per class.
* Heatmap of correlations (geometry metrics vs FID vs F1).
* Optional: simple linear/logistic models predicting “good vs bad” per-class F1 using latent metrics.

**Rationale**

* This turns the latent analysis from “nice visualizations” into explanatory variables for performance.
* It directly applies Mabadeje’s idea that latent stability/geometry is part of an **acceptance criterion** for downstream workflows.([arXiv][3])

**Interpretation**

* For instance: “Classes with silhouette > 0.3 and (\beta_{\text{local}}\in [1,2]) systematically achieve FID < X and F1 > Y; classes outside this range are exactly those with reduced F1.”
* That strongly supports your claim that you have *understood* the latent organization and its implications, rather than just reporting summary metrics.

---

## 6. Guidance scale and ablation analyses (optional but powerful)

### 6.1 Effect of guidance scale on geometry and performance

**Goal**
Show that your class space is robust across a range of CFG scales, and identify a regime that jointly maintains good latent geometry, FID, and classification.

**Input**

* Multiple trained models or multiple sampling CFG scales for the same model.
* For each scale: latent geometry metrics (Section 2), per-class FID (Section 3), classification results (Section 4).

**Output**

* Plots of global FID vs CFG scale, macro-F1 vs scale.
* For a couple of scales: anisotropy figures like you already have.

**Rationale**

* Reviewer concern: “Did you just cherry-pick a single guidance scale?”
* Showing trends vs scale demonstrates that your conclusions are not an artefact of one hyper-parameter choice.

**Interpretation**

* You want a sweet spot where FID is low, classification gains are high, and latent clusters remain well-separated without extreme anisotropy (no over-guidance collapse).

---

## 7. How this package of analyses answers the reviewers’ questions

Putting it all together, this plan lets you argue:

1. **Latent organization without external guidance model**

   * Class embeddings form a well-structured, stable, and data-aligned geometry (Sections 1.1–1.2, 2.2).
   * Mid-block conditional features at high noise exhibit clear class-dependent basins and stable anisotropy, while the unconditional branch remains mixed (Section 2.1), which is exactly what classifier-free guidance theory predicts for a well-trained generative model.([arXiv][2])

2. **Handling of class imbalance**

   * Minority classes are not collapsed or marginalised in latent space (Section 2.3).
   * Per-class FID and downstream classification metrics for minorities are competitive with or improved by your method (Sections 3–4).

3. **Thoroughness and robustness**

   * You do not rely solely on FID; you combine it with cluster validity indices, local anisotropy, stability analysis, and task performance (Sections 2–5).
   * You demonstrate robustness to guidance scale and, if you include it, other ablations (Section 6).

Framed like this, the analysis is not just a collection of plots; it is a *coherent latent-space + performance investigation* that addresses precisely the kind of concerns a Q1 reviewer will raise about class imbalance, latent interpretability, and the necessity (or not) of an external guidance model.

[1]: https://github.com/MedMNIST/MedMNIST?utm_source=chatgpt.com "[pip install medmnist] 18x Standardized Datasets for 2D ..."
[2]: https://arxiv.org/abs/2207.12598?utm_source=chatgpt.com "Classifier-Free Diffusion Guidance"
[3]: https://arxiv.org/abs/2402.11404?utm_source=chatgpt.com "Evaluating the Stability of Deep Learning Latent Feature Spaces"
[4]: https://www.sciencedirect.com/science/article/pii/S2405844025003330?utm_source=chatgpt.com "Cluster validity indices for automatic clustering"
[5]: https://arxiv.org/html/2401.09603v2?utm_source=chatgpt.com "Rethinking FID: Towards a Better Evaluation Metric for ..."
[6]: https://proceedings.neurips.cc/paper_files/paper/2023/file/0bc795afae289ed465a65a3b4b1f4eb7-Paper-Conference.pdf?utm_source=chatgpt.com "Exposing flaws of generative model evaluation metrics and ..."
[7]: https://www.nature.com/articles/s41597-022-01721-8?utm_source=chatgpt.com "MedMNIST v2 - A large-scale lightweight benchmark for 2D ..."
