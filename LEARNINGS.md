# What I Learned

Technical debugging notes and decision points from across the 8-week project, kept in the "what actually happened" voice rather than cleaned up into a polished narrative, because the mistakes and fixes are more informative than the final numbers.

## Week 3 → 4: Starting over on the dataset

The original dataset (2,800 synthetic rows via Kaggle's API) had a churn rate artificially inflated to 57% and only 10 columns. It looked usable at first glance, but it couldn't support the cost-sensitive scenario the project needed and showed collinearity up to 1.0 between features, a red flag that some columns were derived from others. Switching to IBM's real Telco dataset (7,043 rows, 33 columns, 26.5% churn) cost a week of momentum but meant every metric and cost calculation afterward was actually trustworthy.

## Catching leakage before it reached a model

`Churn Score` and `CLTV` looked like ordinary customer attributes. They weren't. They were IBM's own precomputed model outputs, and `Churn Score` correlated with the target at r = 0.665, which is far too high for something masquerading as an input feature. `Churn Reason` was worse: it was only populated for customers who had already churned, meaning its presence/absence *was* the label in another column.

The lesson that stuck: a single feature with a suspiciously high target correlation is worth investigating before it's worth using. We kept `Churn Score` in the dataset anyway, not as a feature but as an external benchmark our own models would be judged against later.

## Why accuracy was the wrong headline number

With a 26.5% churn rate, a model that always predicts "stay" scores ~73.5% accuracy while catching zero churners. Several of our own models scored only marginally better than that trivial baseline on raw accuracy, which is exactly the scenario where accuracy quietly lies. Precision, recall, F1, and ROC-AUC (threshold-independent) were reported instead, and ROC-AUC specifically was used to compare ranking quality before any threshold was chosen.

## The SVM recall problem, and why it wasn't a hyperparameter issue

At default settings, the SVM had a real problem: 0.812 ROC-AUC but recall of only 0.40. It was ranking customers reasonably well but defaulting to "predict stay" far too often on the minority class. Tuning `C`, kernel, and gamma barely moved this. What actually fixed it was setting `class_weight='balanced'`, which took recall from 0.40 to 0.72 and F1 from 0.51 to 0.64. The takeaway: on imbalanced data, how a model treats the minority class during training can matter more than its core hyperparameters.

## Choosing a decision threshold without leaking the test set

A ranking model (ROC-AUC) isn't a decision by itself; it needs a probability cutoff. The default 0.50 threshold is arbitrary on an imbalanced target. To pick a better one without touching the test set, we generated out-of-fold predictions via cross-validation, swept thresholds from 0.1–0.9, and picked the value that maximized F1 (0.36), then applied that fixed threshold to the untouched test set. This moved F1 from 0.50 to ~0.60 and recall from 0.54 to ~0.72. Learning the threshold from the same data used to train, rather than from the test set, is what kept this leakage-free.

## Why the ensembles didn't win

Both a stacking classifier and a soft-voting classifier were tested on the assumption that combining the three tuned models would outperform any single one. They didn't. They landed within ~0.002 ROC-AUC of the best individual model (tuned Gradient Boosting). The likely reason: after tuning, the three base models were making very similar ranking mistakes on the same customers. Ensembling only pays off when base models are both accurate *and* genuinely different in their errors; that diversity wasn't there. Reporting a negative result honestly (ensembles ≈ tied, not better) felt more useful than quietly picking whichever number looked best.

## The PyTorch model: same ranking, different threshold behavior

A small feed-forward network (43 → 32 → 16 → 1, batch norm, dropout, early stopping at epoch 42) was trained with a weighted binary cross-entropy loss to counter the imbalance. Its F1 looked like a leaderboard win (0.627), but almost all of that came from the loss weighting pushing recall to 0.83 at low precision (~0.50), the same trade-off profile as Naive Bayes. Judged by ROC-AUC (threshold-independent), it landed at 0.854, essentially tied with Logistic Regression and Gradient Boosting. The real lesson: a headline F1 number can hide the fact that a model is just operating at a more aggressive threshold, not ranking customers any better.

## Week 8: checking our own numbers before they reached a slide

The first draft of the final deck reported the tuned Gradient Boosting model at 0.858 ROC-AUC, 79.1% recall and 52.9% precision, credited a big recall jump (0.532 → 0.791) to a "sample-weighting fix" on Gradient Boosting, and recommended a 0.15 decision threshold worth +13% net value. Those numbers had been copied from a summary, not from the notebooks. Going back to the notebook outputs, cell by cell, changed almost all of them:

- **The recall fix was misattributed.** The 0.40 → 0.72 improvement belongs to the SVM and comes from `class_weight='balanced'` (Week 6 output). Gradient Boosting never had a sample-weighting fix; its tuning gains were in ranking quality, not recall.
- **The headline metrics didn't match the confusion matrix.** Recomputed from the actual test-set matrix (835 / 200 / 105 / 269), the tuned model is 0.856 ROC-AUC, 71.9% recall, 57.4% precision: 269 of 374 churners caught, not 296. The benchmark gap becomes 0.087.
- **A wrong model had been saved as "final".** While re-running the tuning in a separate notebook to reproduce the net-value figures, we found that a related notebook had stored an untuned, differently-named model class under the "final" name. The net-value numbers were re-derived from the exact tuned model: $87,516 at the default 0.5 threshold → $112,126 at 0.36 on the test set (+28%), replacing $121,184 → $137,052 (+13%).
- **The threshold recommendation changed.** The sweep does show net value peaking near 0.08 under Week 2's cost assumptions, but at ~40% precision and 96% recall that means flagging most of the customer base, more volume than a retention team can act on. We kept 0.36, the F1-optimal value on out-of-fold probabilities, pending validation of real intervention capacity and the $50-per-contact assumption at that scale.
- **The per-1,000 net value moved from $86,007 to $79,431** as a direct consequence of the corrected precision and recall. The model still clears Week 2's bar of beating "do nothing".

Two slides also got their layout fixed on the way (one chart had been mixing single-feature correlations with Random Forest importances, which are two different analyses from two different stages). None of this would have surfaced from re-reading our own report; it only showed up by treating every number on a slide as a claim to re-derive from source. That habit, not trusting a polished summary of your own work, is the thing I'd carry into the next project before any modeling technique.

## Being honest about the benchmark gap

IBM's own Churn Score reaches 0.943 ROC-AUC on the test set, well above our best model's 0.856. Rather than chase that number by re-introducing leakage, the gap is reported as a likely **data-access ceiling** (IBM's score probably draws on data outside the 19 features available here), not a modeling shortfall. A model that's honestly benchmarked and slightly behind is more defensible than one that's quietly cheating and looks better.
