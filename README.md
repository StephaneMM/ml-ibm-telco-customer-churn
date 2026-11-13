# Predicting Customer Churn: An End-to-End ML Investigation

An 8-week, cost-sensitive machine learning project that takes a telecom churn problem from business framing through a deployment-ready, honestly-benchmarked model, built as a capstone for AIT 506 (Machine Learning) at Westcliff University.

**TL;DR:** Tuned Gradient Boosting classifier, ROC-AUC 0.856, recall 71.9% / precision 57.4% at a decision threshold of 0.36 chosen on out-of-fold probabilities, projected **$79K net value per 1,000 customers** vs. doing nothing, benchmarked honestly against (and falling short of) IBM's own leakage-derived Churn Score.

## Team

Group 2, AIT 506 (Summer 2026), Professor Oueichek Ibba:

| Member | Role | Contact |
|---|---|---|
| Thi Thuy Hang Tran | Project Manager | [@hangtran0](https://github.com/hangtran0) · t.tran.3535@westcliff.edu |
| Daniel Garcia | Data Engineer | d.garcia.7309@westcliff.edu |
| Stéphane Marie Mbesse Manga | ML Engineer | [@StephaneMM](https://github.com/StephaneMM) · stephane.mbesse@gmail.com |
| Duc Nguyen | Documentation Lead | [@KirbyNguyen](https://github.com/KirbyNguyen) · d.nguyen.7467@westcliff.edu |

The notebooks were developed in the team's shared workspace, [KirbyNguyen/AIT506-TeamProject](https://github.com/KirbyNguyen/AIT506-TeamProject). This repository is my own organized copy of the full project, with the weekly reports, the final deck, and the write-ups below.

## Problem

Telecom customer acquisition costs more than retention. The goal isn't the most statistically accurate classifier possible; it's a model whose output changes what a retention team actually does: flag at-risk customers early enough to intervene.

This is framed as **cost-sensitive binary classification**: a missed churner (false negative) costs a lost customer; a false alarm (false positive) only wastes a retention offer. That asymmetry drives every modeling decision below: metric choice, threshold selection, and which errors we tolerate.

## Dataset

- **IBM Telco Customer Churn** (Kaggle/IBM DbD): 7,043 customers, 33 raw columns, 26.5% churn rate.
- Swapped out an initial synthetic Kaggle sample (2,800 rows, artificially inflated 57% churn rate, severe feature collinearity) after one week, because a real dataset had to look and behave like a real business problem before modeling started.
- Two columns (`Churn Score`, `CLTV`) turned out to be IBM's own precomputed model outputs (r = 0.665 with the target). They were excluded from training as leakage and repurposed instead as an **external benchmark** the project's own models are judged against.

## Approach (mirrors Géron's ML Project Checklist, one phase per week)

| Phase | What happened |
|---|---|
| Frame the problem | Cost-sensitive framing; recall/precision/ROC-AUC over accuracy |
| Get & explore data | Dataset pivot after Week 3; correlation & driver analysis |
| Prepare the data | Cleaned `Total Charges`, dropped 13 leakage/ID/geo columns, engineered 3 features, built a single `ColumnTransformer` pipeline, 80/20 stratified split |
| Shortlist models | 8 classifiers (incl. a PyTorch MLP) via 5-fold CV; kept 3 balanced rankers + 2 high-recall models |
| Fine-tune & combine | GridSearch (Logistic Regression) + RandomizedSearch (SVM, Gradient Boosting); tried stacking & soft-voting ensembles |
| Present & deploy | Out-of-fold threshold selection, expected-net-value framing, drift-detection check, monitoring/retraining policy |

## Results

**Model comparison (test set):**

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Gradient Boosting (tuned) | 0.668 | 0.543 | 0.599 | 0.856 |
| **Gradient Boosting (tuned) + threshold 0.36** | **0.574** | **0.719** | **0.638** | **0.856** |
| Stacking ensemble | 0.633 | 0.578 | 0.604 | 0.855 |
| Soft voting ensemble | 0.662 | 0.551 | 0.601 | 0.854 |
| Logistic Regression (tuned) | 0.659 | 0.553 | 0.602 | 0.853 |
| SVM (tuned, class-weight balanced) | 0.567 | 0.722 | 0.635 | 0.839 |
| SVM (default) | 0.718 | 0.396 | 0.510 | 0.812 |
| **IBM Churn Score (benchmark)** | n/a | n/a | n/a | **0.943** |

**Why Gradient Boosting, not the top-ranked ensemble or IBM's own score:**
- Ensembles roughly tied the best single model (base learners made overly similar ranking mistakes, so the diversity that makes ensembling pay off wasn't there).
- The IBM benchmark almost certainly draws on data outside the 19 features we had access to; the ~0.086 AUC gap reads as a data-access ceiling, not a modeling failure.
- At the 0.36 threshold, this model catches **269 of 374 actual churners (71.9%)** in the test set and produces an expected net value of about $79K per 1,000 customers, vs. $0 for doing nothing.

> An earlier draft of the final deck carried different figures (0.858 / 79.1% / 52.9%, threshold 0.15). They did not survive a recheck against the notebooks; the numbers above are the verified ones. The full story is in [LEARNINGS.md](LEARNINGS.md#week-8-checking-our-own-numbers-before-they-reached-a-slide).

## Repo structure

```
ml-ibm-telco-customer-churn/
├── README.md
├── LEARNINGS.md
├── requirements.txt
├── notebooks/
│   ├── 01_get_the_data.ipynb              # Week 3: EDA of the first sample, pivot decision
│   ├── 02_data_preparation.ipynb          # Week 4: IBM Telco cleaning, features, pipeline
│   ├── 03_model_exploration.ipynb         # Week 5: 8 classifiers, 5-fold CV, shortlist
│   ├── 03b_pytorch_mlp.ipynb              # Week 5: feed-forward net with weighted BCE
│   ├── 04_tuning_and_ensembling.ipynb     # Week 6: search, threshold, stacking/voting
│   ├── 05_net_value_threshold_sweep.ipynb # Week 8: expected net value vs. threshold
│   ├── 06_launch_and_monitor.ipynb        # Week 8: batch scoring, drift check, policy
│   └── 07_model_comparison_plots.ipynb    # Week 8: figures for the deck
├── reports/                               # weekly write-ups (Weeks 2–6) and final report
├── slides/                                # final presentation (PDF)
└── images/                                # ROC curves, confusion matrix, net-value curve
```

The notebooks download the dataset with `kagglehub` on first run, so no data files are committed.

## Tech stack

Python · pandas · scikit-learn (GridSearchCV, RandomizedSearchCV, StackingClassifier, VotingClassifier) · PyTorch (weighted BCE loss, early stopping) · matplotlib

## What I'd improve next

- Save the fitted final model with `joblib.dump()` so the artifact is versioned alongside the notebooks (the appendix notes it was never written to a named file).
- Test cost-weighted thresholding directly against real retention-offer economics instead of illustrative $1,680/$50 figures.
- Explore geography features (currently dropped for sparsity) aggregated into regions rather than raw city.
- Validate the drift-detection check against real production drift, not just an injected synthetic shift.

## Author's note on honesty

This project deliberately reports where it falls short of a benchmark (IBM's 0.943 vs. our 0.856) and where an approach didn't pay off (ensembling). That's intentional: the goal was a leakage-free, defensible, and honestly-benchmarked model, not a flattering leaderboard number.
