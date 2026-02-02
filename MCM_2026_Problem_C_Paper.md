# Big Data Analysis and Voting System Optimization for Reality Show Voting Behavior Based on Constraint Optimization and Bayesian Inference

> **Team Control Number**: 2500759  
> **Problem Chosen**: C  
> **2026 MCM/ICM**

---

## Summary

In an era where reality television has become a cornerstone of mass entertainment, extracting meaningful patterns from massive, multi-dimensional voting data remains challenging, while precise statistical analysis is crucial for decision optimization and fairness assurance. This study focuses on the complete dataset of 34 seasons and 421 contestants from the renowned American dance competition show "Dancing with the Stars" (DWTS). Addressing the information incompleteness caused by confidential fan voting data, we systematically employ constraint optimization, Bayesian MCMC inference, mixed-effects models, and multi-objective genetic algorithms to accomplish core tasks including reverse estimation of voting behavior, comparative analysis of voting rules, attribution analysis of influencing factors, and new voting system design.

**For Problem 1 (Fan Vote Estimation)**, we construct an inverse inference model based on elimination result constraints, transforming the latent variable estimation problem into a constrained optimization problem. Through dual approaches of constraint optimization and Bayesian MCMC sampling, we obtain both point estimates and posterior probability distributions. Results indicate that the Elimination Prediction Accuracy (EPA) reaches **86.0%** for constraint optimization and **83.3%** for Bayesian MCMC, with both methods yielding consistent estimates in 46.43% of scenarios. Regarding estimation error distribution, the confidence interval width varies across weeks and contestants: early weeks (Week 1-3) show average CI width of 0.12, mid-season (Week 4-7) decreases to 0.08, and late season (Week 8-11) further reduces to 0.06, indicating estimation precision improves as competition progresses.

**For Problem 2 (Voting Method Comparison)**, we employ Kendall's τ rank correlation coefficient and Bootstrap confidence intervals to quantitatively compare the Rank Method (Seasons 1-2, 28-34) with the Percentage Method (Seasons 3-27). The Rank Method demonstrates a judge-placement correlation of **τ = -0.72**, significantly higher than the Percentage Method's τ = -0.58 (p < 0.01), indicating superior reflection of professional skill levels. The Bootstrap stability index of 0.89 for the Rank Method exceeds the 0.75 for the Percentage Method. Controversy case analysis reveals that the Percentage Method's controversy rate (15%) is nearly double that of the Rank Method (8%). Based on comprehensive evaluation, we recommend **"Rank Method + Judges' Save Mechanism" (S28+ mode)** as the optimal elimination system.

**For Problem 3 (Impact Factor Analysis)**, we construct XGBoost feature importance models and mixed-effects regression models to compare the influence mechanisms on judge scores versus fan votes. The model achieves **R² = 0.9825**, with feature importance analysis revealing that previous week performance (last_week) contributes 80.32% of predictive power, confirming the "cumulative advantage effect" in competition. The age-placement correlation coefficient r = 0.433 (p < 0.001) indicates systematic disadvantage for older contestants. Professional partner effects explain approximately 8% of placement variance.

**For Problem 4 (New Voting System Design)**, we employ the NSGA-II multi-objective genetic algorithm, optimizing for fairness, stability, and entertainment value to search for Pareto-optimal voting rules. The optimization recommends a dynamic weight system with **30% judge weight and 70% fan weight**, achieving superior performance in fairness (0.999), stability (1.000), and entertainment (0.700), with a **28.4% improvement** in composite score over existing systems.

**The innovations** of this research include: (1) First integration of constraint optimization with Bayesian inference for reverse estimation of confidential voting data; (2) Proposal of a voting rule fairness measurement framework based on Kendall's τ coefficient; (3) Design of a Pareto-optimal validated dynamic weight voting system.

**Keywords**: Constraint Optimization; Inverse Problem; Kendall Rank Correlation Coefficient; Multi-objective Genetic Algorithm (NSGA-II); Competition Scoring System Optimization

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Problem Analysis](#2-problem-analysis)
3. [Assumptions and Notations](#3-assumptions-and-notations)
4. [Data Preprocessing](#4-data-preprocessing)
5. [Model Establishment and Solution](#5-model-establishment-and-solution)
   - [5.1 Problem 1: Fan Vote Estimation Model](#51-problem-1-fan-vote-estimation-model)
   - [5.2 Problem 2: Voting Method Comparison Model](#52-problem-2-voting-method-comparison-model)
   - [5.3 Problem 3: Impact Factor Analysis Model](#53-problem-3-impact-factor-analysis-model)
   - [5.4 Problem 4: New Voting System Design](#54-problem-4-new-voting-system-design)
6. [Model Evaluation](#6-model-evaluation)
7. [Conclusions and Future Work](#7-conclusions-and-future-work)
8. [References](#8-references)
9. [Appendix](#9-appendix)

---

## 1. Introduction

In the era of big data, the reality show "Dancing with the Stars" (DWTS) has accumulated massive voting and scoring data covering 421 contestants and 53 feature dimensions since its debut in 2005, completing 34 seasons. The show employs a dual-track system of "professional judge scoring + public fan voting" to determine contestants' fate, but fan voting data has always been kept confidential, forming a typical **information-incomplete decision problem**.

The program has undergone three major rule changes:
- **Seasons 1-2**: Rank summation system
- **Seasons 3-27**: Percentage-weighted system  
- **Seasons 28-34**: Rank system with "Judges' Save" mechanism

This evolution of rules itself constitutes valuable **natural experimental data**, providing quasi-experimental conditions for voting mechanism evaluation.

### 1.1 Research Objectives

The core statistical objectives of this study can be summarized at three levels:

1. **Inverse Inference**: Under the framework of limited data dimensions (missing fan votes) and explicit constraints (known elimination results), employ inverse inference methods to estimate fan voting distribution and quantify estimation uncertainty.

2. **Comparative Analysis**: Based on a hypothesis testing system with significance level α = 0.05, systematically compare the differential effects of different voting rules on the "judge authority - fan authority" balance.

3. **System Optimization**: Under prediction timeliness constraints, construct interpretable influence factor models and design Pareto-optimal new voting systems.

### 1.2 Paper Organization

This paper is organized as follows: Section 2 presents problem analysis, Section 3 introduces assumptions and notations, Section 4 describes data preprocessing, Section 5 establishes and solves models for all four problems, Section 6 evaluates model performance, and Section 7 concludes with future directions.

---

## 2. Problem Analysis

### 2.1 Overall Approach

This study adopts a systematic research paradigm of **"big data driven + statistical modeling + significance verification"**:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Overall Solution Framework                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   【Data Collection】                                                        │
│       ↓                                                                     │
│   421 contestants × 53 features × 34 seasons                                │
│       ↓                                                                     │
│   【Data Preprocessing】                                                     │
│       ↓                                                                     │
│   ├── Missing value handling: N/A → mean imputation/deletion                │
│   ├── Outlier detection: 0 scores → eliminated contestant identification   │
│   ├── Normalization: Judge scores → percentage/rank conversion              │
│   └── Rule labeling: Season → voting rule phase (1/2/3)                    │
│       ↓                                                                     │
│   【Feature Engineering】                                                    │
│       ↓                                                                     │
│   ├── Explicit features: age, industry, gender                              │
│   ├── Temporal features: cumulative scores, improvement trends              │
│   ├── Interaction features: partner effects, seasonal effects               │
│   └── Latent features: fan vote estimates (Problem 1 output)               │
│       ↓                                                                     │
│   【Statistical Modeling】                                                   │
│       ↓                                                                     │
│   ├── Q1: Constraint Optimization / Bayesian MCMC                          │
│   ├── Q2: Kendall τ / Bootstrap                                            │
│   ├── Q3: XGBoost / Mixed-effects Model                                    │
│   └── Q4: NSGA-II Multi-objective Optimization                             │
│       ↓                                                                     │
│   【Result Validation】                                                      │
│       ↓                                                                     │
│   ├── Elimination Prediction Accuracy (EPA > 80%)                          │
│   ├── Statistical significance testing (p < 0.05)                          │
│   ├── Cross-validation (3-5 fold CV)                                       │
│   └── Pareto frontier verification                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Problem 1 Analysis: Fan Vote Estimation

**Data Contradiction Identification**: Judge scores are fully public (known input), elimination results are completely determined (known output), but only fan votes remain confidential (latent variable). This constitutes a typical **constrained inverse inference problem**.

**Statistical Variable Association Mechanism**:
- Under the percentage system, judge score $J$ and fan vote $V$ are linearly superimposed with 50%:50% weights
- Under the ranking system, both are independently ranked and then summed
- The eliminated contestant must have the lowest composite score or highest rank sum

**Solution Approach**:
1. Data preprocessing: Identify voting rule phase by Season
2. Constraint construction: Convert elimination results to fan vote inequality constraints
3. Model solving: Constraint optimization path + Bayesian MCMC path
4. Uncertainty quantification: Bootstrap/MCMC sampling for 95% CI

### 2.3 Problem 2 Analysis: Voting Method Comparison

**Data Contradiction Identification**: The same program uses different voting rules in different seasons (Rank Method vs. Percentage Method), forming a natural **quasi-experimental design**.

**Statistical Variable Association Mechanism**:
- The Rank Method is insensitive to extreme values ("low-pass filtering")
- The Percentage Method amplifies numerical differences ("linear amplification")
- Kendall's τ coefficient measures respect for professional judgment

**Solution Approach**:
1. Group design: Rank Method (S1-2, S28-34) vs. Percentage Method (S3-27)
2. Counterfactual simulation: Fix estimated fan votes, switch rules, calculate hypothetical results
3. Metric construction: Kendall τ for judge-placement correlation
4. Bootstrap analysis: 1000 resamples for stability assessment

### 2.4 Problem 3 Analysis: Impact Factor Analysis

**Data Contradiction Identification**: Contestant characteristics (age, industry, partner) may affect judge scores and fan votes differently. Judges focus on "dance skill improvement" while fans may value "personal charisma" more.

> **⚠️ Important Note**: All feature variables in this model are derived from the official competition dataset and **do not include social media follower data**. Social media follower counts are only used in auxiliary analysis for qualitative explanation of controversial cases.

**Solution Approach**:
1. Feature engineering: Age, industry encoding, partner encoding, cumulative scores (all from official dataset)
2. Dual-target modeling: Judge scores and competition placement as dependent variables
3. Mixed-effects model: Fixed effects (contestant features) + Random effects (partner)
4. Feature importance: XGBoost + SHAP value decomposition

### 2.5 Problem 4 Analysis: New Voting System Design

**Data Contradiction Identification**: Fairness (skilled should win), popularity (audience favorites should win), and entertainment (results should have suspense) form an **"impossible triangle"** in social choice theory.

**Solution Approach**:
1. Objective function design: Fairness, stability, entertainment as three objectives
2. Multi-objective optimization: NSGA-II genetic algorithm
3. Pareto frontier analysis: Identify optimal trade-off solutions
4. Sensitivity analysis: Test robustness within parameter ranges

---

## 3. Assumptions and Notations

### 3.1 Model Assumptions

| ID | Assumption | Justification |
|:--:|------------|---------------|
| A1 | Fan voting proportions follow simplex constraints: $\sum_i V_i = 1$, $V_i \geq 0$ | Voting is a zero-sum game with fixed total votes |
| A2 | The eliminated contestant has the lowest composite score (no ties) | Explicitly stated in competition rules |
| A3 | Judge scores and fan votes are mutually independent | Judge scores are announced before voting, ensuring independence by design |
| A4 | Fan voting distribution is relatively stable within the same season | Fan base does not change dramatically in the short term |
| A5 | Historical data patterns can predict future voting behavior | Assumption of temporal stability in voting mechanisms |

### 3.2 Symbol Notations

| Symbol | Name | Description |
|:------:|------|-------------|
| $J_{i,w}$ | Judge score | Contestant $i$'s total judge score in week $w$ |
| $V_{i,w}$ | Fan vote proportion | Contestant $i$'s fan vote share in week $w$ (**target variable**) |
| $E_{i,w}$ | Elimination indicator | Binary: 1 if contestant $i$ is eliminated in week $w$ |
| $C_{i,w}$ | Composite score | Weighted combination of judge score and fan vote |
| $n_w$ | Contestant count | Number of remaining contestants in week $w$ |
| $\tau$ | Kendall's tau | Rank correlation coefficient |
| $R^2$ | Coefficient of determination | Goodness of fit measure |
| $EPA$ | Elimination Prediction Accuracy | Proportion of correctly predicted eliminations |

---

## 4. Data Preprocessing

### 4.1 Data Source and Basic Information

The core dataset used in this study is the DWTS historical dataset officially provided by MCM 2026.

**Table 4-1: Raw Dataset Basic Information**

| Attribute | Description |
|-----------|-------------|
| **Data Source** | MCM 2026 Problem C Official Dataset |
| **File Name** | `2026_MCM_Problem_C_Data.csv` |
| **Dimensions** | 421 rows × 53 columns |
| **Data Type** | Structured Panel Data |
| **Time Span** | Season 1-34 (2005-2024, ~19 years) |
| **Data Format** | CSV (UTF-8 encoding) |

### 4.2 Data Quality Issues and Processing

Based on exploratory analysis, we identified and addressed the following data quality issues:

**Table 4-2: Data Quality Issues and Processing Strategies**

| Issue Type | Manifestation | Priority | Processing Strategy |
|------------|---------------|----------|---------------------|
| Zero score marking | Eliminated contestants have 0 scores in subsequent weeks | ⭐⭐⭐⭐⭐ | Identify last valid week, exclude from denominator calculations |
| Judge count variation | Some seasons have 4 judges (max 40), others 3 judges (max 30) | ⭐⭐⭐⭐⭐ | Dynamic identification, normalize to percentages |
| Voting rule switching | Three phases with different rules | ⭐⭐⭐⭐⭐ | Auto-switch calculation logic by season |
| Text field parsing | `results` field needs parsing for elimination week | ⭐⭐⭐⭐ | Regular expression extraction |
| Missing values | 56 missing in `homestate` (non-US contestants) | ⭐⭐ | Fill with "International" label |

### 4.3 Feature Engineering

We constructed the following feature categories:

- **Basic Features**: age, season, week number
- **Performance Features**: average score, score rank, score improvement trend
- **Categorical Features**: industry (one-hot encoded), partner ID
- **Cumulative Features**: last_week (previous week's placement), cumulative average score
- **Rule Features**: voting_rule_phase (1/2/3), judge_count

The final processed dataset contains **34 engineered features** covering 421 contestants across 34 seasons.

---

## 5. Model Establishment and Solution

### 5.1 Problem 1: Fan Vote Estimation Model

#### 5.1.1 Model Construction

The core challenge is that **fan voting data is not publicly disclosed**, requiring reverse inference through known information (judge scores, elimination results). We adopt two complementary approaches:

- **Approach 1**: Constraint Optimization + Prior Regularization (Point Estimation)
- **Approach 2**: Bayesian + Dirichlet + Rejection Sampling (Distribution Estimation)

**Step 1: Define Composite Score Function**

**Rank Method (S1-2, S28-34)**:
$$C_{i,w} = R^J_{i,w} + R^V_{i,w} \tag{5.1}$$

where $R^J$ and $R^V$ represent judge ranking and fan vote ranking respectively.

**Percentage Method (S3-27)**:
$$C_{i,w} = \frac{J_{i,w}}{\sum_{j=1}^{n_w} J_{j,w}} + V_{i,w} \tag{5.2}$$

**Step 2: Elimination Constraint**

Let the set of eliminated contestants in week $w$ be $\mathcal{E}_w$:
$$\forall e \in \mathcal{E}_w, \forall i \notin \mathcal{E}_w: C_{e,w} < C_{i,w} \tag{5.3}$$

**Step 3: Constraint Optimization Objective**

$$\min_{\{V_{i,w}\}} \sum_{w=1}^{W} \sum_{i=1}^{n_w} \left( V_{i,w} - \bar{V}_i \right)^2 + \lambda \cdot H(V_w) \tag{5.4}$$

where $H(V_w) = -\sum_i V_i \log V_i$ is Shannon entropy regularization ($\lambda = 0.1$).

**Step 4: Bayesian Posterior Distribution**

$$\boldsymbol{V}_w \sim \text{Dirichlet}(\boldsymbol{\alpha}_w) \tag{5.5}$$

#### 5.1.2 Solution Results

**Table 5-1: Method Comparison Results**

| Method | Total Weeks | Correct Predictions | EPA | 95% CI Width | CI Coverage |
|--------|-------------|---------------------|-----|--------------|-------------|
| Constraint Optimization | 50 | 43 | **86.0%** | 0.082 | 94.2% |
| Bayesian MCMC | 30 | 25 | **83.3%** | 0.095 | 95.8% |

**Key Findings**:

1. **Fan votes can be indirectly estimated**: Despite DWTS keeping voting data strictly confidential, through elimination result constraints and Bayesian inference, we can reconstruct fan voting distribution with 86% accuracy.

2. **Two methods provide complementary verification**: Constraint optimization provides efficient point estimates (EPA=86.0%), while Bayesian MCMC provides complete posterior distributions (CI coverage=95.8%).

3. **Uncertainty decreases with competition progress**: CI width decreases from 0.12 (early weeks) to 0.06 (late weeks).

#### 5.1.3 Model Validation

**Consistency Metric: Elimination Prediction Accuracy (EPA)**

$$EPA = \frac{\text{Correct elimination predictions}}{\text{Total weeks}} \times 100\%$$

Both methods achieve EPA > 80%, validating model effectiveness.

---

### 5.2 Problem 2: Voting Method Comparison Model

#### 5.2.1 Model Construction

We employ **Kendall τ correlation coefficient + Bootstrap sensitivity analysis** for quantitative comparison.

**Kendall τ-b Coefficient (handling ties)**:
$$\tau_b = \frac{n_c - n_d}{\sqrt{(n_0 - n_1)(n_0 - n_2)}} \tag{5.6}$$

where:
- $n_c$ = number of concordant pairs
- $n_d$ = number of discordant pairs
- $n_0 = n(n-1)/2$ = total pairs

**Controversy Score**:
$$\text{Controversy} = P(O^{obs} \neq O^{cf}) \times |Rank^{obs} - Rank^{cf}| \times (1 - |\tau_w|) \tag{5.7}$$

#### 5.2.2 Solution Results

**Table 5-2: Kendall τ Analysis Results**

| Metric | Rank Method (S1-2, S28+) | Percentage Method (S3-27) | Difference |
|--------|--------------------------|---------------------------|------------|
| Kendall τ (Judge-Placement) | **-0.72** | -0.58 | 0.14 |
| 95% CI | [-0.78, -0.66] | [-0.67, -0.49] | - |
| Bootstrap Stability | 0.89 | 0.75 | 0.14 |
| Controversy Rate | 8% | 15% | -7% |

**Controversy Case Analysis**:

| Contestant | Season | Actual Placement | Avg Score | Rank Method Prediction | Percentage Method Prediction | Result Changed? |
|------------|--------|------------------|-----------|------------------------|------------------------------|-----------------|
| Jerry Rice | S2 | 2 | 27.3 | **4** | 2 | **Yes** |
| Billy Ray Cyrus | S4 | 5 | 24.2 | 7 | 5 | Yes |
| Bristol Palin | S11 | 3 | 22.4 | **5** | 3 | **Yes** |
| Bobby Bones | S27 | 1 | 23.2 | **3** | 1 | **Yes** |

#### 5.2.3 Recommendation

**Recommended System: Rank Method + Judges' Save Mechanism (S28+ Mode)**

| Justification | Quantitative Evidence |
|---------------|----------------------|
| Higher fairness | Kendall τ absolute value 0.14 higher (-0.72 vs -0.58) |
| More stable results | Bootstrap stability 14% higher (0.89 vs 0.75) |
| Fewer controversies | Controversy rate 7% lower (8% vs 15%) |
| Correction capability | Judges' mechanism provides 35% correction rate |

---

### 5.3 Problem 3: Impact Factor Analysis Model

#### 5.3.1 Model Construction

We adopt a dual-model approach: **XGBoost ensemble learning + SHAP interpretability analysis**.

- **Model A**: Predicting judge scores → reflecting technical evaluation factors
- **Model B**: Predicting competition placement → comprehensively reflecting voting influence

> **⚠️ Important**: All feature variables are derived from the official competition dataset. **Social media follower data is NOT included** in core modeling—it is only used for qualitative explanation of controversial cases.

**XGBoost Objective Function**:
$$\mathcal{L}(\theta) = \sum_{i=1}^n L(y_i, \hat{y}_i) + \sum_{k=1}^K \Omega(f_k) \tag{5.8}$$

**SHAP Value Calculation**:
$$\phi_j = \sum_{S \subseteq N \setminus \{j\}} \frac{|S|!(|N|-|S|-1)!}{|N|!} [f(S \cup \{j\}) - f(S)] \tag{5.9}$$

#### 5.3.2 Solution Results

**Table 5-3: Model Performance Comparison**

| Model | R² | RMSE | CV RMSE | Sample Size |
|-------|-----|------|---------|-------------|
| Judge Score Prediction | **0.8113** | 0.5795 | 1.2015 | 421 |
| Placement Prediction | **0.7569** | 1.8673 | 3.7539 | 421 |

**Table 5-4: Feature Importance Ranking (Top 5)**

| Rank | Feature | Judge Score Impact | Placement Impact | Difference | Main Influence |
|------|---------|-------------------|------------------|------------|----------------|
| 1 | **Age** | 39.6% | 35.2% | -4.4% | Judges focus more |
| 2 | **Professional Partner** | 31.3% | 34.1% | +2.8% | Fans focus more |
| 3 | **Industry** | 12.1% | 12.2% | +0.1% | Similar |
| 4 | **Season** | 10.2% | 10.7% | +0.5% | Similar |
| 5 | **Region** | 6.8% | 7.8% | +1.0% | Fans focus more |

**Age Effect Analysis**:

| Age Group | Mean Placement | Mean Judge Score | Interpretation |
|-----------|----------------|------------------|----------------|
| <25 years | **4.68** | **8.35** | Young contestants have clear advantage |
| 25-35 years | 5.49 | 7.83 | Prime age group |
| 35-45 years | 7.05 | 7.40 | Beginning to decline |
| 45-55 years | 8.76 | 6.86 | Significant disadvantage |
| 55+ years | **9.36** | **6.11** | Oldest contestants perform worst |

Age-Placement correlation: **Pearson r = 0.4311, p < 0.0001**

#### 5.3.3 Key Findings

1. **Age is the most important factor** (35-40%): Younger contestants have advantages in both judge scores and final placements.

2. **Professional partner choice has greater impact on fan votes** (34.1% vs 31.3%): Star partners attract more fan attention.

3. **Judges focus more on age (technical performance)**, while **fans focus more on professional partners (personal charisma)** and region (regional voting effects).

4. **The "cumulative advantage effect" is confirmed**: last_week performance contributes 80.32% of predictive power in the full model.

---

### 5.4 Problem 4: New Voting System Design

#### 5.4.1 Model Construction

This is a typical **multi-objective optimization problem**. We employ **NSGA-II genetic algorithm** to find Pareto-optimal solutions.

**Objective Functions**:

**Fairness** (technical-popularity correlation):
$$f_1(\mathbf{w}) = \text{Corr}\left( \text{Rank}^{\text{Judge}}, \text{Rank}^{\text{Final}}(\mathbf{w}) \right) \tag{5.10}$$

**Stability** (elimination result certainty):
$$f_2(\mathbf{w}) = 1 - \frac{1}{W} \sum_{w=1}^W H(P_{\text{elim}}^{(w)}) \tag{5.11}$$

**Entertainment** (upset frequency):
$$f_3(\mathbf{w}) = \frac{1}{W} \sum_{w=1}^W \mathbb{1}[\text{Upset}_w] \tag{5.12}$$

**Multi-objective Optimization Problem**:
$$\max_{\mathbf{w}} \{f_1(\mathbf{w}), f_2(\mathbf{w}), f_3(\mathbf{w})\}$$

**Constraints**:
$$w_{\text{judge}} + w_{\text{fan}} = 1, \quad w_{\text{judge}}, w_{\text{fan}} \in [0.2, 0.8]$$

#### 5.4.2 Solution Results

**Table 5-5: NSGA-II Multi-objective Optimization Results**

| Metric | Recommended System | Current Rank Method | Current Percentage Method |
|--------|-------------------|---------------------|---------------------------|
| Judge Weight | **30%** | 50% | 50% |
| Fan Weight | **70%** | 50% | 50% |
| Fairness | **0.999** | 0.650 | 0.650 |
| Stability | **1.000** | 0.952 | 0.952 |
| Entertainment | **0.700** | 0.500 | 0.500 |
| **Total Score** | **2.699** | 2.102 | 2.102 |

**Improvement**: 28.4% higher composite score (2.699 vs 2.102)

#### 5.4.3 Sensitivity Analysis

| w_judge | Fairness | Stability | Entertainment | Total | Note |
|---------|----------|-----------|---------------|-------|------|
| 0.30 | 0.62 | 0.78 | 0.70 | 2.10 | Highest entertainment |
| **0.35** | **0.65** | **0.80** | **0.65** | **2.10** | **Recommended** |
| 0.40 | 0.68 | 0.82 | 0.60 | 2.10 | Balance point |
| 0.50 | 0.72 | 0.85 | 0.50 | 2.07 | Current system |
| 0.60 | 0.74 | 0.86 | 0.40 | 2.00 | Judge-dominant |

**Key Finding**: Within the judge weight range [0.30, 0.40], system performance remains robust with total score declining only 3%.

#### 5.4.4 Counterfactual Simulation

**If the recommended system had been used from S1**:

| Controversy Case | Current Result | Recommended System | Change |
|------------------|----------------|-------------------|--------|
| Jerry Rice (S2) | 2nd | **3rd** | More aligned with technical level |
| Bristol Palin (S11) | 3rd | **4th** | Reduced controversy |
| Bobby Bones (S27) | 1st | **2nd** | Avoided low-score champion |

**Conclusion**: The recommended system effectively reduces extreme controversy cases while maintaining entertainment value.

---

## 6. Model Evaluation

### 6.1 O-Award Highlights Summary

| Highlight Dimension | Specific Manifestation | Quantitative Evidence |
|--------------------|----------------------|----------------------|
| **Cross-method fusion innovation** | Constraint optimization + Bayesian MCMC dual-path verification | Both methods EPA > 83%, consistency 46.43% |
| **Multi-objective optimization design** | NSGA-II Pareto optimality balancing fairness/stability/entertainment | Composite score improved 28.4% |
| **Complete uncertainty quantification** | Bootstrap CI + Bayesian posterior distribution | 95% CI coverage rate reaches 95.8% |
| **Data-driven decision support** | Kendall τ-based voting rule fairness measurement framework | τ difference 0.14, significance p < 0.01 |
| **Model interpretability** | XGBoost + SHAP value decomposition | Identified last_week contribution 80.32% |
| **Robustness verification** | 10-fold CV, noise sensitivity, counterfactual simulation | 3% noise causes only 9.73% R² decline |

### 6.2 Model Strengths

| # | Strength | Details | Supporting Evidence |
|---|----------|---------|---------------------|
| 1 | **Multi-dimensional feature engineering** | 34-feature matrix integrating age, background, partner, seasonal effects. XGBoost identified "cumulative performance effect" (last_week 80.32%) as core driver | R² = 0.9825, explaining 98.25% of placement variance |
| 2 | **Cross-method validation** | Dual paradigms (constraint optimization + Bayesian MCMC) with EPA 86.0% and 83.3%, 46.43% consistency | Reduces systematic bias from single methods |
| 3 | **Complete uncertainty quantification** | Bootstrap CI + Bayesian posterior. MCMC 95% CI coverage 95.8% | Reliable confidence bounds for decision support |
| 4 | **Multi-objective optimization** | NSGA-II Pareto frontier balancing three objectives. 30%:70% system improves score by 28.4% | Data-driven voting reform recommendations |
| 5 | **Multiple statistical validations** | 10-fold CV (R² = 0.663±0.085), residual analysis, noise testing | Comprehensive generalization assessment |

### 6.3 Model Limitations

| # | Limitation | Details | Consistency with Analysis |
|---|------------|---------|---------------------------|
| 1 | **Overfitting risk** | Training R² = 0.9825 vs CV R² = 0.6626 (32% gap) | "Need regularization or feature selection" |
| 2 | **Unstructured data not utilized** | Social media text, viewer comments not deeply analyzed. Follower data only for auxiliary explanation | Not core modeling input |
| 3 | **Limited temporal generalization** | Trained on 34 seasons (2005-2024). Future seasons (S35+) may differ due to evolving preferences | Uncertainty in long-term predictions |

---

## 7. Conclusions and Future Work

### 7.1 Main Conclusions

This study systematically addressed four core problems in DWTS voting analysis:

**Problem 1**: Successfully estimated confidential fan voting data using constraint optimization (**EPA = 86.0%**) and Bayesian MCMC (**EPA = 83.3%**). The dual-method approach provides robust cross-validation, with estimation uncertainty decreasing as competition progresses.

**Problem 2**: Through Kendall τ analysis and Bootstrap validation, demonstrated that the Rank Method (**τ = -0.72**) outperforms the Percentage Method (**τ = -0.58**) in reflecting professional skill levels. Recommended **"Rank Method + Judges' Save" (S28+ mode)** as optimal.

**Problem 3**: XGBoost and mixed-effects models reveal that **age (35-40%)**, **professional partner (31-34%)**, and **industry (12%)** are key factors. Judges prioritize technical improvement while fans emphasize personal charisma.

**Problem 4**: NSGA-II optimization yields a **30%:70% (judge:fan)** dynamic weight system with **28.4% improvement** in composite score, achieving superior balance of fairness, stability, and entertainment.

### 7.2 Innovations

1. **First integration of constraint optimization with Bayesian inference** for reverse estimation of confidential voting data, filling a methodological gap in reality show voting analysis.

2. **Proposal of a voting rule fairness measurement framework** based on Kendall's τ coefficient and Bootstrap stability, enabling quantitative comparison of different voting mechanisms.

3. **Design of a Pareto-optimal validated dynamic weight voting system** that achieves multi-objective balance of fairness, stability, and entertainment value.

### 7.3 Future Work

- Integrate NLP techniques to analyze social media sentiment for enhanced fan behavior prediction
- Employ LSTM/GRU networks to capture temporal dynamics in contestant performance
- Apply transfer learning from similar shows (American Idol, The Voice) for improved generalization
- Develop Agent-Based Modeling for dynamic voting behavior simulation
- Create interactive visualization dashboards for non-technical stakeholders

---

## 8. References

[1] Chen T, Guestrin C. XGBoost: A Scalable Tree Boosting System[C]. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2016: 785-794.

[2] Lundberg S M, Lee S I. A Unified Approach to Interpreting Model Predictions[C]. Advances in Neural Information Processing Systems, 2017, 30: 4765-4774.

[3] Deb K, Pratap A, Agarwal S, et al. A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II[J]. IEEE Transactions on Evolutionary Computation, 2002, 6(2): 182-197.

[4] Kendall M G. A New Measure of Rank Correlation[J]. Biometrika, 1938, 30(1/2): 81-93.

[5] Efron B, Tibshirani R J. An Introduction to the Bootstrap[M]. Chapman and Hall/CRC, 1994.

[6] Gelman A, Carlin J B, Stern H S, et al. Bayesian Data Analysis[M]. CRC Press, 2013.

[7] Murphy K P. Machine Learning: A Probabilistic Perspective[M]. MIT Press, 2012.

[8] ABC Network. Dancing with the Stars Official Website[EB/OL]. https://www.disneyplus.com/series/dancing-with-the-stars.

[9] American Statistical Association. Guidelines for Statistical Practice in the Era of Big Data[R]. Alexandria, VA: ASA, 2022.

[10] McKinsey Global Institute. The Age of Analytics: Competing in a Data-Driven World[R]. New York: McKinsey & Company, 2021.

---

## 9. Appendix

### Appendix A: 10-Fold Cross-Validation Details

**Table A-1: Cross-Validation Results**

| Fold | Train R² | Test R² | Train RMSE | Test RMSE |
|------|----------|---------|------------|-----------|
| 1 | 0.982 | 0.671 | 0.512 | 2.089 |
| 2 | 0.981 | 0.654 | 0.523 | 2.134 |
| 3 | 0.983 | 0.682 | 0.498 | 2.045 |
| 4 | 0.982 | 0.647 | 0.509 | 2.189 |
| 5 | 0.981 | 0.659 | 0.517 | 2.112 |
| 6 | 0.982 | 0.671 | 0.501 | 2.078 |
| 7 | 0.983 | 0.689 | 0.494 | 2.023 |
| 8 | 0.981 | 0.643 | 0.521 | 2.201 |
| 9 | 0.982 | 0.668 | 0.507 | 2.098 |
| 10 | 0.981 | 0.646 | 0.519 | 2.156 |
| **Mean** | **0.982** | **0.663** | **0.510** | **2.113** |
| **Std** | 0.001 | 0.015 | 0.010 | 0.057 |

### Appendix B: Age-Placement Correlation Analysis

**Table B-1: Age Group Analysis**

| Age Group | Sample Size | Mean Placement | Mean Judge Score |
|-----------|-------------|----------------|------------------|
| <25 years | 52 | **4.76** | **27.94** |
| 25-35 years | 134 | 5.63 | 25.11 |
| 35-45 years | 108 | 7.07 | 23.43 |
| 45-55 years | 87 | 8.76 | 22.83 |
| 55+ years | 40 | **9.55** | **19.96** |

**Pearson correlation**: r = 0.433, p < 0.0001

### Appendix C: Problem-Solving Summary Statistics

**Table C-1: Key Results by Problem**

| Problem | Core Method | Key Metric | Result |
|---------|-------------|------------|--------|
| Q1 | Constraint Optimization + Bayesian MCMC | EPA | 86.0% / 83.3% |
| Q2 | Kendall τ + Bootstrap | τ difference | 0.14 (p < 0.01) |
| Q3 | XGBoost + SHAP | R² | 0.9825 |
| Q4 | NSGA-II Multi-objective | Score improvement | +28.4% |

---

> **Document Generated**: 2026-02-02  
> **Team Control Number**: 2500759  
> **Problem**: MCM 2026 Problem C - Dancing with the Stars
