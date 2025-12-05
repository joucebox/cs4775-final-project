# CS4775 Final Project Report Outline

## Beyond Viterbi: A Maximum Expected Accuracy Approach to Pairwise Alignment

---

## Title Page

### Title

Beyond Viterbi: A Maximum Expected Accuracy Approach to Pairwise Alignment

### Authors

1. Lawrence Granda - Computer Science, Junior, lg626
2. Joyce Shen - Computer Science, Junior, js3696
3. Soham Goswami - [Major], [Year], sbg226
4. Joshua Ochalek - [Major], [Year], jo447

### Abstract (200 words minimum)

- Problem: Pairwise sequence alignment is fundamental to computational biology
- Challenge: Viterbi alignment finds the single most probable path but ignores alignment uncertainty
- Approach: Implement MEA alignment using posterior match probabilities from a pair HMM
- Methods: MLE parameter estimation, forward-backward algorithm, Viterbi and MEA comparison
- Results: Evaluation on Rfam RNA families across multiple gamma values
- Conclusion: MEA provides tunable precision-recall tradeoff via gamma parameter

### Keywords

- Pairwise sequence alignment
- Hidden Markov Model (HMM)
- Maximum Expected Accuracy (MEA)
- Viterbi algorithm
- Forward-backward algorithm
- RNA sequence analysis
- Posterior probability

### Project Type

- Reimplementation
- Implements pair HMM algorithms from literature and evaluates on Rfam data

### Repository

- URL: https://github.com/joucebox/cs4775-final-project
- Contains: Source code, data, results, and reproduction instructions
- See README.md for setup and usage

### AI Use Attribution

- Document any AI tools used during development

---

## 1. Introduction

### 1.1 Background and Motivation

- Importance of pairwise sequence alignment in bioinformatics
- RNA sequence comparison for functional annotation
- Limitations of traditional scoring-matrix approaches

### 1.2 Problem Statement

- Viterbi alignment returns a single optimal path (MAP estimate)
- Does not account for posterior uncertainty in alignment columns
- MEA alignment maximizes expected number of correctly aligned positions

### 1.3 Objectives

- Implement a 3-state pair HMM for RNA alignment
- Estimate parameters via MLE from curated Rfam alignments
- Compare Viterbi vs MEA alignment quality across gamma values
- Analyze precision-recall tradeoffs

### 1.4 Contributions

- Complete Python implementation of pair HMM algorithms
- Analysis of performance on 368 Rfam pairwise alignments against golden labels
- Analysis of gamma parameter effect on alignment accuracy

---

## 2. Methods

### 2.1 Pair Hidden Markov Model

#### 2.1.1 Model Architecture

- Three-state model: M (match/mismatch), X (insert in X), Y (insert in Y)
- State M: consumes one symbol from sequence x and one from y
- State X: consumes one symbol from x only (gap in y)
- State Y: consumes one symbol from y only (gap in x)

#### 2.1.2 Emission Probabilities

- Match state: P(x_i, y_j | M) - joint emission of base pair
- Insert X state: P(x_i | X) - marginal emission in x
- Insert Y state: P(y_j | Y) - marginal emission in y
- RNA alphabet: {A, C, G, U}

#### 2.1.3 Transition Probabilities

- 3×3 transition matrix between states {M, X, Y}
- Affine gap model: gap open (δ) and gap extend (ε)
- Constraint: P(X→Y) = P(Y→X) = 0 (no direct transitions between insert states)

### 2.2 Parameter Estimation (MLE)

#### 2.2.1 Data Source

- Rfam database: curated RNA family alignments
- 368 pairwise Stockholm alignments extracted from multiple sequence alignments

#### 2.2.2 Count Accumulation

- Iterate through aligned columns to count:
  - Match emissions: count(x_i, y_j | state = M)
  - Insert X emissions: count(x_i | state = X)
  - Insert Y emissions: count(y_j | state = Y)
  - State transitions: count(state*t → state*{t+1})

#### 2.2.3 Normalization with Pseudocounts

- Additive smoothing (pseudocount α = 0.5) to avoid zero probabilities
- Normalize counts to probabilities:
  - P(x, y | M) = (count(x, y) + α) / Σ(count + α)
- Convert to log-space for numerical stability

#### 2.2.4 Gap Parameters

- Gap open probability: δ = P(M→X) + P(M→Y)
- Gap extend probability: ε = (P(X→X) + P(Y→Y)) / 2

### 2.3 Forward-Backward Algorithm

#### 2.3.1 Purpose

- Compute posterior probability of being in match state at position (i, j)
- P(M at (i,j) | x, y) = P(M_ij, x, y) / P(x, y)

#### 2.3.2 Forward Algorithm

- F_M[i][j]: log probability of emitting x[1:i], y[1:j] and ending in state M
- F_X[i][j]: log probability of emitting x[1:i], y[1:j] and ending in state X
- F_Y[i][j]: log probability of emitting x[1:i], y[1:j] and ending in state Y
- Initialization from start distribution
- Recurrence using logsumexp for numerical stability:
  ```
  F_M[i][j] = logsumexp(F_M[i-1][j-1] + log_trans(M,M),
                        F_X[i-1][j-1] + log_trans(X,M),
                        F_Y[i-1][j-1] + log_trans(Y,M)) + log_emit_M(x[i], y[j])
  ```

#### 2.3.3 Backward Algorithm

- B_M[i][j]: log probability of emitting x[i+1:n], y[j+1:m] starting from state M
- Initialize at terminal position (n, m) with end probabilities
- Reverse iteration through the DP grid
- Recurrence symmetric to forward

#### 2.3.4 Posterior Computation

- Log partition function: logZ = logsumexp over terminal states
- Posterior match probability:
  ```
  P(M at i,j | x, y) = exp(F_M[i][j] + B_M[i][j] - logZ)
  ```
- Verify: logZ_forward ≈ logZ_backward (sanity check)

### 2.4 Viterbi Algorithm

#### 2.4.1 Purpose

- Find the single most probable alignment path (MAP estimate)
- Maximize P(path, x, y) over all possible state sequences

#### 2.4.2 Dynamic Programming

- V_M[i][j]: log probability of best path ending at (i,j) in state M
- V_X[i][j]: log probability of best path ending at (i,j) in state X
- V_Y[i][j]: log probability of best path ending at (i,j) in state Y
- Recurrence using max instead of logsumexp:
  ```
  V_M[i][j] = max(V_M[i-1][j-1] + log_trans(M,M),
                  V_X[i-1][j-1] + log_trans(X,M),
                  V_Y[i-1][j-1] + log_trans(Y,M)) + log_emit_M(x[i], y[j])
  ```

#### 2.4.3 Backpointer Tracking

- Ψ_M[i][j], Ψ_X[i][j], Ψ_Y[i][j]: store which state gave the max
- Used for traceback to reconstruct optimal alignment

#### 2.4.4 Traceback

- Start at terminal state with highest score at (n, m)
- Follow backpointers to recover alignment:
  - M → emit aligned pair (x[i], y[j])
  - X → emit (x[i], gap)
  - Y → emit (gap, y[j])

### 2.5 Maximum Expected Accuracy (MEA) Algorithm

#### 2.5.1 Motivation

- Viterbi optimizes for the single best path
- MEA optimizes for expected number of correctly aligned positions
- Incorporates alignment uncertainty via posteriors

#### 2.5.2 Gamma Parameter

- Weight matrix: w[i][j] = P(M at i,j | x, y)^γ
- Interpretation:
  - γ = 1: use raw posterior probabilities
  - γ > 1: accentuate high-confidence matches (higher precision)
  - γ < 1: flatten differences (higher recall)

#### 2.5.3 MEA Dynamic Programming

- DP[i][j]: maximum expected accuracy for aligning x[1:i] with y[1:j]
- Recurrence:
  ```
  DP[i][j] = max(DP[i-1][j-1] + w[i][j],   # align x[i] with y[j]
                 DP[i-1][j],                # gap in y
                 DP[i][j-1])                # gap in x
  ```
- No gap penalty: accuracy score comes entirely from posterior weights

#### 2.5.4 Traceback

- Standard alignment traceback using pointer matrix
- Score represents expected number of correctly aligned positions

---

## 3. Results

### 3.1 Experimental Setup

- Dataset: 368 Rfam pairwise alignments as gold standard
- Gamma values tested: 0.01, 0.1, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0
- Metrics: Precision, Recall, F1 score, Column identity

### 3.2 Synthetic Data (if applicable)

- Controlled experiments with known ground truth
- Parameter recovery analysis

### 3.3 Real Data

#### 3.3.1 MEA vs Viterbi Comparison

- F1 scores across gamma values
- Delta F1 (MEA - Viterbi) analysis
- Precision-recall tradeoff curves

#### 3.3.2 Posterior Analysis

- Posterior mass captured by MEA vs Viterbi alignments
- Per-family performance breakdown
- Efficiency scatter plots (posterior mass per aligned pair)

#### 3.3.3 Column Identity Analysis

- Fraction of exactly matching columns
- Impact of gamma on alignment stringency

### 3.4 Visualization

- Posterior heatmaps with alignment overlays
- MEA vs Viterbi comparison plots
- Performance metrics across gamma values

---

## 4. Discussion

### 4.1 Summary of Findings

- MEA alignment quality compared to Viterbi
- Optimal gamma ranges for precision vs recall
- Computational considerations

### 4.2 Interpretation

- When does MEA outperform Viterbi?
- Role of posterior uncertainty in alignment quality
- Practical gamma selection guidelines

### 4.3 Limitations

- Pairwise vs multiple sequence alignment
- Parameter estimation from limited data
- Computational complexity considerations

### 4.4 Future Work

- Extension to profile HMMs
- Alternative posterior formulations
- Integration with downstream analyses

### 4.5 Conclusion

- Key takeaways for practitioners
- Recommendations for alignment method selection

---

## References

- Use natbib with final_ref.bst style
- Key citations: HMM alignment literature, MEA methods, Rfam database

---

## Figures

### Figure 1: Pair HMM State Diagram

- Three-state model visualization
- Transition and emission probabilities

### Figure 2: F1/Precision/Recall vs Gamma

- Performance metrics across gamma values
- MEA vs Viterbi comparison

### Figure 3: Delta F1 vs Gamma

- MEA improvement over Viterbi
- Identification of optimal gamma range

### Figure 4: Posterior Heatmaps

- Example alignment with posterior probabilities
- MEA vs Viterbi path overlay

### Figure 5: Posterior Gain Analysis

- Multi-panel analysis of posterior mass
- Family-wise heatmap
- Efficiency scatter plot

---

## Tables

### Table 1: Estimated HMM Parameters

- Emission probabilities (match, insert_x, insert_y)
- Transition probabilities
- Gap parameters (δ, ε)

### Table 2: Performance Summary

- Metrics by gamma value
- Statistical significance tests

---

## Algorithms

### Algorithm 1: Forward Algorithm

- Pseudocode for forward DP in log-space

### Algorithm 2: Backward Algorithm

- Pseudocode for backward DP in log-space

### Algorithm 3: Viterbi Algorithm

- Pseudocode for MAP alignment

### Algorithm 4: MEA Algorithm

- Pseudocode for maximum expected accuracy alignment
