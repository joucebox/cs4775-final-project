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

We model the joint generation of a pair of RNA sequences,

$$
X = x_1, \dots, x_{L_X} \;\;\text{and}\;\; Y = y_1, \dots, y_{L_Y},
$$

where each symbol is drawn from the RNA alphabet

$$
\mathcal{A} = \{A, C, G, U\}.
$$

The alignment of the two sequences is represented by a sequence of hidden states

$$
Z = (z_1, \dots, z_T), \quad z_t \in \mathcal{S},
$$

where the state space is fixed to

$$
\mathcal{S} = \{ M, X, Y \}.
$$

Here, $M$ denotes a match (or substitution) state, $X$ an insertion in the first sequence (gap in $Y$), and $Y$ an insertion in the second sequence (gap in $X$).

We associate to each time step $t$ a pair of indices $(i_t, j_t)$ indicating how many characters of $X$ and $Y$ have been consumed up to that step. The state $z_t$ determines which symbols are emitted and how the indices are advanced:

- If $z_t = M$, the model emits a pair $(x_{i_t+1}, y_{j_t+1})$ and advances both indices:

  $$
  (i_{t+1}, j_{t+1}) = (i_t + 1, j_t + 1)
  $$

- If $z_t = X$, the model emits a single symbol $x_{i_t+1}$ and advances only the first sequence:

  $$
  (i_{t+1}, j_{t+1}) = (i_t + 1, j_t)
  $$

- If $z_t = Y$, the model emits a single symbol $y_{j_t+1}$ and advances only the second sequence:
  $$
  (i_{t+1}, j_{t+1}) = (i_t, j_t + 1)
  $$

A path $Z$ is considered valid if, after $T$ steps, all characters of both sequences have been consumed:

$$
(i_T, j_T) = (L_X, L_Y).
$$

#### 2.1.1 Emission Model

The emission probabilities are defined for each state:

- **Match state ($M$).** When $z_t = M$, emit a pair of bases $(x, y) \in \mathcal{A} \times \mathcal{A}$ according to

  $$
  e_M(x, y) = P(\text{emit } (x, y) \mid z_t = M),
  $$

  with normalization: $\sum_{x \in \mathcal{A}} \sum_{y \in \mathcal{A}} e_M(x, y) = 1$.

- **Insertion in $X$ (state $X$).** When $z_t = X$, emit base $x \in \mathcal{A}$ from $X$ with

  $$
  e_X(x) = P(\text{emit } x \mid z_t = X),
  $$

  where $\sum_{x \in \mathcal{A}} e_X(x) = 1$.

- **Insertion in $Y$ (state $Y$).** When $z_t = Y$, emit base $y \in \mathcal{A}$ from $Y$ with
  $$
  e_Y(y) = P(\text{emit } y \mid z_t = Y),
  $$
  and $\sum_{y \in \mathcal{A}} e_Y(y) = 1$.

We can write a unified emission function:

$$
e_{z_t}(\cdot) =
\begin{cases}
e_M(x_{i_t+1}, y_{j_t+1}) & \text{if } z_t = M, \\[4pt]
e_X(x_{i_t+1})            & \text{if } z_t = X, \\[4pt]
e_Y(y_{j_t+1})            & \text{if } z_t = Y.
\end{cases}
$$

#### 2.1.2 Start, Transition, and End Distributions

The HMM parameters include:

- **Start distribution**:

  $$
  \pi(s) = P(z_1 = s), \quad s \in \{M, X, Y\}
  $$

  which defaults to the uniform distribution $\pi(M) = \pi(X) = \pi(Y) = \tfrac{1}{3}$, unless otherwise specified.

- **State transition matrix**:

  $$
  a_{uv} = P(z_{t+1} = v \mid z_t = u), \quad u, v \in \{M, X, Y\}
  $$

  with normalization $\sum_{v} a_{uv} = 1$ for all $u$.

  - _Affine gap penalty:_ The matrix structure supports distinguishing gap opening ($\delta$) and gap extension ($\epsilon$) probabilities.
  - _Constraint:_ $a_{XY} = a_{YX} = 0$ (no direct transitions between insert states).

- **End distribution**:
  $$
  \rho(s) = P(\text{end} \mid z_T = s), \quad s \in \{M, X, Y\}
  $$

#### 2.1.3 Joint Probability and Log-Space Parameterization

For any valid alignment path $Z = (z_1, \ldots, z_T)$, the **joint probability** is:

$$
P(X, Y, Z) = \pi(z_1) \left[ \prod_{t=1}^{T-1} a_{z_t z_{t+1}} \right] \left[ \prod_{t=1}^{T} e_{z_t}(\text{symbols emitted at } t) \right] \rho(z_T)
$$

The **marginal sequence likelihood** sums over all valid alignment paths:

$$
P(X, Y) = \sum_{Z \in \mathcal{Z}(X,Y)} P(X, Y, Z)
$$

where $\mathcal{Z}(X,Y)$ is the set of all alignment paths consuming both sequences.

**Log-space implementation.**  
For numerical stability, all probabilities are represented in log-space:

- $\log \pi(s)$, $\log a_{uv}$, $\log \rho(s)$,
- $\log e_M(x,y)$, $\log e_X(x)$, $\log e_Y(y)$,

The log-joint decomposes as:

$$
\log P(X, Y, Z)
= \log \pi(z_1)
+ \sum_{t=1}^{T-1} \log a_{z_t z_{t+1}}
+ \sum_{t=1}^{T} \log e_{z_t}(\text{symbols emitted at } t)
+ \log \rho(z_T).
$$

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

### 2.4 Viterbi Decoding (MAP Alignment)

#### 2.4.1 Objective

Given the pair HMM of Section 2.1, the maximum a posteriori (MAP) alignment is

$$
Z^* = \arg\max_{Z \in \mathcal{Z}(X,Y)} P(X, Y, Z)
    = \arg\max_{Z \in \mathcal{Z}(X,Y)} \log P(X, Y, Z),
$$

where $\mathcal{Z}(X,Y)$ is the set of all valid alignment paths. We perform this maximization via dynamic programming in log-space.

#### 2.4.2 Dynamic Programming Variables

For $0 \le i \le L_X$ and $0 \le j \le L_Y$, define

$$
V^s_{i,j}
=
\max_{Z: (i_T,j_T)=(i,j),\,z_T = s}
\log P\bigl(X_{1:i}, Y_{1:j}, Z\bigr),
\qquad s \in \{M,X,Y\},
$$

with the convention $V^s_{i,j} = -\infty$ if no such path exists. We work entirely in log-space:
$\log \pi(s)$, $\log a_{uv}$, $\log \rho(s)$, $\log e_M(x,y)$, $\log e_X(x)$, $\log e_Y(y)$.

#### 2.4.3 Initialization

Anchor at empty prefixes $(0,0)$: $V^M_{0,0} = V^X_{0,0} = V^Y_{0,0} = -\infty$.

First non-empty cells:

- $V^X_{1,0} = \log \pi(X) + \log e_X(x_1)$
- $V^Y_{0,1} = \log \pi(Y) + \log e_Y(y_1)$
- $V^M_{1,1} = \log \pi(M) + \log e_M(x_1, y_1)$

Leading gaps via gap-extension transitions:

- First column ($j=0$, gaps in $Y$), $i = 2,\dots,L_X$:
  $$
  V^X_{i,0}
  =
  \log e_X(x_i)
  +
  \max\{ V^M_{i-1,0} + \log a_{MX},\; V^X_{i-1,0} + \log a_{XX} \}.
  $$
- First row ($i=0$, gaps in $X$), $j = 2,\dots,L_Y$:
  $$
  V^Y_{0,j}
  =
  \log e_Y(y_j)
  +
  \max\{ V^M_{0,j-1} + \log a_{MY},\; V^Y_{0,j-1} + \log a_{YY} \}.
  $$

Unreachable cells remain $-\infty$.

#### 2.4.4 Recurrence Relations

For $1 \le i \le L_X$, $1 \le j \le L_Y$:

- Match $M$ (consumes $x_i, y_j$):
  $$
  V^M_{i,j}
  =
  \log e_M(x_i, y_j)
  +
  \max\{
    V^M_{i-1,j-1} + \log a_{MM},
    V^X_{i-1,j-1} + \log a_{XM},
    V^Y_{i-1,j-1} + \log a_{YM}
  \}.
  $$
- Insert $X$ (consumes $x_i$ only):
  $$
  V^X_{i,j}
  =
  \log e_X(x_i)
  +
  \max\{
    V^M_{i-1,j} + \log a_{MX},
    V^X_{i-1,j} + \log a_{XX}
  \}.
  $$
- Insert $Y$ (consumes $y_j$ only):
  $$
  V^Y_{i,j}
  =
  \log e_Y(y_j)
  +
  \max\{
    V^M_{i,j-1} + \log a_{MY},
    V^Y_{i,j-1} + \log a_{YY}
  \}.
  $$

#### 2.4.5 Termination and MAP Score

At $(L_X, L_Y)$, incorporate end probabilities:

$$
\ell^* = \max_{s \in \{M,X,Y\}} \big( V^s_{L_X,L_Y} + \log \rho(s) \big),
$$

with best final state
$s^* = \arg\max_{s \in \{M,X,Y\}} \big( V^s_{L_X,L_Y} + \log \rho(s) \big)$.
The Viterbi log-score of the optimal alignment is $\ell^*$.

#### 2.4.6 Traceback

Traceback is performed by following the backpointers from $(L_X, L_Y, s^*)$, emitting aligned pairs and moving to the previous cell according to the current state, until reaching $(0, 0)$. The resulting sequence is then reversed to yield the optimal alignment.

### 2.5 Maximum Expected Accuracy (MEA) Alignment

Given the pair HMM in Section 2.1, we seek an alignment that maximizes the expected number of correctly aligned nucleotide pairs under the posterior distribution over alignment paths.

#### 2.5.1 Posterior match probabilities

Let $L_X$ and $L_Y$ be the lengths of sequences $X$ and $Y$, respectively. For each pair of positions $(i,j)$, with $1 \le i \le L_X$, $1 \le j \le L_Y$, define the event

$$
M_{ij} = 1 \;\Longleftrightarrow\; x_i \text{ is aligned to } y_j \text{ in state } M \text{ along some valid path } Z.
$$

The posterior match probability is

$$
P_{ij} \equiv P(M_{ij} = 1 \mid X, Y).
$$

Using the forward–backward recursions of the pair HMM, let

- $\alpha_M(i,j)$: forward probability of generating prefixes $x_{1:i}, y_{1:j}$ and being in state $M$ that aligns $(x_i, y_j)$,
- $\beta_M(i,j)$: backward probability of generating suffixes $x_{i+1:L_X}, y_{j+1:L_Y}$ from state $M$ at $(i,j)$,
  and let $P(X,Y) = \sum_{Z} P(X,Y,Z)$ be the marginal likelihood. Then
  $$
  P_{ij} = \frac{\alpha_M(i,j)\,\beta_M(i,j)}{P(X,Y)}.
  $$
  Collect these into $P \in [0,1]^{(L_X+1)\times(L_Y+1)}$ with $P[0,\cdot]=P[\cdot,0]=0$.

#### 2.5.2 MEA objective and weighting schemes

An alignment $A$ is the set of aligned index pairs $A \subseteq \{1,\dots,L_X\}\times\{1,\dots,L_Y\}$ where $(i,j)\in A$ iff $x_i$ is aligned to $y_j$. Its expected accuracy is

$$
\mathbb{E}[\text{acc}(A) \mid X,Y] = \sum_{(i,j)\in A} P_{ij}.
$$

We generalize with weights $w_{ij} = f(P_{ij};\gamma)$ and maximize

$$
A^\star = \arg\max_{A} \sum_{(i,j)\in A} w_{ij}.
$$

Weighting choices:

1. **Power** ($\gamma>0$): $w_{ij}=P_{ij}^{\gamma}$
2. **Threshold** ($\gamma\in(0,1]$): $w_{ij}=P_{ij}-(1-\gamma)$
3. **ProbCons-style** ($\gamma>0.5$): $w_{ij}=2\gamma P_{ij}-1$
4. **Log-odds** ($\gamma\in(0,1)$): $w_{ij}=\log\frac{P_{ij}}{1-P_{ij}} + \log\frac{\gamma}{1-\gamma}$  
   Let $W$ be the weight matrix with $W[0,\cdot]=W[\cdot,0]=0$.

#### 2.5.3 Dynamic programming formulation

View MEA as a maximum-weight path on the grid $\{0,\dots,L_X\}\times\{0,\dots,L_Y\}$:

- Diagonal $(i-1,j-1)\to(i,j)$ aligns $(x_i,y_j)$ with weight $W[i,j]$.
- Horizontal/vertical steps are gaps with zero weight (no explicit gap penalties).

Define $D(i,j)$ as the maximum total weight for prefixes $x_{1:i}, y_{1:j}$.
Boundary: $D(0,0)=0,\; D(i,0)=0,\; D(0,j)=0$.
For $1 \le i \le L_X,\; 1 \le j \le L_Y$:

$$
D(i,j) = \max\{ D(i-1,j-1)+W[i,j],\; D(i-1,j),\; D(i,j-1) \}.
$$

MEA score: $\text{MEA}(X,Y)=D(L_X,L_Y)$.

#### 2.5.4 Traceback and alignment reconstruction

Store the maximizing move (diag/up/left) at each $(i,j)$. Trace back from $(L_X,L_Y)$ to $(0,0)$ to produce the MEA alignment under the pair HMM.

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
