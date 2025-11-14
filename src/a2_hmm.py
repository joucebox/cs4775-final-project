#!/usr/bin/env python3

'''Script for computing posterior probabilities of hidden states at each
   position of a given sequence.
Arguments:
    -f: file containing the sequence (fasta file)
    -mu: the probability of switching states

Outputs:
    posteriors.csv - a KxL matrix outputted as a CSV with the posterior
                     probability of each state at each position

Example Usage:
    python 2b.py -f hmm-sequence.fasta -mu 0.01
'''

import argparse
import numpy as np
import math
import json


'''Computes the log(exp(a) + exp(b)) with better numerical stability'''
def sumLogProbs(a,b):
  if a > b: 
    return a + np.log1p(math.exp(b-a))
  else:
    return b + np.log1p(math.exp(a-b))
  

'''Reads the fasta file and outputs the sequence to analyze.
Arguments:
	filename: name of the fasta file
Returns:
	s: string with relevant sequence
'''
def read_fasta(filename):
    with open(filename, "r") as f:
        s = ""
        for l in f.readlines()[1:]:
            s += l.strip()
    return s


def logsumexp_list(values):
    """Stable log-sum-exp over an iterable of log-values using sumLogProbs."""
    it = iter(values)
    try:
        s = next(it)
    except StopIteration:
        return -np.inf
    for v in it:
        s = sumLogProbs(s, v)
    return s

''' Outputs the forward and backward probabilities of a given observation.
Arguments:
    obs (string): A sequence of observed emissions.
    trans_probs (dict of dicts): Transition log-probabilities between hidden states. 
        The keys of the outer dictionary represent the current hidden state, and the inner keys represent the next hidden state.
        For example, `trans_probs[i][j]` gives the log-probability of transitioning from state `i` to state `j`.
    emiss_probs (dict of dicts): Emission log-probabilities for each hidden state. 
        The outer keys are hidden states, and the inner keys are observed states (emissions). 
        For example, `emiss_probs[j][k]` gives the log-probability of emitting state `k` from hidden state `j`.
    init_probs (dict): Initial log-probabilities for each hidden state.

Returns:
    F (dict): A dictionary of forward log-probabilities. Each key is a hidden state, and each value is a NumPy array of shape `(N)`, where `N` is the length of the observed sequence.
    like_f (float): The log-likelihood of the observation sequence as computed by the forward algorithm, i.e., `log P(obs)`.
    B (dict): A dictionary of backward log-probabilities. Each key is a hidden state, and each value is a NumPy array of shape `(N)`.
    like_b (float): The log-likelihood of the observation sequence as computed by the backward algorithm, i.e., `log P(obs)`.
    R (dict): A dictionary of posterior probabilities (in normal scale, not log-scale). Each key is a hidden state, and each value is a NumPy array of shape `(N)`.

Requirements:
    - F (forward log-probabilities), B (backward log-probabilities), and R (posterior probabilities) are dictionaries with keys as hidden states and values as numpy arrays of shape (N), where N represent the number of emissions (i.e., the length of the observed sequence).
'''
def forward_backward(obs, trans_probs, emiss_probs, init_probs):
    ''' Complete this function. 
    '''
    # Please follow the data format design below
    N = len(obs)                                       # Number of emissions (N)
    K = len(init_probs)                                # Number of hidden states (K)
    F = {state: np.full(N, None) for state in init_probs}   # Forward log-probabilities
    B = {state: np.full(N, None) for state in init_probs}   # Backward log-probabilities
    R = {state: np.full(N, None) for state in init_probs}   # Posterior probabilities (non-log scale)
    
    if isinstance(obs, str):
        x = list(obs)
    else:
        x = obs[:]

    states = list(init_probs.keys())  # e.g., ['h','l']
    

    # ----- Forward pass -----
    # Init t=0
    for s in states:
        F[s][0] = init_probs[s] + emiss_probs[s][x[0]]

    # Recur t=1..N-1
    for t in range(1, N):
        xt = x[t]
        for s in states:
            incoming = [F[sp][t-1] + trans_probs[sp][s] for sp in states]
            F[s][t] = logsumexp_list(incoming) + emiss_probs[s][xt]

    # Forward termination: log P(x)
    like_f = logsumexp_list([F[s][N-1] for s in states])

    # ----- Backward pass -----
    # Init t=N-1: B[s][N-1] = log 1 = 0
    for s in states:
        B[s][N-1] = 0.0

    # Recur t=N-2..0
    for t in range(N-2, -1, -1):
        xt1 = x[t+1]  # emission at t+1
        for s in states:
            outgoing = [trans_probs[s][sn] + emiss_probs[sn][xt1] + B[sn][t+1] for sn in states]
            B[s][t] = logsumexp_list(outgoing)

    # Backward termination: log P(x)
    like_b = logsumexp_list([init_probs[s] + emiss_probs[s][x[0]] + B[s][0] for s in states])

    # ----- Posteriors -----
    # For each t, denom_t = log sum_k F[k][t] * B[k][t]  (in log-domain: logsumexp of F+B)
    for t in range(N):
        denom_t = logsumexp_list([F[s][t] + B[s][t] for s in states])
        for s in states:
            # Posterior in normal scale: exp( log numerator - log denominator )
            R[s][t] = math.exp((F[s][t] + B[s][t]) - denom_t)

    return F, like_f, B, like_b, R


def main():
    parser = argparse.ArgumentParser(
        description='Compute posterior probabilities at each position of a given sequence.')
    parser.add_argument('-f', action="store", dest="f", type=str, required=True)
    parser.add_argument('-mu', action="store", dest="mu", type=float, required=True)

    args = parser.parse_args()
    fasta_file = args.f
    mu = args.mu

    obs_sequence = read_fasta(fasta_file)
    transition_probabilities = {
        'h': {'h': np.log(1 - mu), 'l': np.log(mu)},
        'l': {'h': np.log(mu), 'l': np.log(1 - mu)}
    }
    emission_probabilities = {
        'h': {'A': np.log(0.13), 'C': np.log(0.37), 'G': np.log(0.37), 'T': np.log(0.13)},
        'l': {'A': np.log(0.32), 'C': np.log(0.18), 'G': np.log(0.18), 'T': np.log(0.32)}
    }
    initial_probabilities = {'h': np.log(0.5), 'l': np.log(0.5)}
    F, like_f, B, like_b, R = forward_backward(obs_sequence,
                                              transition_probabilities,
                                              emission_probabilities,
                                              initial_probabilities)
    
    rounded_F = json.dumps({k: [round(float(v), 6) for v in vals] for k, vals in F.items()}, indent=2)
    rounded_B = json.dumps({k: [round(float(v), 6) for v in vals] for k, vals in B.items()}, indent=2)

    with open(fasta_file + ".F.json", "w") as f:
        f.write(rounded_F)
    with open(fasta_file + ".B.json", "w") as f:
        f.write(rounded_B)
    
    
    R_arr = np.array([R[state] for state in ['h', 'l']])
    np.savetxt(fasta_file + ".posteriors.csv", R_arr, delimiter=",", fmt='%.6e')
    print("Backward log-likelihood: {:.3f}".format(like_b))
    print("Forward log-likelihood: {:.3f}".format(like_f))


if __name__ == "__main__":
    main()
