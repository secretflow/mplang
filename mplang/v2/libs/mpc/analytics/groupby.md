# Oblivious Group-by Sum Design

This document outlines the design for Oblivious Group-by Sum algorithms in MPLang. The goal is to compute the sum of values in `data` (held by P0) grouped by `bins` (held by P1), such that:
- P0 learns nothing about the `bins` (permutation/grouping).
- P1 learns nothing about the `data` values (except the final aggregated sums).
- The result is revealed to P1 (or shared).

We propose two approaches based on the trade-off between communication and computation, and the cardinality of groups ($K$).

## Interface

```python
def oblivious_groupby_sum(
    data: Plaintext[P0],
    bins: Plaintext[P1],
    K: int,
    method: str = "auto"
) -> Plaintext[P1]:
    """
    Args:
        data: Input data vector held by P0.
        bins: Bin assignments for each data element held by P1.
              Values must be in [0, K).
        K: The number of bins (groups).
        method: "bfv" (HE-based) or "shuffle" (OT-based).

    Returns:
        A vector of length K held by P1 containing the sum of data for each bin.
    """
```

## Approach 1: HE-based (BFV SIMD)

Best for: **Small K** (e.g., $K < 1000$), Low Bandwidth.

### Algorithm

1.  **Encryption (P0)**:
    - P0 encrypts `data` using a BFV scheme with SIMD packing.
    - Sends ciphertext(s) `Enc(data)` to P1.

2.  **Aggregation (P1)**:
    - P1 holds `bins`. For each bin $k \in [0, K)$:
        - Construct a plaintext mask vector $M_k$ where $M_k[i] = 1$ if $bins[i] == k$, else $0$.
        - Compute homomorphic multiplication: $Enc(Sum_k) = Enc(data) \otimes M_k$.
        - Sum the slots in $Enc(Sum_k)$ to get the total sum for bin $k$.
          - *Optimization*: Instead of full slot summation for every bin (which is expensive), P1 can just compute the element-wise product and accumulate. The final reduction can be done by sending back to P0 or using rotations if $K$ is small enough to pack into result ciphertexts.
          - *Simplified Flow*: P1 computes $Enc(Partial_k) = Enc(data) \cdot M_k$. P1 sends these $K$ ciphertexts (or batched versions) back to P0.

3.  **Decryption & Finalize (P0 -> P1)**:
    - P0 decrypts the partial sums.
    - P0 computes the sum of the vector for each bin.
    - P0 sends the final $K$ sums to P1.
    - *Privacy Note*: To prevent P0 from learning the partial sums (which reveals data distribution), P1 should add a random mask to the result before sending to P0, or use a proper threshold decryption if available. For the "Simplified Flow" above, P0 sees the masked data values. This might leak info.
    - *Refined Privacy Flow*:
        - P1 computes $Enc(V_k) = Enc(data) \cdot M_k$.
        - P1 computes $Enc(S_k) = \text{TotalSum}(Enc(V_k))$ using rotations and additions.
        - P1 masks $Enc(S_k)$ with a random value $r_k$: $Enc(O_k) = Enc(S_k) + Enc(r_k)$.
        - P1 sends $Enc(O_k)$ to P0.
        - P0 decrypts to get $O_k = S_k + r_k$ and sends back to P1.
        - P1 subtracts $r_k$ to get $S_k$.

### Complexity
- **Comm**: $O(N/B)$ ciphertexts (P0->P1) + $O(K)$ ciphertexts (P1->P0). ($B$ is batch size).
- **Comp**: $O(K \cdot N/B)$ homomorphic multiplications and additions.

## Approach 2: OT-based (Shuffle + Prefix Sum)

Best for: **Large K**, High Bandwidth.

### Algorithm

1.  **Sort Permutation (P1)**:
    - P1 calculates a permutation $\pi$ that sorts `data` according to `bins`.
    - P1 calculates boundary indices for each bin.

2.  **Oblivious Shuffle (P0, P1)**:
    - Use a Benes network or similar switching network.
    - P0 inputs `data`. P1 inputs control bits derived from $\pi$.
    - Output: Secret shares of permuted data $\langle D' \rangle_0, \langle D' \rangle_1$.

3.  **Secret Shared Prefix Sum (P0, P1)**:
    - Locally compute prefix sums of shares: $\langle S \rangle_0 = \text{cumsum}(\langle D' \rangle_0)$, $\langle S \rangle_1 = \text{cumsum}(\langle D' \rangle_1)$.

4.  **Oblivious Gather (P0, P1)**:
    - P1 knows the boundary indices $idx_k$.
    - P1 needs $S[idx_k] = \langle S \rangle_0[idx_k] + \langle S \rangle_1[idx_k]$.
    - P1 has $\langle S \rangle_1[idx_k]$ locally.
    - To get $\langle S \rangle_0[idx_k]$ obliviously:
        - Use another permutation network or ORAM to fetch these values without revealing $idx_k$ to P0.
        - Or, since P1 is the result receiver, we can use a simpler selection protocol if we don't hide the access pattern from P0 (but we must hide it to protect bin sizes).
        - A second shuffle network mapping $idx_k \to k$ is secure.

5.  **Difference (P1)**:
    - P1 computes $Result[k] = S[idx_k] - S[idx_{k-1}]$.

### Complexity
- **Comm**: $O(N \log N)$ bits for shuffle.
- **Comp**: Low (symmetric crypto).
