import math

def _comb(n: int, k: int) -> int:
    """C(n,k)，n<k 或 k<0 时返回 0"""
    if k < 0 or n < k:
        return 0
    return math.comb(n, k)
    
def _sum_c(m: int, k: int, t_lo: int, t_hi: int) -> int:
    """Σ_{t=t_lo}^{t_hi} C(m-t, k)"""
    total = 0
    for t in range(t_lo, t_hi + 1):
        total += _comb(m - t, k)
    return total

def compute_edge_comb_index(m: int, edges: list[int]) -> int:
    """edges = [E1, E2, ..., E_{n-1}]，已排序 E1 < E2 < ..."""
    n = len(edges) + 1  # num_agents
    if n == 1:
        return 0
    y = _sum_c(m, n - 2, 1, edges[0] - 1)
    for j in range(1, n - 1):
        k = n - 1 - j
        y += _sum_c(m, k, j + 1, edges[j] - 1)
        y -= _sum_c(m, k, j + 1, edges[j - 1])
    return y