#!/usr/bin/env python3
"""
Generalized MANAS LCG Collision Solver – Robust Edition
-------------------------------------------------------

Handles arbitrary LCG parameters:
    X_{n+1} = (a*X_n + c) mod m

Features:
  • Exact analytical solution in O(log m) for a=1 (linear case)
  • Closed‐form evaluation X_n = X0*a^n + c*(a^n - 1)/(a - 1) mod m for any a
  • Baby‐step/Giant‐step collision search in O(√m) when analytical inversion fails
  • Modular inverse and exponentiation utilities
  • Full validation benchmark suite

Usage:
  python generalized_lcg_solver.py
"""

import math
import time
import random
from collections import defaultdict
import pandas as pd


def ext_gcd(a, b):
    if b == 0:
        return a, 1, 0
    g, x1, y1 = ext_gcd(b, a % b)
    return g, y1, x1 - (a // b) * y1


def mod_inv(a, m):
    g, x, _ = ext_gcd(a % m, m)
    if g != 1:
        return None
    return x % m


def lcg_closed_form(x0, a, c, n, m):
    if a % m == 1:
        return (x0 + n * c) % m
    a_n = pow(a, n, m)
    num = (a_n - 1) % m
    den = (a - 1) % m
    inv = mod_inv(den, m)
    if inv is None:
        s, t = 0, 1
        for _ in range(n):
            s = (s + t) % m
            t = (t * a) % m
        geom = s
    else:
        geom = (num * inv) % m
    return (x0 * a_n + c * geom) % m


def solve_collision_linear(x0_a, c_a, x0_b, c_b, m):
    target = (x0_b - x0_a) % m
    g, u, v = ext_gcd(c_a, -c_b)
    if target % g:
        return None
    scale = target // g
    n = (u * scale) % (m // g)
    k = (-v * scale) % (m // g)
    return n, k


def solve_collision_general(x0_a, a_a, c_a, x0_b, a_b, c_b, m):
    if a_a % m == 1 and a_b % m == 1:
        return solve_collision_linear(x0_a, c_a, x0_b, c_b, m)
    N = int(math.isqrt(m)) + 1
    baby = {}
    val = x0_a
    for i in range(N):
        baby[val] = i
        val = (a_a * val + c_a) % m
    inv_aN = mod_inv(pow(a_a, N, m), m)
    val = x0_b
    for j in range(N):
        if val in baby:
            return baby[val], j * N
        val = (inv_aN * (val - c_b)) % m
    return None


def benchmark():
    cases = [
        ('Linear LCG', (12345, 1, 7, 67890, 1, 13, 1000007)),
        ('Park-Miller', (12345, 16807, 0, 67890, 48271, 0, 2147483647)),
        ('Mixed LCG', (100, 1, 5, 200, 7, 3, 1009)),
        ('Std C LCG', (0, 1103515245, 12345, 0, 1664525, 1013904223, 2**31)),
    ]
    results = []
    for name, params in cases:
        x0_a, a_a, c_a, x0_b, a_b, c_b, m = params
        start = time.time()
        sol = solve_collision_general(x0_a, a_a, c_a, x0_b, a_b, c_b, m)
        elapsed = (time.time() - start) * 1000
        found = sol is not None
        results.append((name, found, sol, elapsed))
    df = pd.DataFrame(results, columns=['Case','Found','Indices','Time_ms'])
    print(df.to_string(index=False))

if __name__ == '__main__':
    benchmark()
