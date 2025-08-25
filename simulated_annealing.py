import numpy as np
import math, random, time

# Import Q (redundancy) and c (relevance) from mi_result.py
from mutual_i.ml_result import relevance, redundancy

# Convert to numpy arrays
Q = np.array(redundancy)
c = np.array(relevance)

def f(x):
    return 0.5 * x @ Q @ x - c @ x

# 🔹 Set target sum to log2(3)
target_sum = math.log2(3)   # ≈ 1.584962500721156

# ---------- Generalized projection onto the simplex of sum = s ----------
def project_to_simplex(v, s=1.0):
    """
    Projects v onto { x : sum(x) = s, x >= 0 }
    """
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(v)+1) > (cssv - s))[0][-1]
    theta = (cssv[rho] - s) / (rho + 1)
    w = np.maximum(v - theta, 0)
    return w

# ---------- Neighbor: add noise then project ----------
def neighbor(x, T, base_sigma=0.05):
    step = np.random.normal(0.0, base_sigma * (1 + T), size=x.shape)
    return project_to_simplex(x + step, target_sum)

def simulated_annealing_simplex(x0, T0 ,cooling ,iters , inner):
    x  = project_to_simplex(x0, target_sum)
    fx = f(x)
    best_x, best_f = x.copy(), fx
    T = T0

    start_time = time.time()   # ⏱️ start timer

    for k in range(iters):
        for _ in range(inner):
            xn = neighbor(x, T)
            fn = f(xn)
            delta = fn - fx

            if delta <= 0:
                x, fx = xn, fn
            else:
                if random.random() < math.exp(-delta / max(T, 1e-12)):
                    x, fx = xn, fn

        if fx < best_f:
            best_x, best_f = x.copy(), fx

        T *= cooling
        if k % 50 == 0:
            elapsed = time.time() - start_time
            print(f"iter={k:4d}  T={T:.4f}  best_f={best_f:.6f}  "
                  f"x={best_x}  sum={best_x.sum():.6f}  time={elapsed:.2f}s")

    total_time = time.time() - start_time   # ⏱️ end timer
    return best_x, best_f, total_time

# ----- Run: start from uniform point scaled to target_sum -----
np.random.seed(0)
n = len(c)  # automatically match length of relevance
x0 = np.ones(n) / n * target_sum

T0=10000.0
cooling=0.9995
iters=20000
inner=200

best_x, best_f, total_time = simulated_annealing_simplex(
    x0,
    T0,
    cooling,
    iters,
    inner
)

print("\nSA  best x:", best_x,
      " sum:", best_x.sum(),
      "\nobjective:", best_f,
      f"\n⏱️ total runtime: {total_time:.2f} seconds")
