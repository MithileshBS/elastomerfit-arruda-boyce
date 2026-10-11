import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================
# 0) Settings
# ==========================
CSV_PATH = "Exp_dataWM.csv"
LAMBDA_COL = "Strain"
STRESS_COL = "Cauchy-stress"

RNG_SEED = 0
N_STARTS = 120

MU_RANGE = (1e-4, 50.0)
N_RANGE  = (1.0, 500.0)

DAMP0 = 1e-2
MAX_ITER = 250
TOL = 1e-12

# ==========================
# 1) Load data
# ==========================
df = pd.read_csv(CSV_PATH)

lam_all = df[LAMBDA_COL].to_numpy(float)
y_all = df[STRESS_COL].to_numpy(float)

print(f"Loaded {len(lam_all)} points")

mask = np.isfinite(lam_all) & np.isfinite(y_all)

lam = lam_all[mask]
y = y_all[mask]

# ==========================
# 2) Denominator floor
# ==========================
y_floor = 0.01 * np.max(np.abs(y))

print(f"Relative error floor = {y_floor}")

# ==========================
# 3) Arruda–Boyce model
# ==========================
a1 = 1/10
a2 = 11/350
a3 = 19/1750
a4 = 519/134750

def sigma_AB5_cauchy_muN(lam, mu, N):

    lam = np.asarray(lam)

    I1 = lam**2 + 2/lam
    S = lam**2 - lam**(-1)

    B = (
        0.5
        + a1*(I1/N)
        + a2*(I1**2/N**2)
        + a3*(I1**3/N**3)
        + a4*(I1**4/N**4)
    )

    return 2*mu*S*B


# ==========================
# 4) Jacobian (absolute residual)
# ==========================
def jacobian_abs(lam, mu, N):

    I1 = lam**2 + 2/lam
    S = lam**2 - lam**(-1)

    B = (
        0.5
        + a1*(I1/N)
        + a2*(I1**2/N**2)
        + a3*(I1**3/N**3)
        + a4*(I1**4/N**4)
    )

    dB_dN = (
        -a1*(I1/N**2)
        -2*a2*(I1**2/N**3)
        -3*a3*(I1**3/N**4)
        -4*a4*(I1**4/N**5)
    )

    dr_dmu = -2*S*B
    dr_dN  = -2*mu*S*dB_dN

    return np.column_stack([dr_dmu, dr_dN])


# ==========================
# 5) Relative residual with FLOOR
# ==========================
def residuals_relative(lam,y,mu,N):

    sigma = sigma_AB5_cauchy_muN(lam,mu,N)

    denom = np.maximum(np.abs(y),y_floor)

    return (y - sigma)/denom


def jacobian_relative(lam,y,mu,N):

    J_abs = jacobian_abs(lam,mu,N)

    denom = np.maximum(np.abs(y),y_floor)

    return J_abs / denom[:,None]


# ==========================
# 6) LM solver
# ==========================
def lm_fit(lam,y,p0):

    p = np.array(p0)

    damp = DAMP0

    def RSSE(p):

        r = residuals_relative(lam,y,p[0],p[1])

        return r@r

    S = RSSE(p)

    for _ in range(MAX_ITER):

        r = residuals_relative(lam,y,p[0],p[1])

        J = jacobian_relative(lam,y,p[0],p[1])

        JTJ = J.T@J
        g   = J.T@r

        D = np.diag(np.diag(JTJ))

        A = JTJ + damp*D

        try:

            dp = np.linalg.solve(A,g)

        except np.linalg.LinAlgError:

            damp*=10
            continue

        p_try = p + dp

        p_try[0] = max(p_try[0],1e-12)
        p_try[1] = max(p_try[1],1.0)

        S_try = RSSE(p_try)

        if S_try < S:

            if abs(S-S_try) < TOL:
                p,S = p_try,S_try
                break

            p,S = p_try,S_try

            damp/=10

        else:

            damp*=10

    return p,S


# ==========================
# 7) Random multi-start
# ==========================
rng = np.random.default_rng(RNG_SEED)

def sample_starts(n):

    mu = np.exp(rng.uniform(np.log(MU_RANGE[0]),np.log(MU_RANGE[1]),n))

    N  = np.exp(rng.uniform(np.log(N_RANGE[0]),np.log(N_RANGE[1]),n))

    return zip(mu,N)


best=None

for mu0,N0 in sample_starts(N_STARTS):

    p,S = lm_fit(lam,y,(mu0,N0))

    if best is None or S < best[1]:

        best=(p,S,(mu0,N0))


(p_fit,RSSE,start)=best

mu,N=p_fit

lambda_m=np.sqrt(N)


sigma_fit = sigma_AB5_cauchy_muN(lam,mu,N)

SSE = np.sum((y-sigma_fit)**2)

print("\n===== Relative LM Fit (with denominator floor) =====")

print("start used:",start)

print("μ =",mu)

print("N =",N)

print("λm =",lambda_m)

print("RSSE =",RSSE)

print("SSE =",SSE)


# ==========================
# 8) Plot
# ==========================
lam_plot = np.linspace(lam.min(),lam.max(),800)

sigma_plot = sigma_AB5_cauchy_muN(lam_plot,mu,N)

plt.figure(figsize=(10,5))

plt.scatter(lam,y,s=10,label="Experimental")

plt.plot(lam_plot,sigma_plot,label="Arruda-Boyce fit")

plt.title(f"Relative-fit: μ={mu:.4f}, N={N:.4f} (λm={lambda_m:.4f})")

plt.xlabel("Stretch λ")

plt.ylabel("Cauchy stress")

plt.legend()

plt.grid()

plt.show()