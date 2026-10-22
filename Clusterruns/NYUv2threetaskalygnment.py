

"""
NYUv2 joint three-task (depth, normal, segmentation) alignment analysis.

Extends the pairwise depth<->segmentation alignment computation (as given)
to the full N_T=3 joint alignment (Definition~definitionalimulti,
Appendix~riccimoretasks):
    - per-task input-output correlation matrices and their SVDs, all
      computed over the SAME segmentation-derived valid-location mask
    - pairwise alignment (sanity check against Table~tab:alignment)
    - joint alignment eigenvalues: eigenvalues of Obar = sum_i V_i V_i^T,
      via the small (k1+k2+k3)x(k1+k2+k3) Gram matrix
    - per-task loadings on each joint mode (which tasks share which
      direction), read directly off that Gram matrix's own eigenvectors
    - a permutation-based significance test for the leading joint eigenvalue
    - stability of the joint spectrum across the PCA dimension d'

============================================================================
CONFIGURATION / ASSUMPTIONS TO VERIFY BEFORE RUNNING ON THE CLUSTER
============================================================================
Depth and segmentation preprocessing below follow your provided code
exactly (nearest-neighbour downsample of segmentation, ignore-label mask,
per-location one-hot at coarse resolution, same mask applied to depth).

Normal-target preprocessing was NOT provided with masking, only the
original unmasked depth<->normal code was. This script masks F_normal the
same way as F_depth (by segmentation's mask_flat), since:
  (a) the joint V_stack construction (Section 8) requires all three tasks'
      correlation matrices to be computed over the identical sample set,
  (b) your depth<->seg code already establishes this convention for depth.
Please verify this is the right choice for normal specifically -- e.g. if
normal maps have their own independent invalid regions (unlikely in NYUv2's
processed version, but not confirmed here), those should also be excluded.

Items marked [VERIFY]:
  1. ROOT                -- NYUv2 dataset root directory
  2. tasks=(...) string  -- exact task names your `datasets` module expects
  3. ignore-label value  -- assumed -1, matching your segmentation code
  4. normal masking       -- see paragraph above
============================================================================
"""

import numpy as np
import torch
import torch.nn.functional as F
import datasets

torch.manual_seed(0)
np.random.seed(0)

ROOT = "/nfs/scistore13/mondegrp/egregoir/Realistic_Hybrid/LibMTL/nyuv2"          # [VERIFY]
NUM_CLASSES = 13
SCALE = 16
N_PERM = 20


def log(msg):
    print(msg, flush=True)


# ============================================================
# 1. LOAD DATASET (all three tasks)
# ============================================================
log("=" * 70)
log("1. Loading dataset")
log("=" * 70)

X, Y_seg, Y_depth, Y_normal = datasets.get_nyuv2_tasks(   # [VERIFY]
    root=ROOT,
    batch_size=636,
    mode="train",
    tasks=("segmentation", "depth", "normal"),              # [VERIFY]
    mtl_format=False,
)

X        = torch.as_tensor(X,        dtype=torch.float32)   # [N, 3, H, W]
Y_seg    = torch.as_tensor(Y_seg,    dtype=torch.long)       # [N, H, W]
Y_depth  = torch.as_tensor(Y_depth,  dtype=torch.float32)    # [N, 1, H, W]
Y_normal = torch.as_tensor(Y_normal, dtype=torch.float32)    # [N, 3, H, W]

N, _, H, W = X.shape
log(f"Dataset: {N} images, {H}x{W}")
log(f"Y_seg    shape: {tuple(Y_seg.shape)}, dtype: {Y_seg.dtype}, "
    f"value range: [{Y_seg.min().item()}, {Y_seg.max().item()}]")
log(f"Y_depth  shape: {tuple(Y_depth.shape)}")
log(f"Y_normal shape: {tuple(Y_normal.shape)}")
assert X.ndim == 4 and X.shape[1] == 3
assert Y_seg.ndim == 3
assert Y_depth.ndim == 4 and Y_depth.shape[1] == 1
assert Y_normal.ndim == 4 and Y_normal.shape[1] == 3
# ============================================================
# 2. PCA ON INPUT (shared across all three tasks)
# ============================================================
log("\n" + "=" * 70)
log("2. PCA on input")
log("=" * 70)

X_flat = X.reshape(N, -1)
X_mean = X_flat.mean(dim=0)
X_c = X_flat - X_mean
U, S, Vt = torch.linalg.svd(X_c, full_matrices=False)

explained = (S ** 2) / (S ** 2).sum()
cumulative = explained.cumsum(dim=0)
d_prime = (cumulative < 0.95).sum().item() + 1
log(f"d' (95% variance): {d_prime}")

Z = U[:, :d_prime]                      # [N, d'], per-image PCA scores

SCALE_ = SCALE
H_s, W_s = H // SCALE_, W // SCALE_
P_s = H_s * W_s
log(f"Downsampled spatial grid: {H_s}x{W_s} = {P_s} locations")

Z_ds = Z.unsqueeze(1).expand(N, P_s, d_prime).reshape(N * P_s, d_prime)

# ============================================================
# 3. SEGMENTATION -- nearest-neighbour downsample + ignore-label mask
# ============================================================
log("\n" + "=" * 70)
log("3. Segmentation target (downsample + ignore-label mask)")
log("=" * 70)

Y_seg_ds = F.interpolate(
    Y_seg.float().unsqueeze(1), size=(H_s, W_s), mode="nearest"
).squeeze(1).long()                                    # [N, H_s, W_s]

mask_ds = (Y_seg_ds != -1)                              # [VERIFY] ignore-label value
mask_flat = mask_ds.reshape(N * P_s)
log(f"Valid coarse locations: {mask_ds.sum().item()} / {N * P_s} "
    f"({100 * mask_ds.float().mean().item():.1f}%)")

Y_seg_flat = Y_seg_ds.reshape(N * P_s)
Y_seg_valid = Y_seg_flat[mask_flat]
M = mask_flat.sum().item()
log(f"Samples after masking: {M}")

F_seg_valid = F.one_hot(Y_seg_valid, num_classes=NUM_CLASSES).float()   # [M, 13]
log(f"F_seg_valid variance: {F_seg_valid.var().item():.4f}")

# ============================================================
# 4. DEPTH -- masked by segmentation validity (matches provided code)
# ============================================================
log("\n" + "=" * 70)
log("4. Depth target")
log("=" * 70)

F_depth_ds = F.avg_pool2d(Y_depth, kernel_size=SCALE, stride=SCALE)      # [N,1,H_s,W_s]
F_depth_full = F_depth_ds.permute(0, 2, 3, 1).reshape(N, P_s, 1)         # kept for permutation test
F_depth_flat = F_depth_full.reshape(N * P_s, 1)
F_depth_valid = F_depth_flat[mask_flat]
log(f"F_depth_valid variance: {F_depth_valid.var().item():.4f}")

# ============================================================
# 5. NORMAL -- masked the same way as depth  [VERIFY, see module docstring]
# ============================================================
log("\n" + "=" * 70)
log("5. Normal target  [masking convention extrapolated -- verify]")
log("=" * 70)

F_normal_ds = F.avg_pool2d(Y_normal, kernel_size=SCALE, stride=SCALE)    # [N,3,H_s,W_s]
F_normal_full = F_normal_ds.permute(0, 2, 3, 1).reshape(N, P_s, 3)       # kept for permutation test
F_normal_flat = F_normal_full.reshape(N * P_s, 3)
F_normal_valid = F_normal_flat[mask_flat]
log(f"F_normal_valid variance: {F_normal_valid.var().item():.4f}")

# ============================================================
# 6. SHARED MASKED PCA FEATURES + MEAN-CENTRING (all three tasks)
# ============================================================
Z_valid = Z_ds[mask_flat]
Z_c = Z_valid - Z_valid.mean(dim=0)

Fs_c = F_seg_valid - F_seg_valid.mean(dim=0)
Fd_c = F_depth_valid - F_depth_valid.mean(dim=0)
Fn_c = F_normal_valid - F_normal_valid.mean(dim=0)

# ============================================================
# 7. CROSS-COVARIANCE + SVD, all three tasks, same M-sample set
# ============================================================
log("\n" + "=" * 70)
log("7. Cross-covariance and SVD")
log("=" * 70)

Sigma_seg    = (Fs_c.T @ Z_c) / M    # [13, d']
Sigma_depth  = (Fd_c.T @ Z_c) / M    # [1,  d']
Sigma_normal = (Fn_c.T @ Z_c) / M    # [3,  d']

log(f"||Sigma_seg||_F:    {Sigma_seg.norm().item():.6f}")
log(f"||Sigma_depth||_F:  {Sigma_depth.norm().item():.6f}")
log(f"||Sigma_normal||_F: {Sigma_normal.norm().item():.6f}")

_, S_seg,    Vt_seg    = torch.linalg.svd(Sigma_seg,    full_matrices=False)
_, S_depth,  Vt_depth  = torch.linalg.svd(Sigma_depth,  full_matrices=False)
_, S_normal, Vt_normal = torch.linalg.svd(Sigma_normal, full_matrices=False)
V_seg    = Vt_seg.T       # [d', 13]
V_depth  = Vt_depth.T     # [d', 1]
V_normal = Vt_normal.T    # [d', 3]

log(f"S_seg:    {S_seg}")
log(f"S_depth:  {S_depth}")
log(f"S_normal: {S_normal}")

# ============================================================
# 8. PAIRWISE ALIGNMENT -- sanity check against Table~tab:alignment
# ============================================================
log("\n" + "=" * 70)
log("8. Pairwise alignment (sanity check)")
log("=" * 70)


def pairwise_alpha(Va, Vb):
    return torch.linalg.svdvals(Va.T @ Vb)


a_ds = pairwise_alpha(V_depth, V_seg)
a_ns = pairwise_alpha(V_normal, V_seg)
a_dn = pairwise_alpha(V_depth, V_normal)
log(f"depth  <-> seg    : alpha_bar = {a_ds.mean().item():.4f}  (values: {a_ds})")
log(f"normal <-> seg    : alpha_bar = {a_ns.mean().item():.4f}  (values: {a_ns})")
log(f"depth  <-> normal : alpha_bar = {a_dn.mean().item():.4f}  (values: {a_dn})")

# ============================================================
# 9. JOINT THREE-TASK ALIGNMENT (Definition~definitionalimulti)
# ============================================================
log("\n" + "=" * 70)
log("9. Joint three-task alignment")
log("=" * 70)

V_stack = torch.cat([V_depth, V_normal, V_seg], dim=1)      # [d', 1+3+13=17]
task_names = ["depth", "normal", "seg"]
task_sizes = [V_depth.shape[1], V_normal.shape[1], V_seg.shape[1]]
splits = torch.cumsum(torch.tensor([0] + task_sizes), dim=0)
log(f"Task dimensions k_i: depth={task_sizes[0]}, normal={task_sizes[1]}, seg={task_sizes[2]}")

# Efficient: eigendecompose the small Gram matrix, not the d' x d' Obar --
# shares the same nonzero eigenvalues (standard A^T A / A A^T duality).
G = V_stack.T @ V_stack
eigvals, E = torch.linalg.eigh(G)
eigvals, E = eigvals.flip(0), E.flip(1)   # descending

log("\nJoint alignment eigenvalues (lambda in [0,3]; "
    "3=all tasks share, 1=idiosyncratic, 0=absent):")
log(f"{eigvals}")

log("\nPer-mode task loadings (squared block-norms, sum to 1 per mode):")
log(f"{'mode':>4}  {'lambda':>7}  {'depth':>7}  {'normal':>7}  {'seg':>7}")
for j in range(len(eigvals)):
    lam = eigvals[j].item()
    if lam < 1e-6:
        continue
    e = E[:, j]
    load = [(e[splits[i]:splits[i + 1]].norm() ** 2).item() for i in range(3)]
    log(f"{j:4d}  {lam:7.4f}  {load[0]:7.4f}  {load[1]:7.4f}  {load[2]:7.4f}")

# ============================================================
# 10. PERMUTATION BASELINE for the leading joint eigenvalue
# ============================================================
log("\n" + "=" * 70)
log(f"10. Permutation baseline ({N_PERM} shuffles; seg held fixed as reference)")
log("=" * 70)
log("Follows the provided code's convention exactly: shuffle images, apply")
log("the ORIGINAL (unpermuted) mask_flat, recompute V for the shuffled task.")

perm_lambda1 = []
for seed in range(N_PERM):
    torch.manual_seed(seed)

    perm_d = torch.randperm(N)
    Fd_perm = F_depth_full[perm_d].reshape(N * P_s, 1)[mask_flat]
    Fd_c_perm = Fd_perm - Fd_perm.mean(dim=0)
    Sigma_d_perm = (Fd_c_perm.T @ Z_c) / M
    _, _, Vtd_p = torch.linalg.svd(Sigma_d_perm, full_matrices=False)

    perm_n = torch.randperm(N)
    Fn_perm = F_normal_full[perm_n].reshape(N * P_s, 3)[mask_flat]
    Fn_c_perm = Fn_perm - Fn_perm.mean(dim=0)
    Sigma_n_perm = (Fn_c_perm.T @ Z_c) / M
    _, _, Vtn_p = torch.linalg.svd(Sigma_n_perm, full_matrices=False)

    V_stack_p = torch.cat([Vtd_p.T, Vtn_p.T, V_seg], dim=1)   # seg held fixed
    G_p = V_stack_p.T @ V_stack_p
    ev_p = torch.linalg.eigvalsh(G_p).flip(0)
    perm_lambda1.append(ev_p[0].item())

perm_mean, perm_std = float(np.mean(perm_lambda1)), float(np.std(perm_lambda1))
observed = eigvals[0].item()
sig = (observed - perm_mean) / perm_std if perm_std > 0 else float("nan")
log(f"Leading eigenvalue permutation baseline: {perm_mean:.4f} +/- {perm_std:.4f}")
log(f"Observed leading eigenvalue:              {observed:.4f}  ({sig:.1f} sigma)")

# ============================================================
# 11. STABILITY OF JOINT ALIGNMENT ACROSS d'
# ============================================================
log("\n" + "=" * 70)
log("11. Stability across d'")
log("=" * 70)
header_dprime = "d'"
log(f"{header_dprime:>5} {'var_exp':>8}  {'lambda_1':>9} {'lambda_2':>9} {'lambda_3':>9}")

d_test_list = [25, 50, 75, 100, 150, 200, 275, 325]
for d_test in d_test_list:
    if d_test > U.shape[1]:
        continue
    Z_test = U[:, :d_test]
    Z_ds_test = Z_test.unsqueeze(1).expand(N, P_s, d_test).reshape(N * P_s, d_test)
    Z_valid_test = Z_ds_test[mask_flat]
    Z_c_test = Z_valid_test - Z_valid_test.mean(dim=0)

    Ss = (Fs_c.T @ Z_c_test) / M
    Sd = (Fd_c.T @ Z_c_test) / M
    Sn = (Fn_c.T @ Z_c_test) / M

    _, _, Vts = torch.linalg.svd(Ss, full_matrices=False)
    _, _, Vtd = torch.linalg.svd(Sd, full_matrices=False)
    _, _, Vtn = torch.linalg.svd(Sn, full_matrices=False)
    Vs, Vd, Vn = Vts.T, Vtd.T, Vtn.T

    Vst = torch.cat([Vd, Vn, Vs], dim=1)
    Gt = Vst.T @ Vst
    evt = torch.linalg.eigvalsh(Gt).flip(0)

    var_exp = cumulative[d_test - 1].item()
    log(f"{d_test:5d} {var_exp:8.3f}  {evt[0].item():9.4f} {evt[1].item():9.4f} {evt[2].item():9.4f}")

log("\nDone.")