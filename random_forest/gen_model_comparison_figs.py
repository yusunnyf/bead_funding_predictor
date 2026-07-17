import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shutil

HAS_LATEX = shutil.which('pdflatex') is not None

plt.rcParams.update({
    'text.usetex': HAS_LATEX,
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
    'font.size': 9,
    'mathtext.fontset': 'cm',
    'figure.dpi': 110,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# ── Data ──────────────────────────────────────────────────────────────────────
models = [
    ('XGBoost + Interactions',   0.7515, 0.7668, 0.0145, 2458, 1413),
    ('RF Tuned',                 0.7502, 0.7705, 0.0139, 2444, 1399),
    ('RF Tuned + Interactions',  0.7488, 0.7645, 0.0135, 2468, 1413),
    ('XGBoost (6 feat)',         0.7472, 0.7636, 0.0154, 2452, 1412),
    ('GBR Tuned',                0.7422, 0.7535, 0.0123, 2496, 1455),
    ('RF Baseline',              0.7419, 0.7622, 0.0176, 2491, 1420),
    ('RF + Interactions',        0.7410, 0.7593, 0.0184, 2488, 1422),
    ('GBR + Interactions',       0.7408, 0.7500, 0.0127, 2527, 1478),
    ('GBR Default',              0.7331, 0.7511, 0.0190, 2527, 1466),
]

names  = [m[0] for m in models]
r2     = np.array([m[1] for m in models])
cv_r2  = np.array([m[2] for m in models])
cv_std = np.array([m[3] for m in models])
rmse   = np.array([m[4] for m in models])
baseline_r2 = 0.7419

def color(name):
    if 'XGB' in name: return '#E6A817'
    if 'GBR' in name: return '#C0392B'
    return '#2471A3'

colors = [color(n) for n in names]

# ── Figure 1: R² comparison bar chart ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.5, 4.2))
y = np.arange(len(names))
bars = ax.barh(y, r2, color=colors, alpha=0.85, height=0.6)
ax.errorbar(cv_r2, y, xerr=cv_std, fmt='none', color='#333', capsize=3,
            linewidth=0.9, label='CV R$^2$ $\\pm$ 1 SD')
ax.axvline(baseline_r2, color='black', lw=0.9, linestyle='--', alpha=0.6,
           label=f'Baseline R$^2$ = {baseline_r2:.4f}')
ax.set_yticks(y)
ax.set_yticklabels(names, fontsize=8)
ax.set_xlabel('$R^2$ (test set)')
ax.set_title('Model Comparison: Test $R^2$ with 5-Fold CV', weight='bold')
ax.set_xlim(0.70, 0.78)
for bar, val in zip(bars, r2):
    ax.text(val + 0.0003, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=7)
legend_patches = [
    mpatches.Patch(color='#2471A3', label='Random Forest'),
    mpatches.Patch(color='#C0392B', label='Gradient Boosting'),
    mpatches.Patch(color='#E6A817', label='XGBoost'),
]
ax.legend(handles=legend_patches + [
    plt.Line2D([0],[0], color='#333', lw=0.9, label='CV R$^2$ $\\pm$ 1 SD'),
    plt.Line2D([0],[0], color='black', lw=0.9, ls='--', alpha=0.6, label=f'Baseline'),
], fontsize=7, loc='lower right')
plt.tight_layout()
plt.savefig('fig_model_comparison_r2.pdf')
plt.savefig('fig_model_comparison_r2.png')
plt.close()
print('Saved fig_model_comparison_r2')

# ── Figure 2: RMSE comparison ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.5, 4.2))
bars = ax.barh(y, rmse, color=colors, alpha=0.85, height=0.6)
ax.axvline(2491, color='black', lw=0.9, linestyle='--', alpha=0.6,
           label='Baseline RMSE = \\$2,491')
ax.set_yticks(y)
ax.set_yticklabels(names, fontsize=8)
ax.set_xlabel('RMSE (\\$, funding per location)')
ax.set_title('Model Comparison: RMSE on Test Set', weight='bold')
ax.set_xlim(2300, 2600)
for bar, val in zip(bars, rmse):
    ax.text(val + 3, bar.get_y() + bar.get_height()/2,
            f'\\${val:,}', va='center', fontsize=7)
legend_patches = [
    mpatches.Patch(color='#2471A3', label='Random Forest'),
    mpatches.Patch(color='#C0392B', label='Gradient Boosting'),
    mpatches.Patch(color='#E6A817', label='XGBoost'),
]
ax.legend(handles=legend_patches + [
    plt.Line2D([0],[0], color='black', lw=0.9, ls='--', alpha=0.6, label='Baseline'),
], fontsize=7, loc='lower right')
plt.tight_layout()
plt.savefig('fig_model_comparison_rmse.pdf')
plt.savefig('fig_model_comparison_rmse.png')
plt.close()
print('Saved fig_model_comparison_rmse')

# ── Figure 3: R² gain over baseline ───────────────────────────────────────────
delta = r2 - baseline_r2
fig, ax = plt.subplots(figsize=(6.5, 3.6))
bar_colors = ['#27AE60' if d > 0 else '#C0392B' for d in delta]
bars = ax.barh(y, delta * 100, color=bar_colors, alpha=0.85, height=0.6)
ax.axvline(0, color='black', lw=0.8)
ax.set_yticks(y)
ax.set_yticklabels(names, fontsize=8)
ax.set_xlabel('$\\Delta R^2$ vs Baseline RF (percentage points)')
ax.set_title('$R^2$ Gain Over RF Baseline', weight='bold')
for bar, val in zip(bars, delta):
    xpos = bar.get_width() + (0.005 if val >= 0 else -0.005)
    ha = 'left' if val >= 0 else 'right'
    ax.text(xpos, bar.get_y() + bar.get_height()/2,
            f'{val*100:+.2f}pp', va='center', fontsize=7, ha=ha)
plt.tight_layout()
plt.savefig('fig_model_comparison_delta.pdf')
plt.savefig('fig_model_comparison_delta.png')
plt.close()
print('Saved fig_model_comparison_delta')

print(f'\nLaTeX rendered: {HAS_LATEX}')
