from google.cloud import bigquery
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV, LassoCV
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shutil, warnings
warnings.filterwarnings('ignore')

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

# ── Load data (same pipeline as bead_rf_final) ─────────────────────────────────
client = bigquery.Client(project='broadband-data')

df_projects = client.query("""
SELECT project_id, state, bead_support, estimated_miles_aerial_fiber,
       estimated_miles_buried_fiber, estimated_jobs
FROM `broadband-data.fp_approved.deployment_projects`
""").to_dataframe()

df_locations = client.query("""
SELECT project_id, COUNT(*) AS funded_locations,
       SAFE_CAST(APPROX_TOP_COUNT(CAST(technology AS STRING), 1)[OFFSET(0)].value AS FLOAT64) AS technology
FROM `broadband-data.fp_approved.locations`
GROUP BY project_id
""").to_dataframe()

state_name_to_abbr = {
    'Alabama':'AL','Alaska':'AK','Arizona':'AZ','Arkansas':'AR','California':'CA',
    'Colorado':'CO','Connecticut':'CT','Delaware':'DE','Florida':'FL','Georgia':'GA',
    'Hawaii':'HI','Idaho':'ID','Illinois':'IL','Indiana':'IN','Iowa':'IA',
    'Kansas':'KS','Kentucky':'KY','Louisiana':'LA','Maine':'ME','Maryland':'MD',
    'Massachusetts':'MA','Michigan':'MI','Minnesota':'MN','Mississippi':'MS',
    'Missouri':'MO','Montana':'MT','Nebraska':'NE','Nevada':'NV','New_Hampshire':'NH',
    'New_Jersey':'NJ','New_Mexico':'NM','New_York':'NY','North_Carolina':'NC',
    'North_Dakota':'ND','Ohio':'OH','Oklahoma':'OK','Oregon':'OR','Pennsylvania':'PA',
    'Rhode_Island':'RI','South_Carolina':'SC','South_Dakota':'SD','Tennessee':'TN',
    'Texas':'TX','Utah':'UT','Vermont':'VT','Virginia':'VA','Washington':'WA',
    'West_Virginia':'WV','Wisconsin':'WI','Wyoming':'WY','District_of_Columbia':'DC'
}
df_nbm = client.query("SELECT state, COUNT(DISTINCT frn) AS state_num_providers FROM `broadband-data.fcc_nbm.nbm_hive` GROUP BY state").to_dataframe()
df_nbm['state'] = df_nbm['state'].map(state_name_to_abbr)
df_nbm = df_nbm.dropna(subset=['state'])
df_state_pop = client.query("SELECT stateabbr AS state, SUM(pop2020) AS state_population FROM `broadband-data.fcc_block_level_pop.us2020` GROUP BY stateabbr").to_dataframe()

df = df_projects.merge(df_locations, on='project_id', how='left')
df = df.merge(df_nbm, on='state', how='left')
df = df.merge(df_state_pop, on='state', how='left')
for col in ['funded_locations','technology','state_num_providers','state_population']:
    df[col] = df[col].fillna(0)
df['total_fiber_miles'] = df['estimated_miles_aerial_fiber'].fillna(0) + df['estimated_miles_buried_fiber'].fillna(0)
df['miles_per_location'] = (df['total_fiber_miles'] / df['funded_locations'].replace(0, np.nan)).fillna(0)
df['jobs_per_location'] = (df['estimated_jobs'] / df['funded_locations'].replace(0, np.nan)).fillna(0)
df['funding_per_location'] = df['bead_support'] / df['funded_locations'].replace(0, np.nan)
df = df.dropna(subset=['funding_per_location'])
low = df['funding_per_location'].quantile(0.025)
high = df['funding_per_location'].quantile(0.975)
df = df[(df['funding_per_location'] >= low) & (df['funding_per_location'] <= high)]
df['log_funding'] = np.log1p(df['funding_per_location'])

feature_cols = ['miles_per_location','technology','jobs_per_location',
                'state_population','total_fiber_miles','state_num_providers']
X = df[feature_cols].fillna(0).values
y = df['log_funding'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

def evaluate(name, model, Xtr, Xte, ytr, yte, X_full, y_full):
    model.fit(Xtr, ytr)
    yp = model.predict(Xte)
    r2 = r2_score(yte, yp)
    cv = cross_val_score(model, X_full, y_full, cv=kf, scoring='r2')
    rmse = np.sqrt(mean_squared_error(np.expm1(yte), np.expm1(yp)))
    mae  = mean_absolute_error(np.expm1(yte), np.expm1(yp))
    print(f'{name}: R²={r2:.4f}  CV={cv.mean():.4f}±{cv.std():.4f}  RMSE=${rmse:,.0f}  MAE=${mae:,.0f}')
    return dict(name=name, r2=r2, cv_r2=cv.mean(), cv_std=cv.std(), rmse=rmse, mae=mae, model=model)

results = []

# Linear models (need scaling)
scaler = StandardScaler()
Xtr_s = scaler.fit_transform(X_train)
Xte_s = scaler.transform(X_test)
X_s   = scaler.fit_transform(X)

results.append(evaluate('OLS',   LinearRegression(),        Xtr_s, Xte_s, y_train, y_test, X_s, y))
results.append(evaluate('Ridge', Ridge(alpha=1.0),          Xtr_s, Xte_s, y_train, y_test, X_s, y))
results.append(evaluate('Lasso', Lasso(alpha=0.001),        Xtr_s, Xte_s, y_train, y_test, X_s, y))

# RF baseline and best
results.append(evaluate('RF Baseline (200 trees)',
    RandomForestRegressor(n_estimators=200, min_samples_split=5, random_state=42, n_jobs=-1),
    X_train, X_test, y_train, y_test, X, y))
results.append(evaluate('RF Tuned',
    RandomForestRegressor(n_estimators=1500, min_samples_split=5, min_samples_leaf=2,
                          max_features=0.5, random_state=42, n_jobs=-1),
    X_train, X_test, y_train, y_test, X, y))
results.append(evaluate('GBR Tuned',
    GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=4,
                              subsample=0.8, random_state=42),
    X_train, X_test, y_train, y_test, X, y))
results.append(evaluate('XGBoost',
    XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=4,
                 subsample=0.8, colsample_bytree=0.8, random_state=42,
                 verbosity=0, n_jobs=-1),
    X_train, X_test, y_train, y_test, X, y))

family_map = {
    'OLS': 'linear', 'Ridge': 'linear', 'Lasso': 'linear',
    'RF Baseline (200 trees)': 'rf', 'RF Tuned': 'rf',
    'XGBoost': 'xgb', 'GBR Tuned': 'gbr',
}
results.sort(key=lambda r: r['r2'], reverse=True)

names  = [r['name'] for r in results]
r2     = np.array([r['r2']     for r in results])
cv_r2  = np.array([r['cv_r2']  for r in results])
cv_std = np.array([r['cv_std'] for r in results])
rmse   = np.array([r['rmse']   for r in results])
family = [family_map.get(r['name'], 'rf') for r in results]

palette = {'linear': '#7D3C98', 'rf': '#2471A3', 'gbr': '#C0392B', 'xgb': '#E6A817'}
colors  = [palette[f] for f in family]

y_pos = np.arange(len(names))

# ── Figure 1: R² all models ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.8, 4.4))
bars = ax.barh(y_pos, r2, color=colors, alpha=0.85, height=0.6)
ax.errorbar(cv_r2, y_pos, xerr=cv_std, fmt='none', color='#333',
            capsize=3, linewidth=0.9)
linear_r2 = np.mean([r['r2'] for r in results if family_map.get(r['name']) == 'linear'])
ax.axvline(linear_r2, color='#7D3C98', lw=0.8, linestyle=':', alpha=0.7,
           label=f'Linear ceiling $\\approx {linear_r2:.2f}$')
ax.set_yticks(y_pos)
ax.set_yticklabels(names, fontsize=8)
ax.set_xlabel('$R^2$ (test set) with 5-fold CV error bars')
ax.set_title('All Models: Linear Baselines vs.~Tree Ensembles', weight='bold')
ax.set_xlim(-0.05, 0.85)
for bar, val in zip(bars, r2):
    ax.text(val + 0.008, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=7)
legend_patches = [
    mpatches.Patch(color='#7D3C98', label='Linear (OLS / Ridge / Lasso)'),
    mpatches.Patch(color='#2471A3', label='Random Forest'),
    mpatches.Patch(color='#C0392B', label='Gradient Boosting (GBR)'),
    mpatches.Patch(color='#E6A817', label='XGBoost'),
    plt.Line2D([0],[0], color='#7D3C98', lw=0.8, ls=':', label='Linear ceiling'),
]
ax.legend(handles=legend_patches, fontsize=7, loc='lower right')
plt.tight_layout()
plt.savefig('fig_all_models_r2.pdf')
plt.savefig('fig_all_models_r2.png')
plt.close()
print('Saved fig_all_models_r2')

# ── Figure 2: RMSE all models (log scale to handle linear outliers) ─────────────
fig, ax = plt.subplots(figsize=(6.8, 4.4))
bars = ax.barh(y_pos, rmse, color=colors, alpha=0.85, height=0.6)
ax.set_xscale('log')
ax.set_yticks(y_pos)
ax.set_yticklabels(names, fontsize=8)
ax.set_xlabel('RMSE (\\$, log scale)')
ax.set_title('All Models: RMSE on Test Set', weight='bold')
for bar, val in zip(bars, rmse):
    ax.text(val * 1.03, bar.get_y() + bar.get_height()/2,
            f'\\${val:,}', va='center', fontsize=7)
legend_patches = [
    mpatches.Patch(color='#7D3C98', label='Linear'),
    mpatches.Patch(color='#2471A3', label='Random Forest'),
    mpatches.Patch(color='#C0392B', label='GBR'),
    mpatches.Patch(color='#E6A817', label='XGBoost'),
]
ax.legend(handles=legend_patches, fontsize=7, loc='lower right')
plt.tight_layout()
plt.savefig('fig_all_models_rmse.pdf')
plt.savefig('fig_all_models_rmse.png')
plt.close()
print('Saved fig_all_models_rmse')
