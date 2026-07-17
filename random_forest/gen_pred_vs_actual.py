from google.cloud import bigquery
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
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

# ── Load data (same pipeline as gen_linear_comparison.py) ──────────────────
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

# RF Tuned — matches the table entry (test R² ≈ 0.7625)
rf = RandomForestRegressor(n_estimators=1500, min_samples_split=5, min_samples_leaf=2,
                           max_features=0.5, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f'RF Tuned  Test R² = {r2:.4f}')

y_actual_dollar = np.expm1(y_test)
y_pred_dollar   = np.expm1(y_pred)

# ── Figure: Predicted vs Actual ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(4.5, 4.5))
ax.scatter(y_actual_dollar / 1000, y_pred_dollar / 1000,
           s=6, alpha=0.35, color='#4878A8', edgecolors='none', rasterized=True)

lims = [0, max(y_actual_dollar.max(), y_pred_dollar.max()) / 1000 * 1.02]
ax.plot(lims, lims, '--', color='#C62828', lw=1.0, label='Perfect prediction')

ax.set_xlabel('Actual Funding per Location (\\$k)' if HAS_LATEX else 'Actual Funding per Location ($k)')
ax.set_ylabel('Predicted Funding per Location (\\$k)' if HAS_LATEX else 'Predicted Funding per Location ($k)')
ax.set_title('Predicted vs.\\ Actual (Test Set, RF Tuned)' if HAS_LATEX else 'Predicted vs. Actual (Test Set, RF Tuned)', weight='bold')
ax.legend(fontsize=7, loc='upper left')
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_aspect('equal')

r2_label = f'$R^2 = {r2:.4f}$' if HAS_LATEX else f'R² = {r2:.4f}'
ax.text(0.97, 0.08, r2_label,
        transform=ax.transAxes, ha='right', fontsize=8,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.8))

plt.savefig('fig_pred_vs_actual.pdf')
plt.savefig('fig_pred_vs_actual.png')
plt.close()
print('Saved fig_pred_vs_actual.pdf / .png')
