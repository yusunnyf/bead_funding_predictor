from google.cloud import bigquery
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import lime, lime.lime_tabular
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

def latex_safe(s):
    s = s.replace('<=', r'$\leq$')
    s = s.replace('>=', r'$\geq$')
    s = s.replace('<',  r'$<$')
    s = s.replace('>',  r'$>$')
    s = s.replace('_',  r'\_')
    return s

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
state_abbr = {
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
df_nbm['state'] = df_nbm['state'].map(state_abbr)
df_nbm = df_nbm.dropna(subset=['state'])
df_pop = client.query("SELECT stateabbr AS state, SUM(pop2020) AS state_population FROM `broadband-data.fcc_block_level_pop.us2020` GROUP BY stateabbr").to_dataframe()

df = df_projects.merge(df_locations, on='project_id', how='left')
df = df.merge(df_nbm, on='state', how='left')
df = df.merge(df_pop, on='state', how='left')
for col in ['funded_locations','technology','state_num_providers','state_population']:
    df[col] = df[col].fillna(0)
df['total_fiber_miles'] = df['estimated_miles_aerial_fiber'].fillna(0) + df['estimated_miles_buried_fiber'].fillna(0)
df['miles_per_location'] = (df['total_fiber_miles'] / df['funded_locations'].replace(0, np.nan)).fillna(0)
df['jobs_per_location']  = (df['estimated_jobs'] / df['funded_locations'].replace(0, np.nan)).fillna(0)
df['funding_per_location'] = df['bead_support'] / df['funded_locations'].replace(0, np.nan)
df = df.dropna(subset=['funding_per_location'])
low, high = df['funding_per_location'].quantile([0.025, 0.975])
df = df[(df['funding_per_location'] >= low) & (df['funding_per_location'] <= high)]
df['log_funding'] = np.log1p(df['funding_per_location'])

feature_cols = ['miles_per_location','technology','jobs_per_location',
                'state_population','total_fiber_miles','state_num_providers']
feat_labels  = ['Miles per Location','Technology Code','Jobs per Location',
                'State Population','Total Fiber Miles','Num. Providers (State)']

X = df[feature_cols].fillna(0)
y = df['log_funding']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestRegressor(n_estimators=200, min_samples_split=5, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

y_actual  = np.expm1(y_test)
high_mask = y_actual > y_actual.quantile(0.90)
candidates = X_test[high_mask]
actual_hi  = y_actual[high_mask]
pred_hi    = np.expm1(rf.predict(candidates))
idx        = np.argmax(actual_hi.values - pred_hi)
example    = candidates.iloc[[idx]]
actual_val = actual_hi.iloc[idx]
pred_val   = pred_hi[idx]
state_info = df.loc[X_test[high_mask].index[idx], 'state']

explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train.values, feature_names=feat_labels, mode='regression', random_state=42)
exp = explainer.explain_instance(
    example.values[0], rf.predict, num_features=6, num_samples=5000)

lime_vals  = exp.as_list()
raw_labels = [v[0] for v in lime_vals]
contribs   = [v[1] for v in lime_vals]
order      = np.argsort(np.abs(contribs))
raw_labels = [raw_labels[i] for i in order]
contribs   = [contribs[i]   for i in order]
colors     = ['#2471A3' if c > 0 else '#C0392B' for c in contribs]
labels     = [latex_safe(l) for l in raw_labels] if HAS_LATEX else raw_labels

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.5, 4.0))

bars = ax.barh(range(len(labels)), contribs, color=colors, alpha=0.85, height=0.55)
ax.axvline(0, color='black', lw=0.9)

# Symmetric xlim with generous padding so labels + values never crowd the bars
xmax = max(abs(min(contribs)), abs(max(contribs)))
pad  = xmax * 0.30
ax.set_xlim(-xmax - pad, xmax + pad)

ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel('LIME contribution to $\\log(1 + \\hat{y})$')
ax.set_title(
    f'LIME Explanation --- High-Cost Project (State: {state_info})\n'
    f'Actual: \\${actual_val:,.0f} vs.\\ Predicted: \\${pred_val:,.0f} per location',
    weight='bold', fontsize=9
)

# Value annotations — placed at bar tip, never crossing zero
gap = xmax * 0.03
for bar, val in zip(bars, contribs):
    if val >= 0:
        xpos, ha = bar.get_width() + gap, 'left'
    else:
        xpos, ha = bar.get_width() - gap, 'right'
    ax.text(xpos, bar.get_y() + bar.get_height() / 2,
            f'{val:+.3f}', va='center', fontsize=7.5, ha=ha)

ax.barh([], [], color='#2471A3', alpha=0.85, label='Increases predicted funding')
ax.barh([], [], color='#C0392B', alpha=0.85, label='Decreases predicted funding')
ax.legend(fontsize=7, loc='lower right')

# Extra left margin so long y-labels are never clipped
fig.subplots_adjust(left=0.32, right=0.95, top=0.88, bottom=0.14)

plt.savefig('fig_lime_example.pdf')
plt.savefig('fig_lime_example.png')
plt.close()
print('Saved fig_lime_example.pdf / .png')
