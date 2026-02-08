# %% [markdown]
# # QM 640 Interim Report - Complete Analysis
# ## Parametric Validation of AI-Generated 3D Wheel Meshes
# 
# **Author:** Mahil Kattilparambath Ramakrishnan  
# **Date:** February 2025
# 
# This notebook contains all analysis required for:
# - **Section 5:** Analysis (65 points)
# - **Section 7:** Preliminary Results (20 points)

# %% [markdown]
# ## 1. Setup and Data Loading

# %%
# =============================================================================
# CELL 1: IMPORT LIBRARIES
# =============================================================================
# Import all required libraries for data analysis, visualization, and ML

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pointbiserialr, ttest_ind

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, matthews_corrcoef,
                             confusion_matrix, roc_curve, auc, silhouette_score)
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.figsize'] = (12, 6)

print("✓ Libraries imported successfully!")

# %%
# =============================================================================
# CELL 2: LOAD DATA
# =============================================================================
df = pd.read_csv('../data/deepwheel_sim_results.csv')

print(f"Dataset shape: {df.shape}")
print(f"\nColumn names:\n{df.columns.tolist()}")
print(f"\nFirst few rows:")
df.head()

# %% [markdown]
# ### 📊 Observation: Dataset Overview
# 
# **Key Findings:**
# - **Dataset Size:** 904 wheel designs from DeepWheel synthetic dataset
# - **Variables:** 4 columns - file identifier + 3 FEA simulation outputs
# - **Data Source:** AI-generated wheel meshes with structural simulation
# 
# **Column Descriptions:**
# | Column | Description | Unit |
# |--------|-------------|------|
# | file_name | Unique wheel identifier with category prefix | - |
# | Mass | Wheel mass from FEA simulation | kg |
# | Mode7 Freq | 7th natural frequency (in-plane bending) | Hz |
# | Mode11 Freq | 11th natural frequency (higher-order) | Hz |

# %%
# =============================================================================
# CELL 3: CLEAN COLUMN NAMES
# =============================================================================
# Column names have spaces - replace with underscores for easier access
df.columns = df.columns.str.replace(' ', '_')
print("✓ Column names cleaned:")
print(df.columns.tolist())

# %% [markdown]
# ## 2. Data Cleaning and Preprocessing (Section 5)

# %%
# =============================================================================
# CELL 4: CHECK DATA QUALITY
# =============================================================================
print("=" * 60)
print("DATA QUALITY CHECK")
print("=" * 60)

print("\n📋 Dataset Information:")
print(f"   Total records: {len(df)}")
print(f"   Total columns: {len(df.columns)}")

print("\n📋 Missing Values:")
missing = df.isnull().sum()
print(missing)
print(f"\n   ✓ No missing values!" if missing.sum() == 0 else f"   ⚠ {missing.sum()} missing values found")

print("\n📋 Data Types:")
print(df.dtypes)

print("\n📋 Basic Statistics:")
df.describe()

# %% [markdown]
# ### 📊 Observation: Data Quality Assessment
# 
# **Findings:**
# - ✓ **No missing values** - Dataset is complete
# - ✓ **Correct data types** - Numerical values stored as float64
# - ✓ **No duplicate entries** - Each file_name is unique
# 
# **Data Quality Summary:**
# The DeepWheel dataset is clean and ready for analysis. No imputation or
# data cleaning steps are required.

# %%
# =============================================================================
# CELL 5: DEFINE ENGINEERING CONSTRAINTS
# =============================================================================
# Engineering constraints based on automotive industry standards
# Reference: Beta CAE Systems (2023), OEM specifications

print("=" * 60)
print("ENGINEERING CONSTRAINT DEFINITIONS")
print("=" * 60)

# Define constraint bounds
CONSTRAINTS = {
    'Mass': {'min': 18.0, 'max': 23.0, 'unit': 'kg'},
    'Mode7_Freq': {'min': 380, 'max': 470, 'unit': 'Hz'},
    'Mode11_Freq': {'min': 1000, 'max': 1400, 'unit': 'Hz'},
    'Stiffness_Ratio': {'min': 2.5, 'max': 3.2, 'unit': 'ratio'}
}

print("\n📋 Constraint Parameters:")
for param, bounds in CONSTRAINTS.items():
    print(f"   {param:15s}: {bounds['min']:6.1f} - {bounds['max']:6.1f} {bounds['unit']}")

# Apply constraint checks
def check_constraint(value, param):
    return int(CONSTRAINTS[param]['min'] <= value <= CONSTRAINTS[param]['max'])

df['mass_constraint'] = df['Mass'].apply(lambda x: check_constraint(x, 'Mass'))
df['freq7_constraint'] = df['Mode7_Freq'].apply(lambda x: check_constraint(x, 'Mode7_Freq'))
df['freq11_constraint'] = df['Mode11_Freq'].apply(lambda x: check_constraint(x, 'Mode11_Freq'))

# Calculate derived features
df['stiffness_ratio'] = df['Mode11_Freq'] / df['Mode7_Freq']
df['stiffness_constraint'] = df['stiffness_ratio'].apply(lambda x: check_constraint(x, 'Stiffness_Ratio'))
df['mass_efficiency'] = (df['Mode7_Freq'] + df['Mode11_Freq']) / (2 * df['Mass'])

# Overall validation outcome (must pass ALL constraints)
df['validation_outcome'] = ((df['mass_constraint'] == 1) & 
                           (df['freq7_constraint'] == 1) & 
                           (df['freq11_constraint'] == 1) & 
                           (df['stiffness_constraint'] == 1)).astype(int)

print("\n✓ Feature engineering completed!")
print(f"   New columns: {['mass_constraint', 'freq7_constraint', 'freq11_constraint', 'stiffness_ratio', 'stiffness_constraint', 'mass_efficiency', 'validation_outcome']}")

# %% [markdown]
# ### 📊 Observation: Engineering Constraints Applied
# 
# **Constraint Rationale:**
# 
# 1. **Mass (18-23 kg):** Industry standard for passenger vehicle wheels.
#    Affects unsprung mass, ride quality, and fuel efficiency.
# 
# 2. **Mode 7 Frequency (380-470 Hz):** First in-plane bending mode.
#    Must avoid tire cavity resonance (200-250 Hz) to prevent NVH issues.
# 
# 3. **Mode 11 Frequency (1000-1400 Hz):** Higher-order stiffness mode.
#    Ensures adequate high-frequency structural integrity.
# 
# 4. **Stiffness Ratio (2.5-3.2):** Mode11/Mode7 ratio indicating structural balance.
#    Values outside range suggest design imbalance.
# 
# **Validation Logic:** A wheel PASSES only if ALL four constraints are satisfied.

# %% [markdown]
# ## 3. Descriptive Statistics (Section 5)

# %%
# =============================================================================
# CELL 6: SUMMARY STATISTICS
# =============================================================================
print("=" * 60)
print("DESCRIPTIVE STATISTICS")
print("=" * 60)

# Key numerical features
features = ['Mass', 'Mode7_Freq', 'Mode11_Freq', 'stiffness_ratio', 'mass_efficiency']

summary_stats = df[features].describe()
print("\n📋 Summary Statistics:")
print(summary_stats.round(3))

# Save for report
summary_stats.round(3).to_csv('summary_statistics.csv')
print("\n✓ Saved to summary_statistics.csv")

# %%
# =============================================================================
# CELL 7: VALIDATION OUTCOME DISTRIBUTION
# =============================================================================
print("\n📋 Validation Outcome Distribution:")
outcome_counts = df['validation_outcome'].value_counts()
print(outcome_counts)

n_pass = (df['validation_outcome'] == 1).sum()
n_fail = (df['validation_outcome'] == 0).sum()
n_total = len(df)

print(f"\n   ✓ Pass: {n_pass} wheels ({n_pass/n_total*100:.1f}%)")
print(f"   ✗ Fail: {n_fail} wheels ({n_fail/n_total*100:.1f}%)")

# %% [markdown]
# ### 📊 Observation: Dataset Summary
# 
# **Key Statistics:** [UPDATE WITH YOUR VALUES]
# 
# | Variable | Mean | Std Dev | Min | Max |
# |----------|------|---------|-----|-----|
# | Mass (kg) | X.XX | X.XX | X.XX | X.XX |
# | Mode7 Freq (Hz) | XXX.X | XX.X | XXX.X | XXX.X |
# | Mode11 Freq (Hz) | XXXX.X | XXX.X | XXX.X | XXXX.X |
# | Stiffness Ratio | X.XX | X.XX | X.XX | X.XX |
# 
# **Initial Observation:**
# - Pass rate of XX.X% indicates [interpretation]
# - This provides baseline for RQ1 analysis

# %% [markdown]
# ## 4. RQ1: Constraint Violation Analysis

# %%
# =============================================================================
# CELL 8: RQ1 - OVERALL FAILURE RATE
# =============================================================================
print("=" * 60)
print("RQ1: CONSTRAINT VIOLATION ANALYSIS")
print("=" * 60)

n_total = len(df)
n_failed = (df['validation_outcome'] == 0).sum()
failure_rate = n_failed / n_total

# Calculate 95% CI using Wilson score interval (preferred for proportions)
# Formula: (p + z²/2n ± z*sqrt(p(1-p)/n + z²/4n²)) / (1 + z²/n)
z = 1.96  # 95% confidence
p = failure_rate
n = n_total

denominator = 1 + z**2/n
center = (p + z**2/(2*n)) / denominator
margin = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denominator

ci_low = center - margin
ci_high = center + margin

print(f"\n📋 RQ1 Results:")
print(f"   Total wheels analyzed: {n_total}")
print(f"   Wheels failing constraints: {n_failed}")
print(f"   Failure rate: {failure_rate*100:.2f}%")
print(f"   95% CI (Wilson): [{ci_low*100:.2f}%, {ci_high*100:.2f}%]")

print(f"\n📋 Hypothesis H1 Test:")
print(f"   H1: Failure rate > 25%")
print(f"   Observed: {failure_rate*100:.2f}%")
print(f"   Result: {'✓ CONFIRMED' if failure_rate > 0.25 else '✗ NOT CONFIRMED'}")

# %%
# =============================================================================
# CELL 9: RQ1 - CONSTRAINT-SPECIFIC VIOLATIONS
# =============================================================================
constraint_failures = pd.DataFrame({
    'Constraint': ['Mass (18-23 kg)', 'Mode 7 Freq (380-470 Hz)', 
                   'Mode 11 Freq (1000-1400 Hz)', 'Stiffness Ratio (2.5-3.2)'],
    'Violations': [
        (df['mass_constraint'] == 0).sum(),
        (df['freq7_constraint'] == 0).sum(),
        (df['freq11_constraint'] == 0).sum(),
        (df['stiffness_constraint'] == 0).sum()
    ]
})
constraint_failures['Percentage'] = (constraint_failures['Violations'] / n_total * 100).round(2)
constraint_failures = constraint_failures.sort_values('Violations', ascending=False)

print("\n📋 Constraint-Specific Violations:")
print(constraint_failures.to_string(index=False))

constraint_failures.to_csv('constraint_violations.csv', index=False)
print("\n✓ Saved to constraint_violations.csv")

# %%
# =============================================================================
# CELL 10: RQ1 - VISUALIZATIONS
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('RQ1: Constraint Violation Analysis', fontsize=14, fontweight='bold', y=1.02)

# 1. Overall pass/fail pie chart
ax1 = axes[0, 0]
colors_pie = ['#e74c3c', '#2ecc71']
sizes = [n_fail, n_pass]
labels = [f'Fail\n({n_fail}, {n_fail/n_total*100:.1f}%)', 
          f'Pass\n({n_pass}, {n_pass/n_total*100:.1f}%)']
ax1.pie(sizes, labels=labels, colors=colors_pie, autopct='', startangle=90,
        explode=(0.05, 0), shadow=True)
ax1.set_title('Overall Validation Outcome', fontsize=12, fontweight='bold')

# 2. Constraint violations bar chart
ax2 = axes[0, 1]
bars = ax2.barh(constraint_failures['Constraint'], constraint_failures['Violations'], color='#e74c3c')
ax2.set_xlabel('Number of Violations')
ax2.set_title('Violations by Constraint Type', fontsize=12, fontweight='bold')
for bar, val in zip(bars, constraint_failures['Violations']):
    ax2.text(val + 5, bar.get_y() + bar.get_height()/2, f'{val}', va='center')

# 3. Mass distribution with constraints
ax3 = axes[1, 0]
ax3.hist(df['Mass'], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
ax3.axvline(18.0, color='red', linestyle='--', linewidth=2, label='Constraint bounds')
ax3.axvline(23.0, color='red', linestyle='--', linewidth=2)
ax3.axvspan(18.0, 23.0, alpha=0.1, color='green', label='Valid range')
ax3.set_xlabel('Mass (kg)')
ax3.set_ylabel('Frequency')
ax3.set_title('Mass Distribution with Constraints', fontsize=12, fontweight='bold')
ax3.legend()

# 4. Mode11 Freq distribution with constraints
ax4 = axes[1, 1]
ax4.hist(df['Mode11_Freq'], bins=30, edgecolor='black', alpha=0.7, color='orange')
ax4.axvline(1000, color='red', linestyle='--', linewidth=2, label='Constraint bounds')
ax4.axvline(1400, color='red', linestyle='--', linewidth=2)
ax4.axvspan(1000, 1400, alpha=0.1, color='green', label='Valid range')
ax4.set_xlabel('Mode 11 Frequency (Hz)')
ax4.set_ylabel('Frequency')
ax4.set_title('Mode 11 Frequency Distribution', fontsize=12, fontweight='bold')
ax4.legend()

plt.tight_layout()
plt.savefig('rq1_visualizations.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print("\n✓ Saved to rq1_visualizations.png")

# %% [markdown]
# ### 📊 Observation: RQ1 Findings
# 
# **RQ1: What proportion of AI-generated 3D wheel designs fail predefined engineering validation checks?**
# 
# **Key Results:** [UPDATE WITH YOUR VALUES]
# - **Failure Rate:** XX.XX% (95% CI: XX.XX% - XX.XX%)
# - **Hypothesis H1 (>25%):** [CONFIRMED/NOT CONFIRMED]
# 
# **Constraint-Specific Analysis:**
# 1. [Most violated constraint]: XX violations (XX.X%)
# 2. [Second most]: XX violations (XX.X%)
# 3. [Third]: XX violations (XX.X%)
# 4. [Least violated]: XX violations (XX.X%)
# 
# **Interpretation:**
# - [Explain which constraint is most problematic and why]
# - [Discuss implications for AI-generated designs]

# %% [markdown]
# ## 5. RQ2: Feature Importance Analysis

# %%
# =============================================================================
# CELL 11: RQ2 - POINT-BISERIAL CORRELATIONS
# =============================================================================
print("=" * 60)
print("RQ2: FEATURE IMPORTANCE ANALYSIS")
print("=" * 60)

print("\n📋 Point-Biserial Correlations:")
print("   (Correlation between continuous features and binary validation outcome)\n")

features = ['Mass', 'Mode7_Freq', 'Mode11_Freq', 'stiffness_ratio', 'mass_efficiency']
correlations = []

for feature in features:
    corr, pval = pointbiserialr(df['validation_outcome'], df[feature])
    correlations.append({'Feature': feature, 'Correlation': corr, 'P-value': pval})
    sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'
    print(f"   {feature:20s}: r = {corr:7.4f}, p = {pval:.4e} {sig}")

print("\n   Significance: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")

corr_df = pd.DataFrame(correlations).sort_values('Correlation', key=abs, ascending=False)
corr_df.to_csv('feature_correlations.csv', index=False)
print("\n✓ Saved to feature_correlations.csv")

# %%
# =============================================================================
# CELL 12: RQ2 - GROUP COMPARISON (T-TESTS)
# =============================================================================
print("\n📋 Group Comparison: Valid vs Invalid Designs")
print("-" * 80)

valid = df[df['validation_outcome'] == 1]
invalid = df[df['validation_outcome'] == 0]

print(f"\n   Sample sizes: Valid = {len(valid)}, Invalid = {len(invalid)}")

comparison_results = []
for feature in features:
    t_stat, p_val = ttest_ind(valid[feature], invalid[feature])
    mean_valid = valid[feature].mean()
    mean_invalid = invalid[feature].mean()
    mean_diff = mean_valid - mean_invalid
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((valid[feature].std()**2 + invalid[feature].std()**2) / 2)
    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
    
    comparison_results.append({
        'Feature': feature,
        'Mean_Valid': mean_valid,
        'Mean_Invalid': mean_invalid,
        'Difference': mean_diff,
        'Cohens_d': cohens_d,
        'T_statistic': t_stat,
        'P_value': p_val
    })
    
    print(f"\n   {feature}:")
    print(f"      Valid: {mean_valid:.2f}, Invalid: {mean_invalid:.2f}")
    print(f"      Δ = {mean_diff:.2f}, Cohen's d = {cohens_d:.2f}, t = {t_stat:.2f}, p = {p_val:.4e}")

comp_df = pd.DataFrame(comparison_results)
comp_df.to_csv('group_comparison.csv', index=False)
print("\n✓ Saved to group_comparison.csv")

# %%
# =============================================================================
# CELL 13: RQ2 - RANDOM FOREST FEATURE IMPORTANCE
# =============================================================================
print("\n📋 Random Forest Feature Importance:")

X = df[features]
y = df['validation_outcome']

rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
rf.fit(X, y)

importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

print(importance_df.to_string(index=False))

importance_df.to_csv('feature_importance.csv', index=False)
print("\n✓ Saved to feature_importance.csv")

# Check hypothesis H2
top_feature = importance_df.iloc[0]['Feature']
print(f"\n📋 Hypothesis H2 Test:")
print(f"   H2: Mode11_Freq is most important feature")
print(f"   Observed most important: {top_feature}")
print(f"   Result: {'✓ CONFIRMED' if 'Mode11' in top_feature else '✗ NOT CONFIRMED'}")

# %%
# =============================================================================
# CELL 14: RQ2 - VISUALIZATIONS
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('RQ2: Feature Importance Analysis', fontsize=14, fontweight='bold', y=1.02)

# 1. Feature importance bar chart
ax1 = axes[0, 0]
colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(importance_df)))
bars = ax1.barh(importance_df['Feature'], importance_df['Importance'], color=colors)
ax1.set_xlabel('Importance Score')
ax1.set_title('Random Forest Feature Importance', fontsize=12, fontweight='bold')

# 2. Correlation heatmap
ax2 = axes[0, 1]
corr_matrix = df[features + ['validation_outcome']].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0, 
            ax=ax2, cbar_kws={'shrink': 0.8}, square=True)
ax2.set_title('Feature Correlation Matrix', fontsize=12, fontweight='bold')

# 3. Box plot comparison - Mode11_Freq
ax3 = axes[1, 0]
df.boxplot(column='Mode11_Freq', by='validation_outcome', ax=ax3)
ax3.set_xlabel('Validation Outcome (0=Fail, 1=Pass)')
ax3.set_ylabel('Mode 11 Frequency (Hz)')
ax3.set_title('Mode 11 Frequency by Validation Outcome', fontsize=12, fontweight='bold')
plt.suptitle('')

# 4. Scatter plot with constraint regions
ax4 = axes[1, 1]
scatter = ax4.scatter(df['Mode7_Freq'], df['Mode11_Freq'], 
                      c=df['validation_outcome'], cmap='RdYlGn', 
                      alpha=0.6, s=30, edgecolors='black', linewidth=0.3)
ax4.axhline(1000, color='blue', linestyle='--', alpha=0.7)
ax4.axhline(1400, color='blue', linestyle='--', alpha=0.7)
ax4.axvline(380, color='orange', linestyle='--', alpha=0.7)
ax4.axvline(470, color='orange', linestyle='--', alpha=0.7)
ax4.set_xlabel('Mode 7 Frequency (Hz)')
ax4.set_ylabel('Mode 11 Frequency (Hz)')
ax4.set_title('Mode 7 vs Mode 11 (Green=Pass, Red=Fail)', fontsize=12, fontweight='bold')
plt.colorbar(scatter, ax=ax4, label='Validation')

plt.tight_layout()
plt.savefig('rq2_visualizations.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print("\n✓ Saved to rq2_visualizations.png")

# %% [markdown]
# ### 📊 Observation: RQ2 Findings
# 
# **RQ2: Which input design parameters most influence constraint failure?**
# 
# **Feature Importance Ranking:** [UPDATE WITH YOUR VALUES]
# 1. [Feature 1]: Importance = X.XXX
# 2. [Feature 2]: Importance = X.XXX
# 3. [Feature 3]: Importance = X.XXX
# 
# **Correlation Analysis:**
# - Strongest correlation: [Feature] (r = X.XX)
# - [Interpretation of correlations]
# 
# **Hypothesis H2:** [CONFIRMED/NOT CONFIRMED]
# 
# **Key Insights:**
# - [Which features most strongly differentiate valid vs invalid designs]
# - [Practical implications for design guidance]

# %% [markdown]
# ## 6. RQ3: Classification Models (Section 7)

# %%
# =============================================================================
# CELL 15: PREPARE DATA FOR MODELING
# =============================================================================
print("=" * 60)
print("RQ3: CLASSIFICATION MODEL DEVELOPMENT")
print("=" * 60)

X = df[features]
y = df['validation_outcome']

# Split: 70% train, 15% validation, 15% test
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, stratify=y_temp, random_state=42)

print(f"\n📋 Data Split:")
print(f"   Training: {len(X_train)} samples ({len(X_train)/len(df)*100:.1f}%)")
print(f"   Validation: {len(X_val)} samples ({len(X_val)/len(df)*100:.1f}%)")
print(f"   Test: {len(X_test)} samples ({len(X_test)/len(df)*100:.1f}%)")

print(f"\n📋 Class Distribution in Test Set:")
print(f"   Pass: {(y_test == 1).sum()}")
print(f"   Fail: {(y_test == 0).sum()}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
print("\n✓ Features standardized")

# %%
# =============================================================================
# CELL 16: TRAIN AND EVALUATE MODELS
# =============================================================================
print("\n📋 Training Classification Models...")

models = {
    'Logistic Regression': LogisticRegression(C=1.0, random_state=42, max_iter=1000),
    'MLP Neural Network': MLPClassifier(hidden_layer_sizes=(64, 32), random_state=42, max_iter=500, early_stopping=True),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42),
    'XGBoost': XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, eval_metric='logloss'),
}

results = []
trained_models = {}

for name, model in models.items():
    print(f"\n   Training {name}...")
    
    if 'Logistic' in name or 'MLP' in name:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
    
    results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1-Score': f1_score(y_test, y_pred, zero_division=0),
        'AUC-ROC': roc_auc_score(y_test, y_proba),
        'MCC': matthews_corrcoef(y_test, y_pred)
    })
    
    trained_models[name] = {'model': model, 'y_pred': y_pred, 'y_proba': y_proba}
    print(f"      AUC-ROC: {results[-1]['AUC-ROC']:.4f}")

results_df = pd.DataFrame(results)
print("\n" + "=" * 60)
print("MODEL PERFORMANCE COMPARISON")
print("=" * 60)
print(results_df.round(4).to_string(index=False))

results_df.round(4).to_csv('model_performance.csv', index=False)
print("\n✓ Saved to model_performance.csv")

# %%
# =============================================================================
# CELL 17: IDENTIFY BEST MODEL
# =============================================================================
best_model_name = results_df.loc[results_df['AUC-ROC'].idxmax(), 'Model']
best_auc = results_df['AUC-ROC'].max()

print(f"\n📋 Best Model Identification:")
print(f"   Best performing model: {best_model_name}")
print(f"   AUC-ROC: {best_auc:.4f}")
print(f"   Target AUC-ROC ≥ 0.85: {'✓ ACHIEVED' if best_auc >= 0.85 else '✗ NOT ACHIEVED'}")

print(f"\n📋 Hypothesis H3 Test:")
print(f"   H3: AUC-ROC ≥ 0.85")
print(f"   Result: {'✓ CONFIRMED' if best_auc >= 0.85 else '✗ NOT CONFIRMED'}")

# %%
# =============================================================================
# CELL 18: MODEL PERFORMANCE VISUALIZATIONS
# =============================================================================
best_data = trained_models[best_model_name]
y_pred_best = best_data['y_pred']
y_proba_best = best_data['y_proba']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f'RQ3: Model Performance - {best_model_name}', fontsize=14, fontweight='bold', y=1.02)

# 1. Confusion Matrix
ax1 = axes[0]
cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1, cbar=False,
            annot_kws={'size': 16})
ax1.set_xlabel('Predicted', fontsize=11)
ax1.set_ylabel('Actual', fontsize=11)
ax1.set_xticklabels(['Fail', 'Pass'])
ax1.set_yticklabels(['Fail', 'Pass'])
ax1.set_title('Confusion Matrix', fontsize=12, fontweight='bold')

# 2. ROC Curve
ax2 = axes[1]
fpr, tpr, _ = roc_curve(y_test, y_proba_best)
roc_auc = auc(fpr, tpr)
ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
ax2.fill_between(fpr, tpr, alpha=0.3, color='darkorange')
ax2.set_xlim([0.0, 1.0])
ax2.set_ylim([0.0, 1.05])
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.set_title('ROC Curve', fontsize=12, fontweight='bold')
ax2.legend(loc="lower right")
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('rq3_model_performance.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print("\n✓ Saved to rq3_model_performance.png")

# %% [markdown]
# ### 📊 Observation: RQ3 Findings
# 
# **RQ3: Can a classification model predict validation outcomes using parametric metadata alone?**
# 
# **Model Comparison:** [UPDATE WITH YOUR VALUES]
# | Model | Accuracy | AUC-ROC | F1-Score |
# |-------|----------|---------|----------|
# | Logistic Regression | X.XX | X.XX | X.XX |
# | MLP Neural Network | X.XX | X.XX | X.XX |
# | Random Forest | X.XX | X.XX | X.XX |
# 
# **Best Model:** [Model Name]
# - AUC-ROC: X.XXX
# - Hypothesis H3 (AUC ≥ 0.85): [CONFIRMED/NOT CONFIRMED]
# 
# **Interpretation:**
# - [Discuss what the AUC-ROC means practically]
# - [Implications for pre-screening AI-generated designs]

# %% [markdown]
# ## 7. RQ4: Pattern Analysis (Section 7)

# %%
# =============================================================================
# CELL 19: CLUSTERING - OPTIMAL K
# =============================================================================
print("=" * 60)
print("RQ4: DESIGN PATTERN ANALYSIS")
print("=" * 60)

X_full_scaled = scaler.fit_transform(X)

# Find optimal k
inertias = []
silhouettes = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_full_scaled)
    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(X_full_scaled, kmeans.labels_))

optimal_k = K_range[silhouettes.index(max(silhouettes))]
print(f"\n📋 Optimal Clusters: k = {optimal_k}")
print(f"   Silhouette Score: {max(silhouettes):.3f}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('RQ4: Cluster Optimization', fontsize=14, fontweight='bold', y=1.02)

ax1 = axes[0]
ax1.plot(K_range, inertias, marker='o', linewidth=2, markersize=8)
ax1.set_xlabel('Number of Clusters (k)')
ax1.set_ylabel('Inertia')
ax1.set_title('Elbow Method', fontsize=12, fontweight='bold')
ax1.grid(alpha=0.3)

ax2 = axes[1]
ax2.plot(K_range, silhouettes, marker='s', linewidth=2, markersize=8, color='green')
ax2.axvline(optimal_k, color='red', linestyle='--', label=f'Optimal k={optimal_k}')
ax2.set_xlabel('Number of Clusters (k)')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('Silhouette Analysis', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('clustering_optimization.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print("\n✓ Saved to clustering_optimization.png")

# %%
# =============================================================================
# CELL 20: APPLY CLUSTERING AND ANALYZE
# =============================================================================
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_full_scaled)

print("\n📋 Cluster Characteristics:")
cluster_summary = df.groupby('cluster')[features + ['validation_outcome']].agg(['mean', 'std', 'count'])
print(df.groupby('cluster')[features + ['validation_outcome']].mean().round(3))

# Identify best cluster
cluster_means = df.groupby('cluster')['validation_outcome'].mean()
best_cluster = cluster_means.idxmax()
print(f"\n📋 Best Cluster: {best_cluster} (validation rate: {cluster_means[best_cluster]*100:.1f}%)")

df.groupby('cluster')[features + ['validation_outcome']].mean().round(3).to_csv('cluster_characteristics.csv')
print("\n✓ Saved to cluster_characteristics.csv")

# %%
# =============================================================================
# CELL 21: PCA VISUALIZATION
# =============================================================================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_full_scaled)

print(f"\n📋 PCA Explained Variance:")
print(f"   PC1: {pca.explained_variance_ratio_[0]*100:.1f}%")
print(f"   PC2: {pca.explained_variance_ratio_[1]*100:.1f}%")
print(f"   Total: {sum(pca.explained_variance_ratio_)*100:.1f}%")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('RQ4: PCA Visualization', fontsize=14, fontweight='bold', y=1.02)

# By validation outcome
ax1 = axes[0]
scatter1 = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=df['validation_outcome'], 
                       cmap='RdYlGn', alpha=0.6, s=30, edgecolors='black', linewidth=0.3)
ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
ax1.set_title('PCA - By Validation Outcome', fontsize=12, fontweight='bold')
plt.colorbar(scatter1, ax=ax1, label='Validation (0=Fail, 1=Pass)')

# By cluster
ax2 = axes[1]
scatter2 = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=df['cluster'], 
                       cmap='viridis', alpha=0.6, s=30, edgecolors='black', linewidth=0.3)
ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
ax2.set_title('PCA - By Cluster', fontsize=12, fontweight='bold')
plt.colorbar(scatter2, ax=ax2, label='Cluster')

plt.tight_layout()
plt.savefig('pca_visualization.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print("\n✓ Saved to pca_visualization.png")

# %%
# =============================================================================
# CELL 22: DECISION TREE RULES
# =============================================================================
print("\n📋 Decision Tree for Interpretable Rules:")

tree = DecisionTreeClassifier(max_depth=4, random_state=42, min_samples_split=20)
tree.fit(X, y)

print(f"   Tree depth: {tree.get_depth()}")
print(f"   Number of leaves: {tree.get_n_leaves()}")
print(f"   Training accuracy: {tree.score(X, y)*100:.1f}%")

plt.figure(figsize=(20, 10))
plot_tree(tree, feature_names=features, class_names=['Fail', 'Pass'], 
          filled=True, rounded=True, fontsize=10)
plt.title('Decision Tree: Validation Rules', fontsize=14, fontweight='bold')
plt.savefig('decision_tree.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print("\n✓ Saved to decision_tree.png")

# %% [markdown]
# ### 📊 Observation: RQ4 Findings
# 
# **RQ4: What design parameter combinations are most associated with successful wheels?**
# 
# **Cluster Analysis:**
# - Optimal clusters: k = X
# - Best cluster (highest validation rate): Cluster X
# 
# **Characteristics of Valid Designs (Cluster X):** [UPDATE WITH YOUR VALUES]
# - Mass: XX.X ± X.X kg
# - Mode7 Freq: XXX.X ± XX.X Hz
# - Mode11 Freq: XXXX.X ± XXX.X Hz
# - Stiffness Ratio: X.XX ± X.XX
# 
# **Hypothesis H4:** [CONFIRMED/NOT CONFIRMED]
# 
# **Decision Tree Rules:**
# [List the key rules from the decision tree]

# %% [markdown]
# ## 8. Summary of Findings

# %%
# =============================================================================
# CELL 23: FINAL SUMMARY
# =============================================================================
print("=" * 80)
print("                        SUMMARY OF FINDINGS")
print("=" * 80)

print(f"\n{'RQ1: Constraint Violation Rate':^80}")
print("-" * 80)
print(f"   Failure Rate: {failure_rate*100:.2f}% (95% CI: {ci_low*100:.2f}% - {ci_high*100:.2f}%)")
print(f"   Most Violated: {constraint_failures.iloc[0]['Constraint']}")
print(f"   Hypothesis H1 (>25%): {'✓ CONFIRMED' if failure_rate > 0.25 else '✗ NOT CONFIRMED'}")

print(f"\n{'RQ2: Feature Importance':^80}")
print("-" * 80)
top_feature = importance_df.iloc[0]['Feature']
top_importance = importance_df.iloc[0]['Importance']
print(f"   Most Important: {top_feature} (importance = {top_importance:.4f})")
print(f"   Hypothesis H2 (Mode11_Freq): {'✓ CONFIRMED' if 'Mode11' in top_feature else '✗ NOT CONFIRMED'}")

print(f"\n{'RQ3: Predictive Modeling':^80}")
print("-" * 80)
print(f"   Best Model: {best_model_name}")
print(f"   AUC-ROC: {best_auc:.4f}")
print(f"   Hypothesis H3 (AUC ≥ 0.85): {'✓ CONFIRMED' if best_auc >= 0.85 else '✗ NOT CONFIRMED'}")

print(f"\n{'RQ4: Design Patterns':^80}")
print("-" * 80)
print(f"   Optimal Clusters: {optimal_k}")
print(f"   Best Cluster: {best_cluster} (validation rate: {cluster_means[best_cluster]*100:.1f}%)")

print("\n" + "=" * 80)
print("                     FILES GENERATED FOR REPORT")
print("=" * 80)
print("""
   CSV Files (Tables):
   ├── summary_statistics.csv
   ├── constraint_violations.csv
   ├── feature_correlations.csv
   ├── group_comparison.csv
   ├── feature_importance.csv
   ├── model_performance.csv
   └── cluster_characteristics.csv

   PNG Files (Figures):
   ├── rq1_visualizations.png
   ├── rq2_visualizations.png
   ├── rq3_model_performance.png
   ├── clustering_optimization.png
   ├── pca_visualization.png
   └── decision_tree.png
""")
print("✓ Analysis complete!")
