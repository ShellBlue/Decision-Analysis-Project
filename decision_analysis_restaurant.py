import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from pyDecision.algorithm.fuzzy_ahp import fuzzy_ahp_method
from pyDecision.algorithm.fuzzy_vikor import fuzzy_vikor_method


# ================================================================
# Data
# ================================================================
df = pd.read_csv("Restaurant Manual Encoded.csv")

ID_col = df["Restaurant ID"]
y = df["Reviews"]
X = df.drop(columns=["Name", "Restaurant ID", "Reviews"])


# ================================================================
# Feature groups
# ================================================================
num = ["Review Count"]

ord_ = [
    "Distance (km) from Yuntech", "Price Range", "Foot Traffic Level",
    "Accessibility Score", "Visibility from main road", "Storefront Size",
    "Signboard Distinctiveness", "Competition Density", "Operational Hours",
]

nom = [
    "Category", "Building Type",
    "Street Location Type", "Surrounding Environment",
]


# ================================================================
# Regression (implicit importance learning)
# ================================================================
preprocess = ColumnTransformer(
    [
        ("num", RobustScaler(), num),
        ("ord", RobustScaler(), ord_),
        ("nom", OneHotEncoder(drop="first", sparse_output=False), nom),
    ]
)

model = Pipeline([("prep", preprocess), ("reg", Ridge())])
model.fit(X, y)

coefs = model.named_steps["reg"].coef_
names = model.named_steps["prep"].get_feature_names_out()


# ================================================================
# Utility construction
# ================================================================
def minmax(x):
    return (x - x.min()) / (x.max() - x.min())


num_util = minmax(df[num])
ord_util = minmax(df[ord_])

nom_util = pd.DataFrame(index=df.index)
for c in nom:
    cols = [n for n in names if n.startswith(f"nom__{c}_")]
    vals = sum(
        (df[c] == col.replace(f"nom__{c}_", "")) * coefs[list(names).index(col)]
        for col in cols
    )
    nom_util[c] = vals

nom_util = minmax(nom_util)

decision_df = pd.concat([num_util, ord_util, nom_util], axis=1)
criteria = decision_df.columns.tolist()


# ================================================================
# Criterion type
# ================================================================
criteria_type = [
    "cost" if c in {
        "Distance (km) from Yuntech", "Price Range", "Competition Density"
    } else "benefit"
    for c in criteria
]


# ================================================================
# ML-derived weights
# ================================================================
crit_w = defaultdict(float)
for name, coef in zip(names, coefs):
    feat = name.split("__", 1)[1]
    for cr in criteria:
        if feat == cr or feat.startswith(cr + "_"):
            crit_w[cr] += abs(coef)
            break

w_ml = np.array([crit_w[c] for c in criteria])
w_ml /= w_ml.sum()


# ================================================================
# FAHP
# ================================================================
n = len(criteria)
pcm = np.array([[[w_ml[i] / w_ml[j]] * 3 for j in range(n)] for i in range(n)])

fahp = fuzzy_ahp_method(pcm)[0]
w_fahp = np.array([(l + m + u) / 3 for l, m, u in fahp])
w_fahp /= w_fahp.sum()

fuzzy_weights = [[[0.9 * w, w, 1.1 * w] for w in w_fahp]]


# ================================================================
# FVIKOR
# ================================================================
decision_matrix = np.array(
    [[[0.9 * v, v, 1.1 * v] for v in row] for row in decision_df.values]
)

S, R, Q, ranking = fuzzy_vikor_method(
    decision_matrix, fuzzy_weights, criteria_type, graph=False
)


# ================================================================
# Defuzzification
# ================================================================
Q = np.asarray(Q, float)
S = np.asarray(S, float)
R = np.asarray(R, float)
ranking = np.asarray(ranking)

if Q.ndim == 2: Q = Q.mean(axis=1)
if S.ndim == 2: S = S.mean(axis=1)
if R.ndim == 2: R = R.mean(axis=1)
if ranking.ndim == 2: ranking = ranking[:, -1]

results = decision_df.copy()
results["Restaurant ID"] = ID_col
results["Q"] = Q
results["Rank"] = ranking


# ================================================================
# Sensitivity analysis (v)
# ================================================================
v_values = [0.0, 0.25, 0.5, 0.75, 1.0]
alt_ids = results["Restaurant ID"].astype(str).values

sensitivity_results = pd.DataFrame(index=alt_ids)

for v in v_values:
    _, _, Q_v, _ = fuzzy_vikor_method(
        decision_matrix, fuzzy_weights, criteria_type,
        strategy_coefficient=v, graph=False
    )

    Q_v = np.asarray(Q_v, float)
    if Q_v.ndim == 2:
        Q_v = Q_v.mean(axis=1)

    order = np.argsort(Q_v)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(Q_v) + 1)

    sensitivity_results[f"v={v}"] = ranks


plt.figure(figsize=(14, 8))
for alt in alt_ids:
    plt.plot(v_values, sensitivity_results.loc[alt], marker="o", alpha=0.7, label=f"Alt {alt}")

plt.xlabel("Compromise parameter v")
plt.ylabel("Alternative rank")
plt.title("VIKOR Sensitivity Analysis")
plt.gca().invert_yaxis()
plt.xticks(v_values)
plt.grid(alpha=0.3)
plt.legend(title="Alternatives", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
plt.tight_layout()
plt.show()


# ================================================================
# Results & plots
# ================================================================
best_idx = results["Q"].idxmin()
print("\nOptimal restaurant alternative:")
print(results.loc[best_idx])

ranked = results.sort_values("Q")


def vikor_sr_plot(S, R, Q, labels):
    plt.figure(figsize=(12, 6))
    sc = plt.scatter(S, R, s=120, c=Q, edgecolor="black")
    for i, lbl in enumerate(labels):
        plt.text(S[i], R[i], str(lbl), fontsize=9, ha="right", va="bottom")
    plt.xlabel("S (group utility)")
    plt.ylabel("R (individual regret)")
    plt.title("VIKOR S–R Compromise Space")
    plt.colorbar(sc, label="Q (lower is better)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


vikor_sr_plot(S, R, Q, results["Restaurant ID"])

plt.figure(figsize=(10, 4))
plt.bar(ranked["Restaurant ID"].astype(str), ranked["Q"])
plt.xticks(rotation=90)
plt.ylabel("VIKOR Q")
plt.title("Restaurant ranking (FVIKOR)")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(range(1, len(ranked) + 1), ranked["Q"], marker="o")
plt.yscale("log")
plt.xlabel("Rank position")
plt.ylabel("Q value")
plt.title("Q-gap stability check")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ================================================================
# Criterion importance
# ================================================================
importance_df = pd.DataFrame({
    "Criterion": criteria,
    "ML Weight (|β|)": w_ml,
    "FAHP Weight": w_fahp,
}).sort_values("FAHP Weight")

for col, title in [
    ("FAHP Weight", "Criterion importance (FAHP)"),
    ("ML Weight (|β|)", "Criterion importance (ML)")
]:
    plt.figure(figsize=(5, 4))
    plt.barh(importance_df["Criterion"], importance_df[col])
    plt.xscale("log")
    plt.xlabel("Weight")
    plt.title(title)
    plt.tight_layout()
    plt.show()
