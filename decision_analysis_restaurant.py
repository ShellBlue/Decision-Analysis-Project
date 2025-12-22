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


# ------------------------------------------------------------------
# Data
# ------------------------------------------------------------------
df = pd.read_csv("Restaurant Manual Encoded.csv")

ID_col = df["Restaurant ID"]     # kept only for labeling results
y = df["Reviews"]
X = df.drop(columns=["Name", "Restaurant ID", "Reviews"])


# ------------------------------------------------------------------
# Feature grouping
# ------------------------------------------------------------------
num = ["Review Count"]

ord_ = [
    "Distance (km) from Yuntech",
    "Price Range",
    "Foot Traffic Level",
    "Accessibility Score",
    "Visibility from main road",
    "Storefront Size",
    "Signboard Distinctiveness",
    "Competition Density",
    "Operational Hours",
]

nom = [
    "Category",
    "Building Type",
    "Street Location Type",
    "Surrounding Environment",
]


# ------------------------------------------------------------------
# Regression stage (learn implicit importance)
# ------------------------------------------------------------------
preprocess = ColumnTransformer(
    [
        ("num", RobustScaler(), num),
        ("ord", RobustScaler(), ord_),
        ("nom", OneHotEncoder(drop="first", sparse_output=False), nom),
    ]
)

model = Pipeline(
    [
        ("prep", preprocess),
        ("reg", Ridge()),
    ]
)

model.fit(X, y)

coefs = model.named_steps["reg"].coef_
names = model.named_steps["prep"].get_feature_names_out()


# ------------------------------------------------------------------
# Utility construction
# ------------------------------------------------------------------
def minmax(x):
    return (x - x.min()) / (x.max() - x.min())


num_util = minmax(df[num])
ord_util = minmax(df[ord_])

# Nominal utilities reconstructed from regression coefficients
nom_util = pd.DataFrame(index=df.index)

for c in nom:
    cols = [n for n in names if n.startswith(f"nom__{c}_")]
    vals = np.zeros(len(df))

    for col in cols:
        cat = col.replace(f"nom__{c}_", "")
        vals += (df[c] == cat) * coefs[list(names).index(col)]

    nom_util[c] = vals

nom_util = minmax(nom_util)


decision_df = pd.concat([num_util, ord_util, nom_util], axis=1)
criteria = decision_df.columns.tolist()


# ------------------------------------------------------------------
# Criterion direction
# ------------------------------------------------------------------
criteria_type = [
    "cost" if c in {
        "Distance (km) from Yuntech",
        "Price Range",
        "Competition Density",
    }
    else "benefit"
    for c in criteria
]


# ------------------------------------------------------------------
# ML-derived criterion weights (|β|)
# ------------------------------------------------------------------
crit_w = defaultdict(float)

for name, coef in zip(names, coefs):
    feat = name.split("__", 1)[1]
    for cr in criteria:
        if feat == cr or feat.startswith(cr + "_"):
            crit_w[cr] += abs(coef)
            break

w_ml = np.array([crit_w[c] for c in criteria])
w_ml /= w_ml.sum()


# ------------------------------------------------------------------
# FAHP
# ------------------------------------------------------------------
n = len(criteria)

pcm = np.array(
    [[[w_ml[i] / w_ml[j]] * 3 for j in range(n)] for i in range(n)]
)

fahp = fuzzy_ahp_method(pcm)[0]
w_fahp = np.array([(l + m + u) / 3 for l, m, u in fahp])
w_fahp /= w_fahp.sum()

fuzzy_weights = [[[0.9 * w, w, 1.1 * w] for w in w_fahp]]


# ------------------------------------------------------------------
# FVIKOR
# ------------------------------------------------------------------
decision_matrix = np.array(
    [[[0.9 * v, v, 1.1 * v] for v in row]
     for row in decision_df.values]
)

S, R, Q, ranking = fuzzy_vikor_method(
    decision_matrix,
    fuzzy_weights,
    criteria_type,
    graph=False,
)


# ------------------------------------------------------------------
# Defuzzification and result alignment
# ------------------------------------------------------------------
Q = np.asarray(Q, dtype=float)
if Q.ndim == 2:
    Q = Q.mean(axis=1)

ranking = np.asarray(ranking)
if ranking.ndim == 2:
    ranking = ranking[:, -1]

S = np.asarray(S, dtype=float)
R = np.asarray(R, dtype=float)

if S.ndim == 2:
    S = S.mean(axis=1)
if R.ndim == 2:
    R = R.mean(axis=1)

results = decision_df.copy()
results["Restaurant ID"] = ID_col
results["Q"] = Q
results["Rank"] = ranking


# ------------------------------------------------------------------
# Best alternative
# ------------------------------------------------------------------
best_idx = results["Q"].idxmin()
optimal = results.loc[best_idx]

print("\nOptimal restaurant alternative (utility profile):")
print(optimal)


# ------------------------------------------------------------------
# Plots
# ------------------------------------------------------------------
def vikor_sr_plot(S, R, Q, labels):
    plt.figure(figsize=(6, 6))
    sc = plt.scatter(S, R, s=120, c=Q, edgecolor="black")

    for i, lbl in enumerate(labels):
        plt.text(S[i], R[i], str(lbl), fontsize=9,
                 ha="right", va="bottom")

    plt.xlabel("S (group utility)")
    plt.ylabel("R (individual regret)")
    plt.title("VIKOR S–R compromise space")
    plt.colorbar(sc, label="Q (lower is better)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


ranked = results.sort_values("Q")

vikor_sr_plot(S, R, Q, results["Restaurant ID"])

plt.figure(figsize=(10, 4))
plt.bar(ranked["Restaurant ID"].astype(str), ranked["Q"])
plt.xticks(rotation=90)
plt.ylabel("VIKOR Q")
plt.title("Restaurant ranking (FVIKOR)")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()


top10 = ranked.head(10)

plt.figure(figsize=(6, 4))
plt.barh(top10["Restaurant ID"].astype(str), top10["Q"])
plt.gca().invert_yaxis()
plt.xlabel("VIKOR Q")
plt.title("Top 10 alternatives")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 3))
plt.plot(range(1, len(ranked) + 1), ranked["Q"], marker="o")
plt.xlabel("Rank position")
plt.ylabel("Q value")
plt.yscale("log")
plt.title("Q-gap stability check")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ------------------------------------------------------------------
# Criterion importance
# ------------------------------------------------------------------
importance_df = (
    pd.DataFrame(
        {
            "Criterion": criteria,
            "ML Weight (|β|)": w_ml,
            "FAHP Weight": w_fahp,
        }
    )
    .sort_values("FAHP Weight")
)

plt.figure()
plt.barh(importance_df["Criterion"], importance_df["FAHP Weight"])
plt.xlabel("Weight")
plt.xscale("log")
plt.title("Criterion importance (FAHP)")
plt.tight_layout()
plt.show()

plt.figure()
plt.barh(importance_df["Criterion"], importance_df["ML Weight (|β|)"])
plt.xlabel("Weight")
plt.xscale("log")
plt.title("Criterion importance (ML)")
plt.tight_layout()
plt.show()