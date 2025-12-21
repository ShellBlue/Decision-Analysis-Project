# ============================================================
# Imports
# ============================================================
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


# ============================================================
# Load data
# ============================================================
df = pd.read_csv("Restaurant Manual Encoded.csv")
ID_col = df["Restaurant ID"]
y = df["Reviews"]
X = df.drop(columns=["Name", "Restaurant ID", "Reviews"])


# ============================================================
# Feature groups
# ============================================================
num = ["Review Count"]
ord_ = [
    "Distance (km) from Yuntech", "Price Range", "Foot Traffic Level",
    "Accessibility Score", "Visibility from main road", "Storefront Size",
    "Signboard Distinctiveness", "Competition Density", "Operational Hours"
]
nom = ["Category", "Building Type", "Street Location Type", "Surrounding Environment"]


# ============================================================
# ML STAGE: Regression model & utilities
# ============================================================
preprocess = ColumnTransformer([
    ("num", RobustScaler(), num),
    ("ord", RobustScaler(), ord_),
    ("nom", OneHotEncoder(drop="first", sparse_output=False), nom)
])
model = Pipeline([("prep", preprocess), ("reg", Ridge())])
model.fit(X, y)

coefs = model.named_steps["reg"].coef_
names = model.named_steps["prep"].get_feature_names_out()

# Min-max utility
minmax = lambda df_: (df_ - df_.min()) / (df_.max() - df_.min())
num_util, ord_util = minmax(df[num]), minmax(df[ord_])

# Nominal utilities via regression
nom_util = pd.DataFrame(index=df.index)
for c in nom:
    cols = [n for n in names if n.startswith(f"nom__{c}_")]
    nom_util[c] = sum((df[c] == col.replace(f"nom__{c}_", "")) * coefs[list(names).index(col)] for col in cols)
nom_util = minmax(nom_util)

# Decision matrix
decision_df = pd.concat([num_util, ord_util, nom_util], axis=1)
criteria = decision_df.columns.tolist()


# ============================================================
# Criterion types & ML weights
# ============================================================
criteria_type = ["cost" if c in ["Distance (km) from Yuntech","Price Range","Competition Density"] else "benefit" for c in criteria]

crit_w = defaultdict(float)
for n, c in zip(names, coefs):
    feat = n.split("__", 1)[1]
    for cr in criteria:
        if feat == cr or feat.startswith(cr + "_"):
            crit_w[cr] += abs(c)
            break
w_ml = np.array([crit_w[c] for c in criteria])
w_ml /= w_ml.sum()


# ============================================================
# FMCDM STAGE: FAHP
# ============================================================
n = len(criteria)
pcm = np.array([[[w_ml[i] / w_ml[j]] * 3 for j in range(n)] for i in range(n)])
fahp = fuzzy_ahp_method(pcm)[0]
w_fahp = np.array([(l + m + u) / 3 for l, m, u in fahp])
w_fahp /= w_fahp.sum()
fuzzy_weights = [[ [0.9*w, w, 1.1*w] for w in w_fahp ]]


# ============================================================
# FMCDM STAGE: FVIKOR
# ============================================================
decision_matrix = np.array([[[0.9*v, v, 1.1*v] for v in row] for row in decision_df.values])
S, R, Q, ranking = fuzzy_vikor_method(decision_matrix, fuzzy_weights, criteria_type, graph=False)

# Defuzzify & align
def flatten_if_needed(arr, axis=-1):
    arr = np.asarray(arr, dtype=float)
    return arr.mean(axis=axis) if arr.ndim == 2 else arr

Q, S, R, ranking = flatten_if_needed(Q), flatten_if_needed(S), flatten_if_needed(R), flatten_if_needed(ranking, 1)

results = decision_df.copy()
results["Restaurant ID"], results["Q"], results["Rank"] = ID_col, Q, ranking

# Optimal restaurant
optimal = results.loc[results["Q"].idxmin()]
print("\nOptimal restaurant alternative (UTILITY PROFILE):")
print(optimal)


# ============================================================
# Plotting
# ============================================================
def vikor_sr_plot(S, R, Q, labels):
    plt.figure(figsize=(6,6))
    sc = plt.scatter(S, R, s=120, c=Q, edgecolor="black")
    for i, lbl in enumerate(labels):
        plt.text(S[i], R[i], str(lbl), fontsize=9, ha="right", va="bottom")
    plt.xlabel("S (Group Utility)"); plt.ylabel("R (Individual Regret)")
    plt.title("VIKOR S–R Compromise Plot"); plt.colorbar(sc, label="Q (lower = better)")
    plt.grid(alpha=0.3); plt.tight_layout(); plt.show()

ranked = results.sort_values("Q")
vikor_sr_plot(S, R, Q, results["Restaurant ID"])

# Rankings & Top 10
plt.figure(figsize=(10,4)); plt.bar(ranked["Restaurant ID"].astype(str), ranked["Q"])
plt.xticks(rotation=90); plt.ylabel("VIKOR Q (lower = better)")
plt.title("Restaurant Rankings (FVIKOR)"); plt.grid(axis="y", alpha=0.3); plt.tight_layout(); plt.show()

top10 = ranked.head(10)
plt.figure(figsize=(6,4)); plt.barh(top10["Restaurant ID"].astype(str), top10["Q"])
plt.gca().invert_yaxis(); plt.xlabel("VIKOR Q"); plt.title("Top 10 Restaurant Locations"); plt.tight_layout(); plt.show()

plt.figure(figsize=(6,3)); plt.plot(range(1,len(ranked)+1), ranked["Q"], marker="o")
plt.xlabel("Rank position"); plt.ylabel("Q value"); plt.yscale("log")
plt.title("Q Gap Analysis (VIKOR Stability)"); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()


# Criterion importance
importance_df = pd.DataFrame({"Criterion": criteria, "ML Weight (|β|)": w_ml, "FAHP Weight": w_fahp}).sort_values("FAHP Weight")
for col, title in [("FAHP Weight","Criterion Importance (FAHP)"), ("ML Weight (|β|)","Criterion Importance (ML)")]:
    plt.figure(); plt.barh(importance_df["Criterion"], importance_df[col])
    plt.xlabel("Weight"); plt.xscale("log"); plt.title(title); plt.tight_layout(); plt.show()