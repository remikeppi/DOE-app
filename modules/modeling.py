"""
Module : Modélisation statistique pour les plans d'expérience (DoE)
Régression linéaire, surface de réponse (RSM), et analyse ANOVA

Développeurs : Mathieu Brucker, Mateis Denot, Rémi Keppi, Faustin Schmitt
Encadrant    : H. Chibane — INSA Strasbourg
"""

import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error

import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Fonctions d'entraînement
# ─────────────────────────────────────────────────────────────────────────────

def fit_linear(X: np.ndarray, y: np.ndarray):
    """Régression linéaire classique (1er ordre)."""
    model = LinearRegression()
    model.fit(X, y)
    return model


def fit_rsm(X: np.ndarray, y: np.ndarray, degree: int = 2):
    """
    Surface de réponse — régression polynomiale du 2e ordre.
    CORRECTION : include_bias=False évite la redondance avec l'intercept de
    LinearRegression et garantit une équation et un R² ajusté corrects.
    """
    model = Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("lr", LinearRegression())
    ])
    model.fit(X, y)
    return model


def compute_metrics(model, X: np.ndarray, y: np.ndarray, model_type: str = "linear") -> dict:
    """Calcule R², RMSE, et R² ajusté."""
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    n = len(y)

    if model_type == "linear":
        p = X.shape[1]
    else:
        # include_bias=False → n_output_features_ = nb termes sans constante
        p = model.named_steps["poly"].n_output_features_

    r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else r2
    return {"R²": round(r2, 4), "R² ajusté": round(r2_adj, 4), "RMSE": round(rmse, 4)}


def get_linear_equation(model: LinearRegression, feature_names: list, target_name: str) -> str:
    """Retourne l'équation du modèle linéaire sous forme lisible."""
    coefs = model.coef_
    intercept = model.intercept_
    terms = [f"{intercept:.4f}"]
    for name, coef in zip(feature_names, coefs):
        sign = "+" if coef >= 0 else "-"
        terms.append(f"{sign} {abs(coef):.4f}·{name}")
    return f"{target_name} = " + " ".join(terms)


def get_rsm_equation(model: Pipeline, feature_names: list, target_name: str) -> str:
    """
    Retourne l'équation RSM (2e ordre) sous forme lisible.
    CORRECTION : avec include_bias=False, get_feature_names_out() ne contient
    pas le terme constant '1', donc on itère directement sur tous les termes.
    """
    poly = model.named_steps["poly"]
    lr = model.named_steps["lr"]
    names = poly.get_feature_names_out(feature_names)
    coefs = lr.coef_
    intercept = lr.intercept_

    terms = [f"{intercept:.4f}"]
    for name, coef in zip(names, coefs):
        if abs(coef) > 1e-10:
            sign = "+" if coef >= 0 else "-"
            terms.append(f"{sign} {abs(coef):.4f}·{name}")
    return f"{target_name} = " + " ".join(terms)


# ─────────────────────────────────────────────────────────────────────────────
# ANOVA via statsmodels
# ─────────────────────────────────────────────────────────────────────────────

def _sanitize_colname(name: str) -> str:
    """
    CORRECTION : conversion robuste d'un nom de colonne en identifiant valide
    pour statsmodels. Gère les tirets, tirets longs (—), parenthèses, %, °, etc.
    """
    clean = re.sub(r"[^a-zA-Z0-9]", "_", name)
    clean = re.sub(r"_+", "_", clean).strip("_")
    if not clean or clean[0].isdigit():
        clean = "var_" + clean
    return clean


def run_anova(df: pd.DataFrame, inputs: list, output: str, degree: int = 2, alpha: float = 0.05):
    """
    Effectue une ANOVA basée sur la régression OLS.
    Retourne (tableau ANOVA propre, résumé OLS) ou (None, message d'erreur).
    """
    col_map = {}
    used: dict = {}
    for col in df.columns:
        base = _sanitize_colname(col)
        if base in used:
            used[base] += 1
            base = f"{base}_{used[base]}"
        else:
            used[base] = 0
        col_map[col] = base

    df_clean = df.rename(columns=col_map)
    clean_inputs = [col_map[i] for i in inputs]
    clean_output = col_map[output]

    terms_clean = list(clean_inputs)
    if degree == 2:
        for x in clean_inputs:
            terms_clean.append(f"I({x}**2)")
        for i in range(len(clean_inputs)):
            for j in range(i + 1, len(clean_inputs)):
                terms_clean.append(f"{clean_inputs[i]}:{clean_inputs[j]}")

    formula = f"{clean_output} ~ " + " + ".join(terms_clean)

    try:
        model_sm = smf.ols(formula, data=df_clean).fit()
        anova_table = sm.stats.anova_lm(model_sm, typ=2)
        anova_table.columns = ["Somme des carrés", "ddl", "F", "p-value"]
        anova_table = anova_table.round(4)
        anova_table["Significatif"] = anova_table["p-value"].apply(
            lambda p: "✅ Oui" if p < alpha else "❌ Non"
        )
        return anova_table, model_sm.summary().as_text()
    except Exception as e:
        return None, str(e)


# ─────────────────────────────────────────────────────────────────────────────
# Importance des variables
# ─────────────────────────────────────────────────────────────────────────────

def compute_importance(model, X: np.ndarray, y: np.ndarray, feature_names: list) -> pd.DataFrame:
    """Calcule l'importance des variables par permutation simple."""
    np.random.seed(0)
    base_r2 = r2_score(y, model.predict(X))
    importances = []
    for i, name in enumerate(feature_names):
        X_perm = X.copy()
        X_perm[:, i] = np.random.permutation(X_perm[:, i])
        perm_r2 = r2_score(y, model.predict(X_perm))
        importances.append({"Variable": name, "Importance": max(0, base_r2 - perm_r2)})
    df_imp = pd.DataFrame(importances).sort_values("Importance", ascending=False)
    total = df_imp["Importance"].sum()
    df_imp["Importance (%)"] = (df_imp["Importance"] / total * 100).round(1) if total > 0 else 0.0
    return df_imp.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Visualisations
# ─────────────────────────────────────────────────────────────────────────────

def plot_predictions(y_true, y_pred, output_name: str):
    """Graphique valeurs réelles vs prédites."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_true, y=y_pred,
        mode="markers",
        marker=dict(color="#0d9488", size=10, opacity=0.8),
        name="Observations"
    ))
    lim = [min(min(y_true), min(y_pred)), max(max(y_true), max(y_pred))]
    fig.add_trace(go.Scatter(
        x=lim, y=lim,
        mode="lines",
        line=dict(color="#f59e0b", dash="dash"),
        name="Idéal (y = ŷ)"
    ))
    fig.update_layout(
        title=f"Valeurs réelles vs prédites — {output_name}",
        xaxis_title="Valeurs réelles",
        yaxis_title="Valeurs prédites",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
        height=400
    )
    return fig


def plot_importance(df_imp: pd.DataFrame, output_name: str):
    """Graphique d'importance des variables."""
    fig = px.bar(
        df_imp,
        x="Importance (%)", y="Variable",
        orientation="h",
        color="Importance (%)",
        color_continuous_scale="teal",
        title=f"Importance des facteurs — {output_name}"
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
        showlegend=False,
        height=350
    )
    return fig


def plot_response_surface(model, X: np.ndarray, input_names: list, output_name: str):
    """
    Surface de réponse 3D pour deux variables (les autres fixées à leur moyenne).
    """
    if X.shape[1] < 2:
        return None

    n_grid = 40
    idx1, idx2 = 0, 1
    x1_range = np.linspace(X[:, idx1].min(), X[:, idx1].max(), n_grid)
    x2_range = np.linspace(X[:, idx2].min(), X[:, idx2].max(), n_grid)
    X1, X2 = np.meshgrid(x1_range, x2_range)

    means = X.mean(axis=0)
    X_grid = np.tile(means, (n_grid * n_grid, 1))
    X_grid[:, idx1] = X1.ravel()
    X_grid[:, idx2] = X2.ravel()

    Z = model.predict(X_grid).reshape(n_grid, n_grid)

    fig = go.Figure(data=[go.Surface(
        x=x1_range, y=x2_range, z=Z,
        colorscale="Teal",
        opacity=0.85
    )])
    fig.add_trace(go.Scatter3d(
        x=X[:, idx1], y=X[:, idx2],
        z=model.predict(X),
        mode="markers",
        marker=dict(size=5, color="#f59e0b"),
        name="Points expérimentaux"
    ))
    fig.update_layout(
        title=f"Surface de réponse — {output_name}",
        scene=dict(
            xaxis_title=input_names[idx1],
            yaxis_title=input_names[idx2],
            zaxis_title=output_name,
            bgcolor="rgba(0,0,0,0)"
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
        height=520
    )
    return fig


def plot_main_effects(model, X: np.ndarray, input_names: list, output_name: str):
    """Graphique des effets principaux de chaque facteur."""
    means = X.mean(axis=0)
    n_grid = 50
    figs_data = []

    for i, name in enumerate(input_names):
        x_range = np.linspace(X[:, i].min(), X[:, i].max(), n_grid)
        X_effect = np.tile(means, (n_grid, 1))
        X_effect[:, i] = x_range
        y_effect = model.predict(X_effect)
        figs_data.append((name, x_range, y_effect))

    n = len(figs_data)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=input_names)

    for idx, (name, x_r, y_r) in enumerate(figs_data):
        r = idx // cols + 1
        c = idx % cols + 1
        fig.add_trace(
            go.Scatter(x=x_r, y=y_r, mode="lines",
                       line=dict(color="#0d9488", width=2.5), name=name),
            row=r, col=c
        )

    fig.update_layout(
        title=f"Effets principaux — {output_name}",
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
        height=300 * rows
    )
    return fig


def plot_anova(anova_df: pd.DataFrame, output_name: str, alpha: float = 0.05):
    """Graphique p-values de l'ANOVA."""
    df_plot = anova_df.copy().dropna()
    df_plot = df_plot[df_plot.index != "Residual"]
    df_plot["-log10(p)"] = -np.log10(df_plot["p-value"].clip(1e-15))

    fig = px.bar(
        df_plot.reset_index(),
        x="index", y="-log10(p)",
        color="-log10(p)",
        color_continuous_scale="RdYlGn",
        title=f"Significativité des facteurs (ANOVA) — {output_name}",
        labels={"index": "Facteur", "-log10(p)": "-log₁₀(p-value)"}
    )
    fig.add_hline(y=-np.log10(alpha), line_dash="dash",
                  line_color="#f59e0b", annotation_text=f"Seuil α={alpha}")
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
        showlegend=False,
        height=380
    )
    return fig
