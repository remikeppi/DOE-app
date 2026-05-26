"""
Module : Optimisation multi-objectif pour les plans d'expérience (DoE)
Utilise des méthodes basées sur la désirabilité et la grille de recherche.
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ─────────────────────────────────────────────────────────────────────────────
# Désirabilité individuelle
# ─────────────────────────────────────────────────────────────────────────────

def desirability_maximize(y, y_min, y_max, s=1.0):
    """Désirabilité pour maximisation."""
    d = np.clip((y - y_min) / (y_max - y_min + 1e-12), 0, 1)
    return d ** s


def desirability_minimize(y, y_min, y_max, s=1.0):
    """Désirabilité pour minimisation."""
    d = np.clip((y_max - y) / (y_max - y_min + 1e-12), 0, 1)
    return d ** s


def desirability_target(y, y_min, target, y_max, s1=1.0, s2=1.0):
    """Désirabilité pour ciblage (valeur cible)."""
    d = np.where(
        y <= target,
        np.clip((y - y_min) / (target - y_min + 1e-12), 0, 1) ** s1,
        np.clip((y_max - y) / (y_max - target + 1e-12), 0, 1) ** s2
    )
    return d


def global_desirability(desirabilities: list) -> np.ndarray:
    """Désirabilité globale = moyenne géométrique des désirabilités."""
    d_stack = np.column_stack(desirabilities)
    return np.power(np.prod(d_stack, axis=1), 1.0 / len(desirabilities))


# ─────────────────────────────────────────────────────────────────────────────
# Optimisation par grille (exhaustive search)
# ─────────────────────────────────────────────────────────────────────────────

def optimize_grid(models: dict, X: np.ndarray, input_names: list,
                  objectives: dict, n_points: int = 50) -> pd.DataFrame:
    """
    Optimisation par grille exhaustive.

    Parameters
    ----------
    models : dict {output_name: fitted_model}
    X : array d'entrées originales
    input_names : noms des colonnes d'entrée
    objectives : dict {output_name: {"goal": "maximize"|"minimize"|"target",
                                      "target": float (si goal=target),
                                      "weight": float}}
    n_points : résolution de la grille par dimension

    Returns
    -------
    DataFrame trié par désirabilité globale décroissante
    """
    # Bornes
    bounds = [(X[:, i].min(), X[:, i].max()) for i in range(X.shape[1])]

    # Génération de la grille (Latin Hypercube simplifié)
    n_samples = min(n_points ** X.shape[1], 100_000)  # limiter en mémoire
    np.random.seed(42)
    X_grid = np.column_stack([
        np.random.uniform(lo, hi, n_samples) for lo, hi in bounds
    ])

    # Prédictions
    predictions = {}
    for name, model in models.items():
        predictions[name] = model.predict(X_grid)

    # Désirabilités individuelles
    desirabilities = []
    for out_name, obj in objectives.items():
        y_pred = predictions[out_name]
        y_min, y_max = y_pred.min(), y_pred.max()
        goal = obj.get("goal", "maximize")

        if goal == "maximize":
            d = desirability_maximize(y_pred, y_min, y_max)
        elif goal == "minimize":
            d = desirability_minimize(y_pred, y_min, y_max)
        else:  # target
            target = obj.get("target", (y_min + y_max) / 2)
            d = desirability_target(y_pred, y_min, target, y_max)

        w = obj.get("weight", 1.0)
        desirabilities.append(d ** w)

    D = global_desirability(desirabilities)

    # Assemblage du résultat
    df_result = pd.DataFrame(X_grid, columns=input_names)
    for name, pred in predictions.items():
        df_result[f"Pred_{name}"] = pred.round(4)
    df_result["Désirabilité"] = D.round(4)
    df_result = df_result.sort_values("Désirabilité", ascending=False).reset_index(drop=True)

    return df_result


# ─────────────────────────────────────────────────────────────────────────────
# Pareto Front (pour visualisation bi-objectif)
# ─────────────────────────────────────────────────────────────────────────────

def pareto_front(y1: np.ndarray, y2: np.ndarray,
                 goal1: str = "maximize", goal2: str = "maximize"):
    """Identifie les solutions Pareto-optimales pour deux objectifs."""
    sign1 = -1 if goal1 == "maximize" else 1
    sign2 = -1 if goal2 == "maximize" else 1

    y1s = sign1 * y1
    y2s = sign2 * y2

    is_pareto = np.ones(len(y1), dtype=bool)
    for i in range(len(y1)):
        for j in range(len(y1)):
            if i == j:
                continue
            if y1s[j] <= y1s[i] and y2s[j] <= y2s[i] and (y1s[j] < y1s[i] or y2s[j] < y2s[i]):
                is_pareto[i] = False
                break
    return is_pareto


# ─────────────────────────────────────────────────────────────────────────────
# Visualisations
# ─────────────────────────────────────────────────────────────────────────────

def plot_desirability_map(df_result: pd.DataFrame, input_names: list, top_n: int = 500):
    """Carte de désirabilité dans l'espace des facteurs (2 premiers)."""
    df_top = df_result.head(top_n)
    if len(input_names) >= 2:
        fig = px.scatter(
            df_top, x=input_names[0], y=input_names[1],
            color="Désirabilité",
            color_continuous_scale="Teal",
            title="Carte de désirabilité",
            size_max=8
        )
    else:
        fig = px.scatter(
            df_top, x=input_names[0], y="Désirabilité",
            color="Désirabilité",
            color_continuous_scale="Teal",
            title="Désirabilité vs " + input_names[0]
        )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
        height=420
    )
    return fig


def plot_pareto(y1: np.ndarray, y2: np.ndarray, name1: str, name2: str,
                goal1: str, goal2: str):
    """Graphique du front de Pareto pour deux objectifs."""
    is_pareto = pareto_front(y1, y2, goal1, goal2)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y1[~is_pareto], y=y2[~is_pareto],
        mode="markers",
        marker=dict(color="#475569", size=5, opacity=0.4),
        name="Solutions dominées"
    ))
    fig.add_trace(go.Scatter(
        x=y1[is_pareto], y=y2[is_pareto],
        mode="markers+lines",
        marker=dict(color="#f59e0b", size=9, symbol="star"),
        line=dict(color="#f59e0b", width=1.5),
        name="Front de Pareto"
    ))
    fig.update_layout(
        title="Front de Pareto",
        xaxis_title=f"{name1} ({goal1})",
        yaxis_title=f"{name2} ({goal2})",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
        height=420
    )
    return fig


def plot_optimal_radar(best_row: pd.Series, input_names: list):
    """Graphique radar des paramètres optimaux."""
    vals = [best_row[n] for n in input_names]
    fig = go.Figure(data=go.Scatterpolar(
        r=vals + [vals[0]],
        theta=input_names + [input_names[0]],
        fill="toself",
        fillcolor="rgba(13,148,136,0.3)",
        line=dict(color="#0d9488", width=2)
    ))
    fig.update_layout(
        title="Paramètres de la solution optimale",
        polar=dict(
            radialaxis=dict(visible=True),
            bgcolor="rgba(0,0,0,0)"
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
        height=400
    )
    return fig
