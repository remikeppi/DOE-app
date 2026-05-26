"""
Module : Importation et exploration des données expérimentales (DoE)

Développeurs : Mathieu Brucker, Mateis Denot, Rémi Keppi, Faustin Schmitt
Encadrant    : H. Chibane — INSA Strasbourg
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ─────────────────────────────────────────────────────────────────────────────
# Chargement brut du fichier
# ─────────────────────────────────────────────────────────────────────────────

def load_raw(uploaded_file) -> pd.DataFrame:
    """Charge un fichier CSV ou Excel et retourne un DataFrame brut propre."""
    name = uploaded_file.name.lower()

    if name.endswith(".csv"):
        df = None
        for sep in [";", ",", "\t"]:
            try:
                uploaded_file.seek(0)
                tmp = pd.read_csv(uploaded_file, sep=sep, header=0)
                if tmp.shape[1] > 1:
                    df = tmp
                    break
            except Exception:
                pass
        if df is None:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, header=0)

    elif name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file, header=0)
    else:
        raise ValueError("Format non supporté. Utilisez CSV ou Excel (.xlsx / .xls).")

    # Nettoyage des noms de colonnes
    df.columns = [str(c).strip() for c in df.columns]

    # Dédoublonnage des colonnes
    seen: dict = {}
    new_cols = []
    for col in df.columns:
        if col in seen:
            seen[col] += 1
            new_cols.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            new_cols.append(col)
    df.columns = new_cols

    # Conversion numérique : reconstruction via dictionnaire (fix pandas 2.x CoW)
    converted_cols = {}
    for col in df.columns:
        raw_series = df[col].reset_index(drop=True)
        try:
            cleaned = (
                raw_series.astype(str)
                .str.strip()
                .str.replace(",", ".", regex=False)
            )
            numeric = pd.to_numeric(cleaned, errors="raise")
            converted_cols[col] = numeric
        except Exception:
            converted_cols[col] = raw_series

    df = pd.DataFrame(converted_cols)
    df = df.dropna(how="all").reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Encodage des colonnes textuelles
# ─────────────────────────────────────────────────────────────────────────────

def encode_text_columns(df: pd.DataFrame, cols: list) -> tuple:
    """Encode les colonnes textuelles en entiers. Retourne (df_encodé, encodings)."""
    encodings = {}
    df_enc = df.copy()

    for col in cols:
        series = df_enc[col]
        is_text = (
            series.dtype == object
            or str(series.dtype) in ("string", "str")
            or series.apply(lambda x: isinstance(x, str)).any()
        )
        if is_text:
            unique_vals = sorted(series.dropna().unique(), key=str)
            mapping = {val: idx for idx, val in enumerate(unique_vals)}
            df_enc[col] = series.map(mapping)
            encodings[col] = mapping

    return df_enc, encodings


def get_numeric_columns(df: pd.DataFrame) -> list:
    return df.select_dtypes(include=[np.number]).columns.tolist()


# ─────────────────────────────────────────────────────────────────────────────
# Données de démonstration
# ─────────────────────────────────────────────────────────────────────────────

def generate_demo_data() -> pd.DataFrame:
    np.random.seed(42)
    n = 16
    T = np.random.uniform(60, 120, n)
    P = np.random.uniform(1, 5, n)
    fiber = np.random.choice(["Carbon", "Glass", "Kevlar"], n)
    Y1 = 50 + 0.5 * T - 2 * P + np.random.normal(0, 2, n)
    Y2 = 100 - 0.3 * T + 3 * P + np.random.normal(0, 3, n)

    return pd.DataFrame({
        "Temperature (°C)": T.round(1),
        "Pression (bar)":   P.round(2),
        "Fiber Type":       fiber,
        "Rendement (%)":    Y1.round(2),
        "Resistance (MPa)": Y2.round(2),
    })


# ─────────────────────────────────────────────────────────────────────────────
# Affichage — onglet Données
# ─────────────────────────────────────────────────────────────────────────────

def afficher_import(df: pd.DataFrame, inputs: list, outputs: list,
                    encodings: dict = None):
    """Affiche les statistiques et aperçus des données validées."""

    # ── Métriques ─────────────────────────────────────────────────────────────
    st.markdown("### 📊 Aperçu des données")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Expériences", df.shape[0])
    col2.metric("Variables totales", df.shape[1])
    col3.metric("Facteurs (X)", len(inputs))
    col4.metric("Réponses (Y)", len(outputs))

    # ── Encodage texte ────────────────────────────────────────────────────────
    if encodings:
        st.markdown("### 🔤 Variables textuelles encodées")
        st.warning(
            "Les colonnes suivantes contenaient du **texte** et ont été "
            "encodées automatiquement en entiers pour la modélisation."
        )
        for col, mapping in encodings.items():
            with st.expander(f"🔢 Encodage de `{col}`", expanded=True):
                rows = [{"Valeur originale": k, "Code numérique": v}
                        for k, v in mapping.items()]
                st.dataframe(pd.DataFrame(rows), hide_index=True,
                             use_container_width=True)

    # ── Tableau brut ──────────────────────────────────────────────────────────
    with st.expander("🔍 Tableau de données", expanded=True):
        st.dataframe(df.style.highlight_null(color="#ff6b6b"),
                     use_container_width=True)

    # ── Statistiques descriptives — par blocs de 4 variables ─────────────────
    with st.expander("📈 Statistiques descriptives", expanded=True):
        cols_ok = [c for c in inputs + outputs if c in df.columns]
        if cols_ok:
            BLOCK = 4   # nombre de variables par tableau

            if len(cols_ok) <= BLOCK:
                # Assez peu de colonnes → tableau transposé simple
                desc = df[cols_ok].describe().round(4)
                desc_T = desc.T.reset_index().rename(columns={"index": "Variable"})
                st.dataframe(
                    desc_T.style.background_gradient(
                        subset=["mean", "std"], cmap="Blues", axis=0
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                # Beaucoup de colonnes → un tableau par bloc de BLOCK variables
                st.caption(
                    f"**{len(cols_ok)} variables** — affichage en blocs de {BLOCK} "
                    f"pour une meilleure lisibilité."
                )
                n_blocks = (len(cols_ok) + BLOCK - 1) // BLOCK

                for b in range(n_blocks):
                    chunk = cols_ok[b * BLOCK : (b + 1) * BLOCK]
                    st.markdown(
                        f"<span style='color:#0d9488;font-weight:600;font-size:0.9rem'>"
                        f"Variables {b*BLOCK+1} à {b*BLOCK+len(chunk)} / {len(cols_ok)}"
                        f"</span>",
                        unsafe_allow_html=True,
                    )
                    desc = df[chunk].describe().round(4)
                    # Noms de colonnes raccourcis pour l'affichage
                    short_names = {c: (c[:22] + "…" if len(c) > 22 else c) for c in chunk}
                    desc_display = desc.rename(columns=short_names)
                    # Transposé : lignes = variables, colonnes = stats
                    desc_T = desc_display.T.reset_index().rename(
                        columns={"index": "Variable"}
                    )
                    st.dataframe(
                        desc_T.style.background_gradient(
                            subset=["mean", "std"], cmap="Blues", axis=0
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )
                    if b < n_blocks - 1:
                        st.markdown("")  # espace entre les blocs

    # ── Distributions facteurs ────────────────────────────────────────────────
    if inputs:
        st.markdown("### 📉 Distribution des facteurs (entrées)")
        n = len(inputs)
        cpp = min(n, 4)
        rows_n = (n + cpp - 1) // cpp
        fig = make_subplots(rows=rows_n, cols=cpp, subplot_titles=inputs)
        for i, col in enumerate(inputs):
            r, c = i // cpp + 1, i % cpp + 1
            fig.add_trace(
                go.Histogram(x=df[col], name=col,
                             marker_color="#0d9488", opacity=0.8),
                row=r, col=c
            )
        fig.update_layout(showlegend=False,
                          plot_bgcolor="rgba(0,0,0,0)",
                          paper_bgcolor="rgba(0,0,0,0)",
                          font_color="#e2e8f0",
                          height=300 * rows_n)
        st.plotly_chart(fig, use_container_width=True)

    # ── Distributions réponses ────────────────────────────────────────────────
    if outputs:
        st.markdown("### 📈 Distribution des réponses (sorties)")
        n = len(outputs)
        cpp = min(n, 4)
        rows_n = (n + cpp - 1) // cpp
        fig2 = make_subplots(rows=rows_n, cols=cpp, subplot_titles=outputs)
        for i, col in enumerate(outputs):
            r, c = i // cpp + 1, i % cpp + 1
            fig2.add_trace(
                go.Histogram(x=df[col], name=col,
                             marker_color="#f59e0b", opacity=0.8),
                row=r, col=c
            )
        fig2.update_layout(showlegend=False,
                           plot_bgcolor="rgba(0,0,0,0)",
                           paper_bgcolor="rgba(0,0,0,0)",
                           font_color="#e2e8f0",
                           height=300 * rows_n)
        st.plotly_chart(fig2, use_container_width=True)

    # ── Matrice de corrélation ────────────────────────────────────────────────
    all_cols = [c for c in inputs + outputs if c in df.columns]
    if len(all_cols) >= 2:
        st.markdown("### 🔗 Matrice de corrélation")
        # Noms raccourcis pour la heatmap
        short = {c: (c[:18] + "…" if len(c) > 18 else c) for c in all_cols}
        corr = df[all_cols].rename(columns=short).corr().round(3)
        fig3 = px.imshow(corr, color_continuous_scale="RdBu_r",
                         zmin=-1, zmax=1, text_auto=True, aspect="auto")
        fig3.update_layout(plot_bgcolor="rgba(0,0,0,0)",
                           paper_bgcolor="rgba(0,0,0,0)",
                           font_color="#e2e8f0",
                           height=max(400, 60 * len(all_cols)))
        st.plotly_chart(fig3, use_container_width=True)
