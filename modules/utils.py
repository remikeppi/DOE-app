"""
Module utilitaires — fonctions partagées entre les modules
"""

import streamlit as st


def badge(text: str, color: str = "#0d9488") -> str:
    """Retourne du HTML pour un badge coloré."""
    return f'<span style="background:{color};color:white;padding:2px 10px;border-radius:12px;font-size:0.8rem;font-weight:600">{text}</span>'


def card_metric(label: str, value, delta=None):
    """Affiche une métrique stylisée."""
    st.metric(label=label, value=value, delta=delta)


def interpretation_r2(r2: float) -> str:
    if r2 >= 0.95:
        return "🟢 Excellent — le modèle explique très bien la variance"
    elif r2 >= 0.85:
        return "🟡 Bon — le modèle est satisfaisant"
    elif r2 >= 0.70:
        return "🟠 Passable — le modèle explique une partie de la variance"
    else:
        return "🔴 Faible — le modèle ne capture pas bien les données"


def interpretation_rmse(rmse: float, y_range: float) -> str:
    ratio = rmse / (y_range + 1e-12)
    if ratio < 0.05:
        return "🟢 Erreur très faible par rapport à la plage de variation"
    elif ratio < 0.15:
        return "🟡 Erreur acceptable"
    else:
        return "🔴 Erreur élevée — vérifiez la qualité des données"


def section_header(icon: str, title: str, subtitle: str = ""):
    st.markdown(f"""
    <div style="border-left:4px solid #0d9488;padding-left:16px;margin:24px 0 12px 0">
        <h2 style="margin:0;color:#e2e8f0">{icon} {title}</h2>
        {"<p style='margin:4px 0 0 0;color:#94a3b8;font-size:0.9rem'>" + subtitle + "</p>" if subtitle else ""}
    </div>
    """, unsafe_allow_html=True)
