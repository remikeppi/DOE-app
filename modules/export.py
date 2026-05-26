"""
Module : Export CSV et PDF
Génère des rapports téléchargeables depuis l'application DoE.

Développeurs : Mathieu Brucker, Mateis Denot, Rémi Keppi, Faustin Schmitt
Encadrant    : H. Chibane — INSA Strasbourg
"""

import io, os
import pandas as pd
import numpy as np
from datetime import datetime
from fpdf import FPDF

_HERE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGO_PATH = os.path.join(_HERE, "insa_logo.png")
FONT_REG  = os.path.join(_HERE, "LiberationSans-Regular.ttf")
FONT_BOLD = os.path.join(_HERE, "LiberationSans-Bold.ttf")
FONT_MONO = os.path.join(_HERE, "LiberationMono-Regular.ttf")


# ─────────────────────────────────────────────────────────────────────────────
# Export CSV
# ─────────────────────────────────────────────────────────────────────────────

def export_csv(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False, sep=";", decimal=",", encoding="utf-8-sig")
    return buf.getvalue().encode("utf-8-sig")

def export_predictions_csv(df, inputs, outputs, models) -> bytes:
    result = df[inputs + outputs].copy()
    X = df[inputs].values
    for out, model in models.items():
        result[f"Pred_{out}"] = model.predict(X).round(4)
        result[f"Residu_{out}"] = (df[out] - result[f"Pred_{out}"]).round(4)
    return export_csv(result)

def export_optimization_csv(df_opt: pd.DataFrame) -> bytes:
    return export_csv(df_opt)


# ─────────────────────────────────────────────────────────────────────────────
# Classe PDF
# ─────────────────────────────────────────────────────────────────────────────

class DoEReport(FPDF):
    TEAL  = (13, 148, 136)
    DARK  = (15, 23, 42)
    SLATE = (100, 116, 139)
    WHITE = (255, 255, 255)
    LIGHT = (241, 245, 249)

    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=18)
        self.set_margins(14, 14, 14)
        # Police Unicode
        self.add_font("Sans",     "",  FONT_REG,  uni=True)
        self.add_font("Sans",     "B", FONT_BOLD, uni=True)
        self.add_font("Mono",     "",  FONT_MONO, uni=True)

    def _sans(self, size=9, bold=False):
        self.set_font("Sans", "B" if bold else "", size)

    def _mono(self, size=8):
        self.set_font("Mono", "", size)

    def header(self):
        self.set_fill_color(*self.TEAL)
        self.rect(0, 0, 210, 8, "F")
        if os.path.exists(LOGO_PATH):
            self.image(LOGO_PATH, x=14, y=10, h=11)
        self.set_xy(0, 11)
        self._sans(8, bold=True)
        self.set_text_color(*self.TEAL)
        self.cell(0, 5, "DoE Analyzer — Plans d'experiences | INSA Strasbourg",
                  align="R")
        self.ln(13)
        self.set_draw_color(*self.TEAL)
        self.set_line_width(0.3)
        self.line(14, 25, 196, 25)
        self.ln(5)
        self.set_text_color(0, 0, 0)

    def footer(self):
        self.set_y(-11)
        self._sans(7)
        self.set_text_color(*self.SLATE)
        self.cell(0, 5,
            f"DoE Analyzer  |  INSA Strasbourg  |  "
            f"M. Brucker, M. Denot, R. Keppi, F. Schmitt  |  "
            f"Page {self.page_no()}", align="C")

    def section_title(self, title: str):
        self.set_fill_color(*self.TEAL)
        self.set_text_color(*self.WHITE)
        self._sans(11, bold=True)
        self.cell(0, 8, f"  {title}", ln=True, fill=True)
        self.ln(3)
        self.set_text_color(0, 0, 0)

    def sub_title(self, title: str):
        self._sans(10, bold=True)
        self.set_text_color(*self.TEAL)
        self.cell(0, 6, title, ln=True)
        self.set_text_color(0, 0, 0)
        self.ln(1)

    def kv(self, key: str, value: str, bold_val=False):
        self._sans(9)
        self.set_text_color(*self.SLATE)
        self.cell(58, 5, f"{key} :")
        self.set_text_color(0, 0, 0)
        self._sans(9, bold=bold_val)
        self.cell(0, 5, str(value), ln=True)

    def equation_box(self, eq: str):
        self.set_fill_color(*self.LIGHT)
        self.set_draw_color(*self.TEAL)
        self.set_line_width(0.4)
        self._mono(7.5)
        self.set_text_color(*self.DARK)
        # Wrap
        words = eq.split(" ")
        lines_out, cur = [], ""
        for w in words:
            test = (cur + " " + w).strip()
            if self.get_string_width(test) < 168:
                cur = test
            else:
                if cur:
                    lines_out.append(cur)
                cur = w
        if cur:
            lines_out.append(cur)
        h_box = 6 * len(lines_out) + 4
        self.rect(14, self.get_y(), 182, h_box, "FD")
        self.set_xy(17, self.get_y() + 2)
        for line in lines_out:
            self.cell(0, 6, line, ln=True)
            self.set_x(17)
        self.ln(3)
        self.set_text_color(0, 0, 0)
        self.set_draw_color(0, 0, 0)
        self.set_line_width(0.2)

    def table(self, headers, rows, col_widths=None):
        n = len(headers)
        if col_widths is None:
            col_widths = [182 / n] * n
        # Header row
        self.set_fill_color(*self.DARK)
        self.set_text_color(*self.WHITE)
        self._sans(8, bold=True)
        for h, w in zip(headers, col_widths):
            self.cell(w, 7, str(h)[:20], border=0, align="C", fill=True)
        self.ln()
        # Data rows
        self._sans(8)
        for i, row in enumerate(rows):
            fill = (i % 2 == 0)
            self.set_fill_color(*self.LIGHT)
            self.set_text_color(0, 0, 0)
            for val, w in zip(row, col_widths):
                self.cell(w, 6, str(val)[:22], border=0, align="C", fill=fill)
            self.ln()
        self.ln(3)


# ─────────────────────────────────────────────────────────────────────────────
# Rapport PDF complet
# ─────────────────────────────────────────────────────────────────────────────

def generate_pdf_report(
    df, inputs, outputs, models,
    metrics_lin, metrics_rsm, eq_lin, eq_rsm,
    anova_results=None, df_opt=None, objectives=None, encodings=None,
) -> bytes:

    pdf = DoEReport()
    now = datetime.now().strftime("%d/%m/%Y %H:%M")

    # ── PAGE DE GARDE ─────────────────────────────────────────────────────────
    pdf.add_page()
    if os.path.exists(LOGO_PATH):
        pdf.image(LOGO_PATH, x=45, y=35, w=120)
    pdf.set_y(85)

    pdf.set_fill_color(*DoEReport.TEAL)
    pdf.set_text_color(*DoEReport.WHITE)
    pdf._sans(18, bold=True)
    pdf.cell(0, 14, "Rapport DoE Analyzer", align="C", fill=True, ln=True)
    pdf.set_fill_color(*DoEReport.DARK)
    pdf._sans(11, bold=True)
    pdf.cell(0, 10, "Plans d'experiences -- Resultats d'analyse",
             align="C", fill=True, ln=True)
    pdf.ln(8)

    pdf.set_fill_color(*DoEReport.LIGHT)
    pdf.set_text_color(0, 0, 0)
    y0 = pdf.get_y()
    pdf.rect(28, y0, 154, 56, "F")
    pdf.set_xy(34, y0 + 5)

    for k, v in [
        ("Date", now),
        ("Etablissement", "INSA Strasbourg"),
        ("Encadrant", "H. Chibane"),
        ("Developpeurs", "M. Brucker, M. Denot, R. Keppi, F. Schmitt"),
        ("Experiences", str(df.shape[0])),
        ("Facteurs (X)", str(len(inputs))),
        ("Reponses (Y)", str(len(outputs))),
    ]:
        pdf.set_x(34)
        pdf._sans(9, bold=True)
        pdf.set_text_color(*DoEReport.SLATE)
        pdf.cell(52, 7, f"{k} :")
        pdf._sans(9)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 7, v, ln=True)
    pdf.ln(5)

    pdf._sans(9, bold=True); pdf.set_text_color(*DoEReport.TEAL)
    pdf.cell(0, 5, "Facteurs d'entree (X) :", ln=True)
    pdf._sans(9); pdf.set_text_color(0, 0, 0)
    for inp in inputs:
        pdf.cell(0, 5, f"    + {inp}", ln=True)
    pdf.ln(2)
    pdf._sans(9, bold=True); pdf.set_text_color(*DoEReport.TEAL)
    pdf.cell(0, 5, "Reponses de sortie (Y) :", ln=True)
    pdf._sans(9); pdf.set_text_color(0, 0, 0)
    for out in outputs:
        pdf.cell(0, 5, f"    + {out}", ln=True)

    # ── SECTION 1 : DONNEES ───────────────────────────────────────────────────
    pdf.add_page()
    pdf.section_title("1 -- Resume des donnees")

    if encodings:
        pdf.sub_title("Variables textuelles encodees")
        for col, mapping in encodings.items():
            pdf._sans(8, bold=True); pdf.cell(0, 5, f"  {col} :", ln=True)
            pdf._sans(8)
            pdf.cell(0, 5,
                "    " + "  |  ".join(f"{k} -> {v}" for k, v in mapping.items()),
                ln=True)
        pdf.ln(2)

    pdf.sub_title("Statistiques descriptives")
    cols_ok = [c for c in inputs + outputs if c in df.columns]
    desc = df[cols_ok].describe().round(3)
    stats = ["count", "mean", "std", "min", "max"]

    # Découpage en blocs de 4 variables max pour éviter les débordements
    BLOCK = 4
    n_blocks = (len(cols_ok) + BLOCK - 1) // BLOCK
    for b in range(n_blocks):
        chunk = cols_ok[b * BLOCK : (b + 1) * BLOCK]
        if n_blocks > 1:
            pdf._sans(8, bold=True)
            pdf.set_text_color(*DoEReport.TEAL)
            pdf.cell(0, 5,
                f"  Variables {b*BLOCK+1} a {b*BLOCK+len(chunk)} / {len(cols_ok)}",
                ln=True)
            pdf.set_text_color(0, 0, 0)
        short_chunk = [c[:22] for c in chunk]
        # Largeurs : colonne stat fixe + colonnes data réparties
        avail = 182 - 22  # largeur page moins colonne "Stat."
        col_w_data = max(16, avail // len(chunk))
        cw_chunk = [22] + [col_w_data] * len(chunk)
        rows_desc = []
        for stat in stats:
            if stat in desc.index:
                row = [stat] + [str(round(desc.loc[stat, c], 3))
                                if c in desc.columns else "-" for c in chunk]
                rows_desc.append(row)
        pdf.table(["Stat."] + short_chunk, rows_desc, cw_chunk)
        if b < n_blocks - 1:
            pdf.ln(2)

    # ── SECTION 2 : MODELISATION ──────────────────────────────────────────────
    if models:
        pdf.add_page()
        pdf.section_title("2 -- Resultats de modelisation")

        # Recap
        pdf.sub_title("Recapitulatif des performances")
        r_headers = ["Reponse", "R2 (lin.)", "RMSE (lin.)", "R2 (RSM)", "RMSE (RSM)"]
        r_rows = []
        for out in outputs:
            if out not in models: continue
            row = [out[:26]]
            row += ([metrics_lin[out]["R²"], metrics_lin[out]["RMSE"]]
                    if out in metrics_lin else ["--", "--"])
            row += ([metrics_rsm[out]["R²"], metrics_rsm[out]["RMSE"]]
                    if out in metrics_rsm else ["--", "--"])
            r_rows.append(row)
        if r_rows:
            pdf.table(r_headers, r_rows, [52, 28, 28, 28, 28])

        # Detail
        for out in outputs:
            if out not in models: continue
            pdf.ln(2)
            pdf.sub_title(f"Reponse : {out}")
            if out in metrics_lin:
                m = metrics_lin[out]
                pdf.kv("R2 (lineaire)", str(m.get("R²", m.get("R2", "--"))), bold_val=True)
                pdf.kv("R2 ajuste", str(m.get("R² ajusté", m.get("R2 ajuste", "--"))))
                pdf.kv("RMSE", str(m.get("RMSE", "--")), bold_val=True)
                if out in eq_lin:
                    pdf._sans(8); pdf.set_text_color(*DoEReport.SLATE)
                    pdf.cell(0, 5, "Equation lineaire :", ln=True)
                    pdf.set_text_color(0, 0, 0)
                    pdf.equation_box(eq_lin[out])
            if out in metrics_rsm:
                m = metrics_rsm[out]
                pdf.kv("R2 (RSM)", str(m.get("R²", m.get("R2", "--"))), bold_val=True)
                pdf.kv("R2 ajuste (RSM)", str(m.get("R² ajusté", m.get("R2 ajuste", "--"))))
                pdf.kv("RMSE (RSM)", str(m.get("RMSE", "--")), bold_val=True)
                if out in eq_rsm:
                    pdf._sans(8); pdf.set_text_color(*DoEReport.SLATE)
                    pdf.cell(0, 5, "Equation RSM :", ln=True)
                    pdf.set_text_color(0, 0, 0)
                    pdf.equation_box(eq_rsm[out])

    # ── SECTION 3 : ANOVA ────────────────────────────────────────────────────
    if anova_results:
        pdf.add_page()
        pdf.section_title("3 -- Analyse de Variance (ANOVA)")
        for out, (anova_df, alpha) in anova_results.items():
            if anova_df is None: continue
            pdf.sub_title(f"Reponse : {out}  (alpha = {alpha})")
            df_a = anova_df.reset_index()
            hd = ["Facteur", "SC", "ddl", "F", "p-value", "Sig."]
            rw = []
            for _, row in df_a.iterrows():
                sig = str(row.get("Significatif", "")).replace("✅ ", "Oui").replace("❌ ", "Non")
                rw.append([
                    str(row.iloc[0])[:20],
                    str(round(row["Somme des carrés"], 3)),
                    str(int(row["ddl"])) if not pd.isna(row["ddl"]) else "--",
                    str(round(row["F"], 3)) if not pd.isna(row["F"]) else "--",
                    str(round(row["p-value"], 4)) if not pd.isna(row["p-value"]) else "--",
                    sig,
                ])
            pdf.table(hd, rw, [46, 26, 14, 22, 22, 22])
            pdf.ln(2)

    # ── SECTION 4 : OPTIMISATION ─────────────────────────────────────────────
    if df_opt is not None and not df_opt.empty:
        pdf.add_page()
        pdf.section_title("4 -- Optimisation multi-objectif")

        if objectives:
            pdf.sub_title("Objectifs definis")
            goal_fr = {"maximize": "Maximiser", "minimize": "Minimiser",
                       "target": "Atteindre une cible"}
            for out, obj in objectives.items():
                tstr = (f" (cible = {obj['target']:.3f})"
                        if "target" in obj else "")
                pdf.kv(out[:28],
                       f"{goal_fr.get(obj['goal'], obj['goal'])}{tstr}"
                       f"  |  poids = {obj['weight']}")

        pdf.ln(3)
        pdf.sub_title("Solution optimale")
        best = df_opt.iloc[0]
        pdf.set_fill_color(*DoEReport.TEAL)
        pdf.set_text_color(*DoEReport.WHITE)
        pdf._sans(9, bold=True)
        pdf.cell(0, 8,
                 f"  Desirabilite globale : {best['Desirabilite'] if 'Desirabilite' in best else best.get('Désirabilité', '--'):.4f}",
                 fill=True, ln=True)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(2)
        for inp in inputs:
            if inp in best:
                pdf.kv(f"  {inp[:28]}", f"{best[inp]:.4f}", bold_val=True)
        for pc in [c for c in df_opt.columns if c.startswith("Pred_")]:
            pdf.kv(f"  {pc.replace('Pred_', '')[:24]} (predit)",
                   f"{best[pc]:.4f}", bold_val=True)

        pdf.ln(4)
        pdf.sub_title("Top 10 solutions")
        top = df_opt.head(10)
        th = [c[:13] for c in top.columns]
        tw = [max(12, 182 // len(top.columns))] * len(top.columns)
        tr = [[str(round(v, 3)) if isinstance(v, float) else str(v)
               for v in row] for row in top.values]
        pdf.table(th, tr, tw)

    return bytes(pdf.output())
