#!/usr/bin/env python3
"""
Figuras de datos del Cap. 5 (matplotlib -> PDF vectorial).
Salida a la carpeta imagenes/ de la plantilla LaTeX.

  1. fig_b0b1_metricas.pdf  — B0 vs B1 x 3 modelos en exactitud, F1 macro y kappa.
  2. fig_divergencia.pdf    — divergencia exactitud <-> kappa (flechas B0->B1).

Paleta Okabe-Ito (segura para daltonismo). Codificación secundaria (trama/relleno
+ etiquetas) para legibilidad en blanco y negro.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / \
    "TFM___JORGE_GARCELA_N_GO_MEZ/Plantilla_TFG_ingles_2019/imagenes"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 10, "font.family": "serif", "axes.edgecolor": "#666666",
    "axes.linewidth": 0.8, "axes.grid": True, "grid.color": "#e6e6e6",
    "grid.linewidth": 0.7, "axes.axisbelow": True, "figure.dpi": 150,
})

MODELOS = ["gemini-\n3.1-flash-lite", "gpt-4o-mini", "gpt-5.4-nano"]
# (B0, B1) por modelo
ACC = {"B0": [0.600, 0.564, 0.571], "B1": [0.661, 0.612, 0.569]}
F1  = {"B0": [0.388, 0.410, 0.430], "B1": [0.443, 0.438, 0.364]}
KAP = {"B0": [0.026, 0.033, 0.052], "B1": [0.066, 0.014, 0.004]}

C_B0, C_B1 = "#E69F00", "#0072B2"   # naranja / azul (Okabe-Ito)

# ---------------------------------------------------------------- Figura 1
def fig_barras():
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.1), constrained_layout=True)
    paneles = [("Exactitud", ACC), ("F1 macro", F1), (r"$\kappa$ de Cohen", KAP)]
    x = range(len(MODELOS)); w = 0.36
    for ax, (titulo, d) in zip(axes, paneles):
        b0 = ax.bar([i - w/2 for i in x], d["B0"], w, label="B0 (baseline)",
                    color=C_B0, edgecolor="white", linewidth=1.2)
        b1 = ax.bar([i + w/2 for i in x], d["B1"], w, label="B1 (Agent Skills)",
                    color=C_B1, edgecolor="white", linewidth=1.2, hatch="///")
        for bars in (b0, b1):
            for r in bars:
                h = r.get_height()
                ax.annotate(f"{h:.3f}", (r.get_x()+r.get_width()/2, h),
                            ha="center", va="bottom", fontsize=6.6,
                            xytext=(0, 1.5), textcoords="offset points")
        ax.set_title(titulo, fontsize=10.5, pad=6)
        ax.set_xticks(list(x)); ax.set_xticklabels(MODELOS, fontsize=7.5)
        ax.set_ylim(0, max(max(d["B0"]), max(d["B1"])) * 1.22)
        ax.tick_params(length=0)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
    axes[0].legend(loc="upper right", fontsize=7.6, frameon=False)
    fig.savefig(OUT / "fig_b0b1_metricas.pdf", bbox_inches="tight")
    plt.close(fig)

# ---------------------------------------------------------------- Figura 2
def fig_divergencia():
    fig, ax = plt.subplots(figsize=(5.4, 4.2), constrained_layout=True)
    cols = ["#0072B2", "#D55E00", "#009E73"]
    nombres = ["gemini-3.1-flash-lite", "gpt-4o-mini", "gpt-5.4-nano"]
    # colocación de etiqueta por modelo (dx, dy, alineación) para evitar overflow
    lab = [(-8, 6, "right"), (8, -2, "left"), (8, 2, "left")]
    for i, nom in enumerate(nombres):
        a0, a1 = ACC["B0"][i], ACC["B1"][i]
        k0, k1 = KAP["B0"][i], KAP["B1"][i]
        ax.annotate("", xy=(a1, k1), xytext=(a0, k0),
                    arrowprops=dict(arrowstyle="-|>", color=cols[i], lw=1.8,
                                    shrinkA=4, shrinkB=4))
        ax.scatter([a0], [k0], s=42, facecolors="white", edgecolors=cols[i],
                   linewidths=1.8, zorder=3)                       # B0 hueco
        ax.scatter([a1], [k1], s=52, color=cols[i], zorder=3)     # B1 relleno
        dx, dy, ha = lab[i]
        ax.annotate(nom, (a1, k1), fontsize=7.8, color=cols[i], ha=ha,
                    xytext=(dx, dy), textcoords="offset points", va="center")
    ax.set_xlim(0.55, 0.68)
    # leyenda de forma (B0 hueco / B1 relleno)
    ax.scatter([], [], s=42, facecolors="white", edgecolors="#444",
               linewidths=1.6, label="B0 (baseline)")
    ax.scatter([], [], s=52, color="#444", label="B1 (Agent Skills)")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.set_xlabel("Exactitud"); ax.set_ylabel(r"$\kappa$ de Cohen")
    ax.axhline(0, color="#bbbbbb", lw=0.8, ls="--")
    ax.tick_params(length=0)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.text(0.99, 0.02,
            "En 2 de 3 modelos la flecha va a la derecha (exactitud ↑)\n"
            r"y hacia abajo ($\kappa$ ↓): las skills mejoran la exactitud"
            "\nsin mejorar el acuerdo. Solo gemini mejora en ambas.",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.0,
            color="#333333")
    fig.savefig(OUT / "fig_divergencia.pdf", bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    fig_barras(); fig_divergencia()
    print("Figuras escritas en", OUT)
    for f in ("fig_b0b1_metricas.pdf", "fig_divergencia.pdf"):
        print("  ", (OUT / f).exists(), f)
