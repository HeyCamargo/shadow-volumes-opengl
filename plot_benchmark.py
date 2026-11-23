"""Gera gráficos, tabela LaTeX e resumo a partir de resultados/benchmark.csv.

Uso:
    python plot_benchmark.py                 # lê ./resultados
    python plot_benchmark.py --dir resultados
"""
import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({k: (float(v) if v not in ("", "nan") else float("nan")) for k, v in r.items()})
    return rows


def get(rows, tris, lights, shadows):
    for r in rows:
        if r["tris_total"] == tris and r["lights"] == lights and r["shadows"] == shadows:
            return r
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="resultados")
    a = ap.parse_args()
    rows = load(os.path.join(a.dir, "benchmark.csv"))
    if not rows:
        raise SystemExit("benchmark.csv está vazio.")

    tris_list = sorted({r["tris_total"] for r in rows})
    lights_list = sorted({r["lights"] for r in rows})
    cmap = plt.get_cmap("viridis")
    colors = {t: cmap(i / max(len(tris_list) - 1, 1) * 0.9) for i, t in enumerate(tris_list)}

    # ---- 1) FPS x nº de luzes ------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for t in tris_list:
        for sh, ls, mk in ((1, "-", "o"), (0, "--", "s")):
            pts = [(l, get(rows, t, l, sh)) for l in lights_list]
            pts = [(l, r["fps"]) for l, r in pts if r]
            if pts:
                ax.plot(*zip(*pts), ls=ls, marker=mk, color=colors[t], ms=4,
                        label=f"{int(t)} tri ({'com' if sh else 'sem'} sombras)")
    ax.set_yscale("log")
    ax.set_xticks(lights_list)
    ax.set_xlabel("Número de luzes")
    ax.set_ylabel("FPS (escala log)")
    ax.set_title("Desempenho x número de luzes")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(a.dir, "fig1_fps_x_luzes.png"), dpi=200)
    plt.close(fig)

    # ---- 2) tempo de quadro x triângulos (com sombras) --------------------------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for l in lights_list:
        pts = [(t, get(rows, t, l, 1)) for t in tris_list]
        pts = [(t, r["frame_ms"]) for t, r in pts if r]
        if pts:
            ax.plot(*zip(*pts), marker="o", ms=4, label=f"{int(l)} luz(es)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Triângulos na cena (casters)")
    ax.set_ylabel("Tempo de quadro (ms, escala log)")
    ax.set_title("Custo x complexidade da malha (com sombras)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(a.dir, "fig2_quadro_x_triangulos.png"), dpi=200)
    plt.close(fig)

    # ---- 3) custo por etapa (malha mais complexa, com sombras) ------------------
    tmax = tris_list[-1]
    parts = [("cpu_sil_ms", "CPU: silhueta", "#4c78a8"),
             ("cpu_ext_ms", "CPU: extrusão", "#f58518"),
             ("cpu_upload_ms", "CPU: envio ao buffer", "#e45756"),
             ("gpu_stencil_ms", "GPU: stencil (volumes)", "#72b7b2"),
             ("gpu_light_ms", "GPU: luz", "#54a24b"),
             ("gpu_ambient_ms", "GPU: ambiente", "#b6992d")]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    xs = np.arange(len(lights_list))
    bottom = np.zeros(len(lights_list))
    sel = [get(rows, tmax, l, 1) for l in lights_list]
    for key, label, col in parts:
        vals = np.array([r[key] if r and not np.isnan(r[key]) else 0.0 for r in sel])
        if vals.sum() > 0:
            ax.bar(xs, vals, bottom=bottom, label=label, color=col)
            bottom += vals
    other = np.array([max(r["frame_ms"] - b, 0.0) if r else 0.0 for r, b in zip(sel, bottom)])
    ax.bar(xs, other, bottom=bottom, label="Outros (driver, sincronização)", color="#bab0ac")
    ax.set_xticks(xs)
    ax.set_xticklabels([int(l) for l in lights_list])
    ax.set_xlabel("Número de luzes")
    ax.set_ylabel("ms por quadro")
    ax.set_title(f"Custo por etapa ({int(tmax)} triângulos, com sombras)")
    ax.legend(fontsize=7)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(a.dir, "fig3_custo_por_etapa.png"), dpi=200)
    plt.close(fig)

    # ---- 4) sobrecusto das sombras (quadro com / quadro sem) ------------------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for t in tris_list:
        pts = []
        for l in lights_list:
            on, off = get(rows, t, l, 1), get(rows, t, l, 0)
            if on and off and off["frame_ms"] > 0:
                pts.append((l, on["frame_ms"] / off["frame_ms"]))
        if pts:
            ax.plot(*zip(*pts), marker="o", ms=4, color=colors[t], label=f"{int(t)} tri")
    ax.axhline(1.0, color="k", lw=0.8)
    ax.set_xticks(lights_list)
    ax.set_xlabel("Número de luzes")
    ax.set_ylabel("Tempo com sombras / sem sombras")
    ax.set_title("Sobrecusto do z-fail em relação à iluminação sem sombras")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(a.dir, "fig4_sobrecusto_sombras.png"), dpi=200)
    plt.close(fig)

    # ---- tabela LaTeX (FPS com / sem sombras) ---------------------------------
    with open(os.path.join(a.dir, "tabela_fps.tex"), "w", encoding="utf-8") as f:
        f.write("\\begin{table}[t]\n\\centering\n")
        f.write("\\caption{FPS médio com / sem sombras, por número de luzes.}\n\\label{tab:fps}\n")
        f.write("\\begin{tabular}{r" + "c" * len(lights_list) + "}\n\\hline\n")
        f.write("Triângulos & " + " & ".join(f"{int(l)} luz(es)" for l in lights_list) + " \\\\\n\\hline\n")
        for t in tris_list:
            cells = []
            for l in lights_list:
                on, off = get(rows, t, l, 1), get(rows, t, l, 0)
                cells.append(f"{on['fps']:.0f} / {off['fps']:.0f}" if on and off else "--")
            f.write(f"{int(t)} & " + " & ".join(cells) + " \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")

    # ---- resumo em texto -------------------------------------------------------
    lines = ["RESUMO DO BENCHMARK", "=" * 40]
    lines.append("Tempo de quadro x nº de luzes (com sombras): ajuste linear  ms = a + b * luzes")
    for t in tris_list:
        pts = [(l, get(rows, t, l, 1)) for l in lights_list]
        pts = [(l, r["frame_ms"]) for l, r in pts if r]
        if len(pts) >= 3:
            x, y = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
            b, a0 = np.polyfit(x, y, 1)
            pred = a0 + b * x
            ss_res, ss_tot = ((y - pred) ** 2).sum(), ((y - y.mean()) ** 2).sum()
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            lines.append(f"  {int(t):6d} tri: b = {b:7.3f} ms/luz | a = {a0:7.3f} ms | R2 = {r2:.4f}")
    lines.append("")
    lines.append(f"Participação da CPU (silhueta+extrusão+envio) no quadro, {int(tmax)} tri, com sombras:")
    for l in lights_list:
        r = get(rows, tmax, l, 1)
        if r:
            lines.append(f"  {int(l)} luz(es): CPU {r['cpu_total_ms']:.2f} ms de {r['frame_ms']:.2f} ms "
                         f"({100 * r['cpu_total_ms'] / r['frame_ms']:.0f}%)")
    lines.append("")
    lines.append("Sobrecusto médio das sombras (quadro com / quadro sem):")
    for t in tris_list:
        rat = []
        for l in lights_list:
            on, off = get(rows, t, l, 1), get(rows, t, l, 0)
            if on and off and off["frame_ms"] > 0:
                rat.append(on["frame_ms"] / off["frame_ms"])
        if rat:
            lines.append(f"  {int(t):6d} tri: {np.mean(rat):.2f}x")
    text = "\n".join(lines)
    with open(os.path.join(a.dir, "resumo.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(text)
    print("\nArquivos gerados em:", os.path.abspath(a.dir))
    print("  fig1_fps_x_luzes.png, fig2_quadro_x_triangulos.png, fig3_custo_por_etapa.png,")
    print("  fig4_sobrecusto_sombras.png, tabela_fps.tex, resumo.txt")


if __name__ == "__main__":
    main()