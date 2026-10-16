"""공식(평면환산거리)과 실측 환승 거리를 비교하는 덤벨 차트를 그린다."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
FORMULA, MEASURED = "#eb6834", "#2a78d6"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

for name in ("Noto Sans CJK KR", "NanumGothic", "AppleGothic", "Malgun Gothic"):
    if any(name in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = name
        break

df = pd.read_csv(ROOT / "data" / "transfer_distance.csv")
df["label"] = df["station"] + " " + df["route"]
df["gap"] = df["formula_m"] - df["measured_m"]
df = df.sort_values("measured_m").reset_index(drop=True)

fig, ax = plt.subplots(figsize=(8, 5.2), dpi=200)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)

for i, r in df.iterrows():
    ax.plot([r.measured_m, r.formula_m], [i, i], color=GRID, lw=2, zorder=1, solid_capstyle="round")
    ax.text(r.formula_m + 14, i, f"+{r.gap:.0f}m", va="center", fontsize=8, color=MUTED)

ax.scatter(df.measured_m, df.index, s=60, color=MEASURED, edgecolor=SURFACE, linewidth=2, zorder=3, label="직접 잰 거리 (실측)")
ax.scatter(df.formula_m, df.index, s=60, color=FORMULA, edgecolor=SURFACE, linewidth=2, zorder=3, label="공식 계산 거리 (평면환산거리)")

ax.set_yticks(df.index, df.label, color=INK, fontsize=9)
ax.set_xlim(0, 900)
ax.set_xlabel("환승 거리 (m)", color=MUTED, fontsize=9)
ax.tick_params(axis="x", colors=MUTED, labelsize=8)
ax.tick_params(axis="y", length=0)
ax.grid(axis="x", color=GRID, lw=0.8)
ax.set_axisbelow(True)
for s in ax.spines.values():
    s.set_visible(False)

ax.set_title("10개 경로 모두에서 공식이 실제보다 길게 계산했다", loc="left", fontsize=12, color=INK, pad=28, fontweight="bold")
ax.text(0, 1.035, "부산 도시철도 휠체어 환승 경로별 공식 계산값과 실측값 (실측 거리 오름차순)",
        transform=ax.transAxes, fontsize=8.5, color=MUTED)
ax.legend(loc="lower right", frameon=False, fontsize=8.5, labelcolor=INK)

fig.tight_layout()
out = ROOT / "images" / "formula_vs_measured.png"
fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
print("saved", out)
