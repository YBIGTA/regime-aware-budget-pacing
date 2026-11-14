"""발표용 그림: 입찰 승수 상한 1.0 vs 1.2 (B_state_PI).

입력: traces_5min.csv, decomposition.json (export_traces.py가 생성)
출력: cap-comparison-ko.png (300 dpi), cap-comparison-ko.svg
"""
import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

HERE = Path(__file__).resolve().parent
DAYS = (100065, 100080)

for path in ("/System/Library/Fonts/AppleSDGothicNeo.ttc",):
    if Path(path).exists():
        font_manager.fontManager.addfont(path)
mpl.rcParams.update({
    "font.family": "Apple SD Gothic Neo",
    "axes.unicode_minus": False,
    "svg.fonttype": "none",
    "font.size": 13,
})

INK = "#111827"
MUTED = "#6b7280"
GRID = "#e5e7eb"
C10 = "#2563eb"   # 상한 1.0
C12 = "#ea580c"   # 상한 1.2
COMMON = "#94a3b8"
GOOD = "#15803d"
BAD = "#b91c1c"

rows = list(csv.DictReader(open(HERE / "traces_5min.csv")))
deco = {int(k): v for k, v in json.loads((HERE / "decomposition.json").read_text()).items()}


def series(day):
    R = [r for r in rows if int(r["day"]) == day]
    hour = np.array([0.0] + [float(r["hour"]) for r in R])
    budget = float(R[0]["budget"])
    tgt = np.array([0.0] + [float(r["target_cumulative"]) for r in R])
    g = {}
    for cap in ("1_0", "1_2"):
        cum = np.array([0.0] + [float(r[f"cumulative_cap{cap}"]) for r in R])
        g[cap] = 100 * (tgt - cum) / budget
    return hour, g, budget


fig = plt.figure(figsize=(17, 11.2))
gs = fig.add_gridspec(2, 2, left=0.06, right=0.985, top=0.80, bottom=0.095,
                      hspace=0.42, wspace=0.14, height_ratios=[1.05, 1])

# ---------- 헤더 ----------
fig.text(0.06, 0.965, "입찰 승수 상한 완화의 효과는 날짜별로 달랐다",
         fontsize=26, fontweight="bold", color=INK, va="top")
fig.text(0.06, 0.912,
         "200일 평균 합산 손실 10.15% 감소  ·  82일 개선 / 114일 악화 / 4일 사실상 동일",
         fontsize=17, color=INK, va="top")
fig.text(0.06, 0.872,
         "B_state_PI, 검증 200일. 경매 테이프·일일 예산·목표 지출 경로·기본 승수 예측기·PI 계수는 동일하고, "
         "최종 출력과 anti-windup의 승수 상한만 1.0 → 1.2로 바꿨다.  합산 손실 = 추종 오차 + 미소진율",
         fontsize=12, color=MUTED, va="top")

# ---------- 위쪽: 시간별 추종 격차 ----------
YLIM = (-1.0, 2.4)
titles = {100065: "100065 · 합산 손실 개선 최대", 100080: "100080 · 합산 손실 악화 최대"}
title_color = {100065: GOOD, 100080: BAD}
top_axes = []
for col, day in enumerate(DAYS):
    ax = fig.add_subplot(gs[0, col])
    top_axes.append(ax)
    hour, g, budget = series(day)
    ax.axhline(0, color=INK, lw=1.0, zorder=1)
    ax.plot(hour, g["1_0"], color=C10, lw=3.2, solid_capstyle="round", label="상한 1.0 (실선)", zorder=2)
    ax.plot(hour, g["1_2"], color=C12, lw=1.9, ls=(0, (4, 2.2)), label="상한 1.2 (점선)", zorder=3)
    ax.set_xlim(0, 24)
    ax.set_ylim(*YLIM)
    ax.set_xticks(range(0, 25, 3))
    ax.set_xticklabels([f"{h}시" for h in range(0, 25, 3)])
    ax.set_xlabel("시각", fontsize=13)
    if col == 0:
        ax.set_ylabel("추종 격차 (일일 예산 대비 %)", fontsize=13)
    ax.set_title(titles[day], loc="left", fontsize=17, fontweight="bold",
                 color=title_color[day], pad=10)
    ax.text(0.0, YLIM[1] + 0.02, "", transform=ax.transData)
    ax.text(0.2, -0.93, "▲ 양수: 누적 목표보다 뒤처짐   ▼ 음수: 누적 목표보다 앞섬",
            fontsize=10.5, color=MUTED, va="bottom")

    d = deco[day]
    s1, s2 = d["score_cap1_0"], d["score_cap1_2"]
    te1, te2 = 100 * s1["tracking_error"], 100 * s2["tracking_error"]
    un1, un2 = 100 * s1["unspent"], 100 * s2["unspent"]
    box = (f"추종 오차   {te1:.3f}% → {te2:.3f}%   ({te2 - te1:+.5f}%p)\n"
           f"미소진율     {un1:.3f}% → {un2:.3f}%   ({un2 - un1:+.5f}%p)")
    bx, by = (0.20, 0.97) if day == 100065 else (0.015, 0.97)
    ax.text(bx, by, box, transform=ax.transAxes, ha="left", va="top", fontsize=11.5,
            color=INK, linespacing=1.6,
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=GRID))

    if day == 100065:
        ax.axvspan(18, 24, color=C12, alpha=0.06, zorder=0)
        ax.annotate("추가 지출 대부분이 18시 이후 발생\n(순증 +82.555 중 +82.481)",
                    xy=(21.6, 0.62), xytext=(12.6, 1.05), fontsize=12, color=C12,
                    fontweight="bold", ha="center", va="center",
                    arrowprops=dict(arrowstyle="->", color=C12, lw=1.4))
    else:
        ax.axvspan(18, 24, color=MUTED, alpha=0.07, zorder=0)
        ins = ax.inset_axes([0.60, 0.55, 0.385, 0.42])
        ins.axhline(0, color=INK, lw=0.8)
        m = hour >= 18
        ins.plot(hour[m], g["1_0"][m], color=C10, lw=2.6)
        ins.plot(hour[m], g["1_2"][m], color=C12, lw=1.6, ls=(0, (4, 2.2)))
        ins.set_xlim(18, 24)
        ins.set_ylim(-0.8, 0.5)
        ins.set_xticks([18, 20, 22, 24])
        ins.set_xticklabels(["18시", "20시", "22시", "24시"], fontsize=10)
        ins.tick_params(axis="y", labelsize=10)
        ins.set_title("18:00~24:00 확대", fontsize=11, color=INK, loc="left", pad=4)
        ins.grid(color=GRID, lw=0.7)
        ins.set_facecolor("white")
        ins.annotate(f"최종 {g['1_0'][-1]:+.3f}% → {g['1_2'][-1]:+.3f}%",
                     xy=(24, g["1_2"][-1]), xytext=(20.1, 0.33), fontsize=10, color=C12,
                     arrowprops=dict(arrowstyle="->", color=C12, lw=1))
        mark_inset(ax, ins, loc1=3, loc2=4, fc="none", ec="#9ca3af", lw=0.8)

# ---------- 아래쪽: 워터폴 ----------
for col, day in enumerate(DAYS):
    ax = fig.add_subplot(gs[1, col])
    d = deco[day]
    steps = [
        ("둘 다 낙찰한 건의\n결제액 차이", d["common_payment_delta"], f"{d['common_wins']:,}건", COMMON),
        ("1.2에서만 낙찰한\n건의 지출", d["only12_spend"], f"{d['only12_wins']:,}건", C12),
        ("1.0에서만 낙찰한\n건의 지출 차감", d["only1_spend_neg"], f"{d['only1_wins']:,}건", C10),
    ]
    total = d["total_spend_delta"]
    x = np.arange(4)
    run = 0.0
    tops = []
    for i, (label, v, cnt, color) in enumerate(steps):
        ax.bar(i, v, bottom=run, color=color, width=0.62, zorder=2)
        nxt = run + v
        ax.plot([i + 0.31, i + 0.69], [nxt, nxt], color=MUTED, lw=1, ls=":", zorder=1)
        tops.append((i, run, nxt, v, cnt))
        run = nxt
    ax.bar(3, total, color=INK, width=0.62, zorder=2)
    ax.axhline(0, color=INK, lw=1.0, zorder=1)

    lo = min(0, *[min(a, b) for _, a, b, _, _ in tops], total)
    hi = max(0, *[max(a, b) for _, a, b, _, _ in tops], total)
    span = hi - lo
    ax.set_ylim(lo - 0.30 * span, hi + 0.42 * span)
    pad = 0.03 * span
    for i, a, b, v, cnt in tops:
        top, bottom = max(a, b), min(a, b)
        if v >= 0:
            ax.text(i, top + pad, f"{v:+,.3f}\n({cnt})", ha="center", va="bottom", fontsize=12.5, color=INK)
        else:
            ax.text(i, bottom - pad, f"{v:+,.3f}\n({cnt})", ha="center", va="top", fontsize=12.5, color=INK)
    tl = f"{total:+,.3f}\n(순 낙찰 {d['net_win_change']:+d}건)"
    if total >= 0:
        ax.text(3, total + pad, tl, ha="center", va="bottom", fontsize=13, fontweight="bold", color=INK)
    else:
        ax.text(3, total - pad, tl, ha="center", va="top", fontsize=13, fontweight="bold", color=INK)

    ax.set_xticks(x)
    ax.set_xticklabels([s[0] for s in steps] + ["하루 총지출\n차이 (1.2 - 1.0)"], fontsize=12)
    ax.set_xlim(-0.55, 3.55)
    if col == 0:
        ax.set_ylabel("지출 차이 (1.2 - 1.0)", fontsize=13)
    ax.set_title(f"{day} · 하루 총지출 차이 분해  (일일 예산 {d['budget']:,.3f})", loc="left", fontsize=15, fontweight="bold",
                 color=INK, pad=10)
    lo_t, hi_t = ax.get_ylim()
    ticks = [t for t in ax.get_yticks() if lo_t <= t <= hi_t]
    ax.text(1.0, 1.035, f"세로축 범위 다름: {ticks[0]:g} ~ {ticks[-1]:g}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=11.5, color="#92400e",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35", fc="#fef3c7", ec="#f59e0b", lw=1))
    if day == 100080:
        ax.text(0.015, 0.97, "13건 추가 · 17건 감소,\n악화는 대부분 미소진율에서 발생",
                transform=ax.transAxes, ha="left", va="top", fontsize=12.5, color=BAD,
                fontweight="bold", linespacing=1.5)

for ax in fig.axes:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#9ca3af")
    ax.tick_params(colors=INK)
for ax in top_axes:
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
for ax in fig.axes[-2:]:
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)

h10, = top_axes[0].plot([], [], color=C10, lw=3.2, label="상한 1.0 (실선)")
h12, = top_axes[0].plot([], [], color=C12, lw=1.9, ls=(0, (4, 2.2)), label="상한 1.2 (점선)")
fig.legend(handles=[h10, h12], loc="upper right", bbox_to_anchor=(0.985, 0.925), ncol=2,
           frameon=False, fontsize=14, handlelength=3.2, columnspacing=2)

fig.text(0.06, 0.016,
         "추종 격차 = 100 × (목표 누적 지출 - 실제 누적 지출) / 일일 예산, 원본 5분 trace(288구간).  "
         "두 날짜는 날짜별 합산 손실 변화의 최대 개선·최대 악화 사례.  아래 패널의 세로축은 날짜마다 다르다.",
         fontsize=10.5, color=MUTED)

fig.savefig(HERE / "cap-comparison-ko.png", dpi=300, facecolor="white")
fig.savefig(HERE / "cap-comparison-ko.svg", facecolor="white")
print("saved")
