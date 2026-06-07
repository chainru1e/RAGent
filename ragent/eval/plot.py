"""평가 결과 시각화 모듈."""

import json
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import os

# 한글 폰트 설정
def _set_korean_font():
    candidates = ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]
    available = {f.name for f in fm.fontManager.ttflist}
    for font in candidates:
        if font in available:
            matplotlib.rc("font", family=font)
            return
    matplotlib.rc("font", family="DejaVu Sans")

_set_korean_font()
matplotlib.rcParams["axes.unicode_minus"] = False

MODE_COLORS = {
    "Dense Only":        "#5B9BD5",
    "Hybrid":            "#70AD47",
    "Hybrid + Reranker": "#ED7D31",
}


def plot_avg_score(metrics: dict, k_values: list[int], save_path: str = None):
    """K값별 평균 관련도 점수 묶음 막대 그래프."""
    modes = list(metrics.keys())
    x = np.arange(len(k_values))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, mode in enumerate(modes):
        scores = [metrics[mode][f"avg_score@{k}"] for k in k_values]
        bars = ax.bar(x + i * width, scores, width, label=mode,
                      color=MODE_COLORS.get(mode, "#999"), zorder=3)
        for bar, val in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.05,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Top-K", fontsize=12)
    ax.set_ylabel("평균 관련도 점수 (1~5)", fontsize=12)
    ax.set_title("검색 모드별 평균 관련도 점수 비교", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"@{k}" for k in k_values])
    ax.set_ylim(0, 5.5)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"저장: {save_path}")
    plt.show()


def plot_relevant_rate(metrics: dict, k_values: list[int], save_path: str = None):
    """K값별 Relevant Rate (score >= 4) 묶음 막대 그래프."""
    modes = list(metrics.keys())
    x = np.arange(len(k_values))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, mode in enumerate(modes):
        rates = [metrics[mode][f"relevant_rate@{k}"] * 100 for k in k_values]
        bars = ax.bar(x + i * width, rates, width, label=mode,
                      color=MODE_COLORS.get(mode, "#999"), zorder=3)
        for bar, val in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.8,
                    f"{val:.0f}%", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Top-K", fontsize=12)
    ax.set_ylabel("관련 청크 비율 (score ≥ 4, %)", fontsize=12)
    ax.set_title("검색 모드별 Relevant Rate 비교", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"@{k}" for k in k_values])
    ax.set_ylim(0, 110)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"저장: {save_path}")
    plt.show()


def save_metrics_json(metrics: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"수치 저장: {path}")
