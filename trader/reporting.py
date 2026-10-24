"""Small helpers for writing markdown reports and figures."""

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def fmt(v, digits: int = 3) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        if math.isnan(v):
            return "–"
        if v != 0 and (abs(v) < 10 ** -digits or abs(v) >= 1e6):
            return f"{v:.2e}"
        return f"{v:.{digits}f}"
    return str(v)


def md_table(df: pd.DataFrame, digits: int = 3, index: bool = True) -> str:
    cols = ([df.index.name or ""] if index else []) + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for idx, row in df.iterrows():
        cells = ([str(idx)] if index else []) + [fmt(v, digits) for v in row.tolist()]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def savefig(fig, path) -> str:
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return str(path)
