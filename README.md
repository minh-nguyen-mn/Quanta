# LLM-Assisted Multi-Persona Trading Strategy (QQQ)

## Quanta Fellowship – LLM-Based Strategy Builder

---

## Overview

This project implements a **bias-free, multi-signal quantitative trading strategy** for QQQ using a **persona-driven signal design framework**. The strategy is evaluated using **strict train / validation / blind out-of-sample splits**, adheres to **explicit leverage constraints**, and includes **full diagnostic analysis** (PnL curves, correlation matrices, Sharpe metrics).

The goal of this submission is to demonstrate:
- Robust signal construction
- Absence of lookahead bias
- Orthogonal signal behavior
- Strong out-of-sample generalization

---

## Dataset & Splits

The QQQ dataset is split exactly as specified:

| Dataset | Date Range | Trading Days |
|------|-----------|--------------|
| **Train** | 2000-01-03 → 2015-12-31 | 4,025 |
| **Validation** | 2016-01-04 → 2021-12-31 | 1,511 |
| **Blind OOS** | 2022-01-03 → 2025-12-05 | 986 |

**Important:**
- Blind OOS data is never used for parameter selection, tuning, or model adjustment.
- All signals and parameters are finalized before evaluating blind performance.

---

## Execution & Lookahead Bias Control

- All indicators are computed using **rolling windows only**.
- No statistics are calculated using full-sample information.
- Signals are generated at the **close of day _t_**.
- Positions are executed at the **close of day _t_** and held until the **close of day _t+1_**.
- PnL is computed as:

