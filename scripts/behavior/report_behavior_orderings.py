"""Readable report and figures for corrected behavioral analysis C."""
from pathlib import Path
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from sdu import paths as SP  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = SP.RESULTS_BEHAVIOR / "analysis/behavior_extra"
os.environ.setdefault("MPLCONFIGDIR", str(OUT / "mplconfig"))
for name in ["times.ttf", "timesbd.ttf", "timesi.ttf"]:
    file = Path("C:/Windows/Fonts") / name
    if file.exists():
        font_manager.fontManager.addfont(str(file))
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42, "axes.labelsize": 10,
                     "axes.titlesize": 11, "xtick.labelsize": 9, "ytick.labelsize": 9})
bayes = pd.read_csv(OUT / "ordering_bayes_factors.csv")
mass = pd.read_csv(OUT / "ordering_probability_sensitivity.csv")
cells = pd.read_csv(OUT / "ordering_cell_summary.csv")
components = pd.read_csv(OUT / "ordering_constraint_components.csv")
loo = pd.read_csv(OUT / "ordering_leave_one_out.csv")
names = {"reliability": "Reliability weighting", "current": "Current-coherence grouping",
         "reset": "Contextual-reset pattern", "flat": "Approximately equal cells"}
colors = {"reliability": "#B27935", "current": "#398B71", "reset": "#7260A4", "flat": "#777777"}
primary = bayes[bayes.prior_tau.eq(.1) & bayes.delta.eq(.02) & bayes.min_gap.eq(.02)].copy()

fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.2), sharey=True)
for ax, exp, color in zip(axes, ["Exp 1", "Exp 2"], ["#398B71", "#5B7C99"]):
    data = cells[cells.experiment.eq(exp)].set_index("cell").loc[["HL", "LL", "HH", "LH"]]
    ax.errorbar(np.arange(4), data["mean"], yerr=[data["mean"] - data.ci_low, data.ci_high - data["mean"]],
                marker="o", color=color, linewidth=1.2, capsize=3, markersize=5)
    ax.set(title=exp.replace("Exp", "Experiment"), xticks=np.arange(4), xticklabels=data.index,
           xlabel="Previous / current coherence", ylim=(-.035, .205))
    ax.axhline(0, color="#777777", linewidth=.7, linestyle="--")
axes[0].set_ylabel("Previous-duration slope (s/s)")
fig.subplots_adjust(left=.105, right=.985, bottom=.22, top=.87, wspace=.14)
for ext in ["png", "pdf"]:
    fig.savefig(OUT / f"ordering_observed_cells.{ext}", dpi=350)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.9), sharey=True)
for ax, exp in zip(axes, ["Exp 1", "Exp 2"]):
    for pattern, label in names.items():
        data = bayes[bayes.experiment.eq(exp) & bayes.delta.eq(.02) & bayes.min_gap.eq(.02) & bayes.pattern.eq(pattern)].sort_values("prior_tau")
        ax.plot(range(3), data.BF_vs_unconstrained, marker="o", markersize=4.5, linewidth=1.3,
                color=colors[pattern], label=label)
    ax.axhline(1, color="#555555", linewidth=.9, linestyle="--")
    ax.set(title=exp.replace("Exp", "Experiment"), xticks=[0, 1, 2], xticklabels=[".05", ".10", ".20"],
           xlabel="Prior scale for mean contrasts", yscale="log", ylim=(.001, 300))
    ax.grid(axis="y", which="major", linewidth=.4, alpha=.25)
axes[0].set_ylabel("Bayes factor relative to unrestricted means")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=9)
fig.subplots_adjust(left=.11, right=.985, bottom=.32, top=.86, wspace=.15)
fig.text(.5, .98, "Complete constraints: equality bound .02; minimum gap .02", ha="center", va="top", fontsize=10)
for ext in ["png", "pdf"]:
    fig.savefig(OUT / f"ordering_evidence_sensitivity.{ext}", dpi=350)
plt.close(fig)


def bf(exp, pattern):
    return float(primary[primary.experiment.eq(exp) & primary.pattern.eq(pattern)].BF_vs_unconstrained.iloc[0])


def ratio_range(exp, first, second):
    data = bayes[bayes.experiment.eq(exp)].pivot(index=["prior_tau", "delta", "min_gap"], columns="pattern", values="BF_vs_unconstrained")
    ratio = data[first] / data[second]
    return float(ratio.min()), float(ratio.max())


def table(data, columns, formats=None):
    formats = formats or {}
    result = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in data.iterrows():
        result.append("| " + " | ".join(formats.get(c, str)(row[c]) for c in columns) + " |")
    return "\n".join(result)


evidence = pd.DataFrame([{"模式": names[p], "实验 1 BF": bf("Exp 1", p), "实验 2 BF": bf("Exp 2", p)} for p in names])
evidence_text = table(evidence, ["模式", "实验 1 BF", "实验 2 BF"], {"实验 1 BF": lambda x: f"{x:.3g}", "实验 2 BF": lambda x: f"{x:.3g}"})
loo_summaries = []
for exp in ["Exp 1", "Exp 2"]:
    wide = loo[loo.experiment.eq(exp)].pivot(index="dropped_participant", columns="pattern", values="BF_vs_unconstrained")
    target = "current" if exp == "Exp 1" else "reset"
    count = int(wide.idxmax(axis=1).eq(target).sum())
    loo_summaries.append(f"{exp}：逐个移除被试后，{names[target]} 在 {count}/22 次复核中仍为首位。")
first_range = ratio_range("Exp 1", "current", "reliability")
second_range = ratio_range("Exp 2", "reset", "current")
flat_range = ratio_range("Exp 2", "reset", "flat")
report = f"""**四种转换的排序检验：修正与深入分析（分析 C）**

本轮仅修改排序分析。E、F 不纳入当前理论论证；A、B、D 的结果文件保持不变。未改 Kalman 候选模型或拟合结果；稿件整合由独立构建步骤处理。旧链式约束输出保存在 `orderings_legacy_chain_constraints.csv`。

**修正的含义**

单个被试、单种转换内拟合 reproduction bias ~ current duration + previous duration，再对 22 位被试的 previous-duration slope 作等权汇总。两个实验使用与四格行为分析一致的 4,693、4,654 个有效试次。四格均值及标准误在 `ordering_cell_summary.csv`；它们不是试次级 LMM 的替代估计。

“≈”要求同一组内的任意两格之差不超过 δ。“>”要求较高组中每一格都高于较低组中的每一格；不再用相邻比较的链条代表整组关系。主展示沿用原设置 δ=.02、最小组间差=.02；另检查 δ=.015/.03，以及仅要求方向正确的最小差=0。δ=.02 对应在 0.8–1.6 s 的前一时长范围上，斜率差产生约 16 ms 的端到端影响差。界限是事后敏感性假定，未称为预注册阈值。

1. 可靠性排序：HL > {{LL, HH}} > LH。
2. 当前一致性分组：{{HL, LL}} > {{HH, LH}}。
3. 情景重置排序：LL > {{HH, LH, HL}}。
4. 基准：四格都在近似相等的范围内。

**描述性吻合与证据比较分开**

修正后的 `orderings.csv` 仅报告正态抽样中满足完整约束的比例。`ordering_probability_sensitivity.csv` 另提供 100,000 次按被试成组重抽样；这些比例均不叫“机制的后验概率”。

正式的补充比较使用三个正交的四格均值对比，采用零中心、各方向相同的 proper normal–inverse-Wishart 先验。组内协方差也被积分，而非固定在样本估计上。三个先验尺度 .05、.10、.20 都报告，两个实验共用同一套先验。详细公式在 `ordering_methods.md`。受约束统计假设相对不受约束假设的 Bayes factor 为：

BF(H,U) = P(均值满足 H | 数据,U) / P(均值满足 H | U)。

该证据比较取决于明示的群体模型、先验和界限；不是 Kalman 模型比较，也不能唯一识别认知机制。

主展示设置 δ=.02、最小差=.02、先验尺度=.10：

{evidence_text}

在这组设置下，实验 1 的当前一致性分组相对可靠性排序 BF={bf('Exp 1','current')/bf('Exp 1','reliability'):.2f}；实验 2 的情景重置排序相对当前一致性分组 BF={bf('Exp 2','reset')/bf('Exp 2','current'):.2f}，相对四格近似相同 BF={bf('Exp 2','reset')/bf('Exp 2','flat'):.2f}。

**稳健程度及解释限制**

跨全部界限、两种排序标准及三种先验：实验 1 的当前分组/可靠性排序 BF 为 {first_range[0]:.2f}–{first_range[1]:.2f}；实验 2 的情景重置/当前分组为 {second_range[0]:.2f}–{second_range[1]:.2f}，情景重置/四格近似相同为 {flat_range[0]:.2f}–{flat_range[1]:.2f}。首位排序保持，但证据强度随假定改变，不能只摘录主展示数值。

{' '.join(loo_summaries)}

实验 2 中，LL 比每一其余条件至少高 .02 的后验区域质量约 .789；其余三格所有两两差均在 ±.02 内的区域质量约 .243。完整重置排序的区域质量约 .229。最主要的精度限制来自“其余三格是否足够相似”，不能把较大的 BF 写成完整排序已经具有接近 100% 的概率。

两种标准还必须区分：当最小组间差=0 时，不同排序区域可能重叠；相应比例不能归一化成三个解释的选择概率。主展示最小差=δ 时四个区域互斥，但它们也未穷尽所有可能模式。相当一部分抽样位于四个区域之外，报告保留了这一部分。

A 中的跨实验交互检验并没有改变。两个实验的排序证据不同，不等于两个实验的机制已经被统计区分。情景重置排序的提出也包含事后观察，应保持探索性表述。

**可用于论文的结论**

在明确的排序约束及敏感性假定下，实验 1 更支持按当前一致性分组的行为预测；实验 2 更支持 LL 单独升高的预测。结果为一致性调节的延续及其可能受到的情景门控提供互补约束。它们不能单独决定具体 Kalman 实现或感知、决策、执行阶段。

**文件与复现**

运行 `.venv\\Scripts\\python.exe behavior/behavior_orderings.py`；再运行 `report_behavior_orderings.py` 生成本报告与图。七项语义与数值检查位于 `test_behavior_orderings.py`。随机种子、输入哈希、抽样次数、先验及 Monte Carlo 区间定义在 `ordering_analysis_manifest.json`。所有先验区域至少取得 113 个 Monte Carlo 命中；罕见后验区域的数值误差另随表报告。

方法依据：[Murphy 的高斯共轭分析](https://groups.seas.harvard.edu/courses/cs281/papers/murphy-2007.pdf)，[Hoijtink 等的 Bayes factor 教程](https://jorismulder.com/wp-content/uploads/2021/01/hoijtink2019_bftutorial.pdf)。
"""
(OUT / "ordering_analysis_report.md").write_text(report, encoding="utf-8")
methods = r"""**Statistical specification of the corrected ordering analysis**

Let b_s=(LL,LH,HL,HH)' contain one participant's controlled previous-duration slopes.
All analyses retain the participant vector as the resampling unit. Define z_s=M b_s,
where M has rows (.5,-.5,.5,-.5), (.5,.5,-.5,-.5), (.5,-.5,-.5,.5).
These orthonormal contrasts remove the unrestricted common slope. All hypotheses
are unchanged by adding the same constant to the four slopes.

Likelihood: z_s independently follows MVN(mu,Sigma), s=1,...,22.
Prior: Sigma~IW(nu0,Psi0); mu|Sigma~N(0,Sigma/kappa0).
Set d=3, kappa0=1, nu0=5, Psi0=kappa0*(nu0-d+1)*tau^2 I.
The prior marginal for mu is a multivariate t with df=3 and scale tau^2 I;
tau is a t scale, not a standard deviation. tau=.05,.10,.20 is a sensitivity grid.

kappa_n=kappa0+n; nu_n=nu0+n; mu_n=n*mean(z)/kappa_n.
Psi_n=Psi0+sum_s[(z_s-mean(z))(z_s-mean(z))']
      +kappa0*n/kappa_n*mean(z)*mean(z)'.
The exact marginal mean posterior is multivariate t with df=nu_n-d+1,
location mu_n and scale Psi_n/[kappa_n*(nu_n-d+1)].
Cell deviations are reconstructed as M' mu. Region membership is tested on those
deviations; retaining a common slope would give identical results.

Every near-equal group must have max-min<=delta. Every ordered group must have
min(higher)-max(lower)>gap. The primary setting follows the earlier analysis's
delta=gap=.02; delta=.015/.03 and gap=0 are sensitivity settings. With gap=0,
regions can overlap. No region probabilities are normalized into mechanism weights.

For an encompassing prior pi_U(mu,Sigma), the constrained statistical hypothesis
uses that SAME prior conditioned on its complete region H. Its evidence ratio is
BF_HU=P_U(H|data)/P_U(H). Pairwise ratios are BF_HU/BF_KU. This is a comparison of
specified constrained population-mean models, not of full observer implementations.
The priors, bands, and contextual-reset interpretation are post hoc; all sensitivity
settings and the approximate-equality baseline are retained.

Monte Carlo integration uses 2,000,000 prior and 1,000,000 posterior draws per prior
scale/experiment, in chunks. Draws are independent multivariate-t draws, not MCMC.
The output's 97.5% binomial intervals quantify integration error, not substantive
data uncertainty. Using both prior and posterior intervals yields conservative
at-least-95% ratio bounds. Zero-hit events must not be treated as impossible.
Descriptive normal sampling and participant bootstrap are reported separately.

Sources: Murphy (2007), section 9 (proper NIW posterior and marginal mean);
Hoijtink et al. (2019), encompassing/informative-hypothesis comparisons.
https://groups.seas.harvard.edu/courses/cs281/papers/murphy-2007.pdf
https://jorismulder.com/wp-content/uploads/2021/01/hoijtink2019_bftutorial.pdf
"""
(OUT / "ordering_methods.md").write_text(methods, encoding="utf-8")
log = OUT / "behavior_extra.log"
notice = "C SECTION SUPERSEDED: use ordering_analysis_report.md and ordering_analysis_manifest.json. Original chained-constraint C numbers below are retained only as history.\n\n"
if log.exists() and not log.read_text(encoding="utf-8").startswith("C SECTION SUPERSEDED"):
    snapshot = OUT / "behavior_extra_original.log"
    if not snapshot.exists():
        snapshot.write_bytes(log.read_bytes())
    log.write_text(notice + log.read_text(encoding="utf-8"), encoding="utf-8")
print(json.dumps({"report": "ordering_analysis_report.md", "figures": ["ordering_observed_cells.png", "ordering_evidence_sensitivity.png"], "primary_BF_current_vs_reliability_exp1": bf("Exp 1", "current") / bf("Exp 1", "reliability"), "primary_BF_reset_vs_current_exp2": bf("Exp 2", "reset") / bf("Exp 2", "current")}, indent=2))
