# Glasauer & Shi (2021) duration reproduction

Source: GIN `sglasauer/duration_reproduction` (doi:10.12751/g-node.hdsam3), CC BY-SA 4.0 (LICENSE). Glasauer S, Shi Z.
The origin of Vierordt's law: the experimental protocol matters. PsyCh J 10, 2021. https://doi.org/10.1002/pchj.464

`Glasauer_Shi_2021b_duration_reproduction.csv` was exported from `rep_randwalk.mat` (copied from
context-efficient-observer-/data/context_reproduction/glasauer_shi_2021). **Its column labels `NSub` and `condition`
are swapped** (checked 2026-11-09): `condition` (1-14) is the participant, `NSub` (1-20) the 40-trial block, and `NT`
(1-800) the trial order within a participant. `RandLevel` 1 = random walk (lag-1 correlation of durations .99),
2 = randomized (≈ 0); each participant has 400 trials of each, in halves. One row has a missing RandLevel.
`kmodel1pv.m`, `fitdata1pv.m`, `data_fit_paper.m`: the authors' two-stage model and fitting code.
