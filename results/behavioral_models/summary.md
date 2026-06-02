# Behavioral Model Results

## Descriptives
- experiment1: N=22, trials=4693, trials/sub=213.3 [188, 230]
- experiment2: N=22, trials=4654, trials/sub=211.5 [186, 230]

## Main LMMs
- experiment1 (current uncertainty moderation): low uncertainty / 70% coherence slope=0.0190; high uncertainty / 30% coherence slope=0.0662; interaction beta=0.0472, SE=0.0201, z=2.35, p=0.01884; converged=True
- experiment2 (same/switch moderation): Switch slope=0.0789; Same slope=0.1077; interaction beta=0.0289, SE=0.0216, z=1.33, p=0.1821; converged=True

## Subject-Level Controlled SDI Contrasts
- experiment1 / HighUncertainty - LowUncertainty: mean diff=0.0477, t(21)=3.52, p=0.002026, dz=0.751
- experiment2 / Same - Switch: mean diff=0.0279, t(21)=2.26, p=0.03432, dz=0.483

## Response History Key Terms
- experiment1 `preDur_c:curCoherence`: beta=-0.1224, SE=0.0525, z=-2.33, p=0.0198
- experiment1 `preResp_long`: beta=0.0704, SE=0.0071, z=9.90, p=4.018e-23
- experiment2 `preResp_long`: beta=0.0435, SE=0.0092, z=4.74, p=2.095e-06
- experiment2 `preResp_long:same_transition`: beta=0.0417, SE=0.0126, z=3.32, p=0.0009064

All outputs are in `results/behavioral_models`.