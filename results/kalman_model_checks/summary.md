# Reprocessed Kalman Validation

Winner model: `C1_S0_B2` / `C_Q1__S_baseline__B_B2`

## PPC subject-level CTI/SDI recovery
- Code Exp 1 / manuscript Exp 2 fixed: CTI r=0.995, SDI r=0.949, mean RMSE=0.1996
- Code Exp 2 / manuscript Exp 1 dynamic: CTI r=0.996, SDI r=0.751, mean RMSE=0.1825

## Behavioral effect recovery
- Exp1 dynamic High-Low (observed): mean diff=0.0406, SEM=0.0168
- Exp1 dynamic High-Low (predicted): mean diff=0.0031, SEM=0.0050
- Exp2 fixed Same-Switch (observed): mean diff=0.0404, SEM=0.0131
- Exp2 fixed Same-Switch (predicted): mean diff=0.0072, SEM=0.0036

## Parameter recovery (5 simulations per subject)
- Code Exp 1 / manuscript Exp 2 fixed: q1: r=0.49, q2: r=0.35, q3: r=0.49, lambda: r=0.26, r_base: r=0.46, d0: r=0.93, alpha_d0: r=0.84, alpha_q1: r=0.65
- Code Exp 2 / manuscript Exp 1 dynamic: q1: r=0.34, q2: r=0.11, q3: r=0.33, lambda: r=0.08, r_base: r=0.48, d0: r=0.96, alpha_d0: r=0.93, alpha_q1: r=0.71

All outputs are in `preprocessing_recheck_20260527/kalman_validation_outputs`.
