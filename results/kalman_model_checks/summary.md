# Kalman Model Validation

Winner models:
- Code Experiment 1 / displayed Experiment 2 fixed: `C1_S0_B2` / `C_Q1__S_baseline__B_B2`
- Code Experiment 2 / displayed Experiment 1 dynamic: `C1_S0_B2` / `C_Q1__S_baseline__B_B2`

## PPC subject-level CTI/SDI recovery
- Code Experiment 1 / displayed Experiment 2 fixed: CTI r=0.995, SDI r=0.949, mean RMSE=0.1996
- Code Experiment 2 / displayed Experiment 1 dynamic: CTI r=0.996, SDI r=0.751, mean RMSE=0.1825

## Behavioral effect recovery
- Exp1 dynamic High-Low (observed): mean diff=0.0406, SEM=0.0168
- Exp1 dynamic High-Low (predicted): mean diff=0.0031, SEM=0.0050
- Exp2 fixed Same-Switch (observed): mean diff=0.0404, SEM=0.0131
- Exp2 fixed Same-Switch (predicted): mean diff=0.0072, SEM=0.0036

All outputs are in `results/kalman_model_checks`.
