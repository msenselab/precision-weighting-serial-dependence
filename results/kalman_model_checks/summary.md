# Kalman Model Validation

Winner models:
- Experiment 1: `C1_S0_B2` / `C_Q1__S_baseline__B_B2`
- Experiment 2: `C1_S1_B2` / `C_Q1__S_x_reset__B_B2`

## PPC subject-level CTI/SDI recovery
- Experiment 1: CTI r=0.993, SDI r=0.806, mean RMSE=0.1850
- Experiment 2: CTI r=0.995, SDI r=0.934, mean RMSE=0.1984

## Behavioral effect recovery
- Experiment 1 High-Low (observed): mean diff=0.0406, SEM=0.0168
- Experiment 1 High-Low (predicted): mean diff=0.0057, SEM=0.0042
- Experiment 2 Same-Switch (observed): mean diff=0.0404, SEM=0.0131
- Experiment 2 Same-Switch (predicted): mean diff=0.0107, SEM=0.0047

All outputs are in `results/kalman_model_checks`.
