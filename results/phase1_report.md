# Atomlens Phase 1 Report

## Run Summary

- Calculator: `harmonic`
- Structure path: `cspbi3_iodine_displacement`
- Attribution steps: `65`
- Energy delta: `9.60000000e-03`
- Attribution total: `9.60000000e-03`
- Completeness residual: `-1.73472348e-18`

## Top Atomic Contributions

| rank | atom | attribution |
| ---: | --- | ---: |
| 1 | `I3` | `3.20000000e-03` |
| 2 | `I4` | `3.20000000e-03` |
| 3 | `I2` | `3.20000000e-03` |
| 4 | `Cs0` | `-0.00000000e+00` |
| 5 | `Pb1` | `-0.00000000e+00` |

## Convergence

| steps | energy delta | attribution total | residual | max abs atomic |
| ---: | ---: | ---: | ---: | ---: |
| 17 | `9.60000000e-03` | `9.60000000e-03` | `0.00000000e+00` | `3.20000000e-03` |
| 33 | `9.60000000e-03` | `9.60000000e-03` | `0.00000000e+00` | `3.20000000e-03` |
| 65 | `9.60000000e-03` | `9.60000000e-03` | `-1.73472348e-18` | `3.20000000e-03` |
| 129 | `9.60000000e-03` | `9.60000000e-03` | `1.73472348e-18` | `3.20000000e-03` |

## Figures

- Attribution plot: `results/phase1_attribute_harmonic.png`
- Convergence plot: `results/phase1_convergence_harmonic.png`
- Structure with attribution: `results/phase1_attribute_harmonic.extxyz`

## Environment

- `ase`: `3.29.0`
- `atomlens`: `0.1.0`
- `chgnet`: `None`
- `mace-torch`: `None`
- `numpy`: `2.4.2`
- `plotly`: `None`
- `scipy`: `1.18.1`
- `torch`: `2.13.0`
