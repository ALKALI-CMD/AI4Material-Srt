# Phase 1 Protocol

## Goal

Build a runnable attribution slice for pretrained atomic force fields:

1. verify calculator loading;
2. run energy and force predictions on simple periodic structures;
3. compute coordinate-path integrated force attribution;
4. check numerical completeness and symmetry behavior;
5. save a manifest for every local or Volcengine run.

## Attribution Definition

For fixed atom order, fixed composition, and fixed cell, define a path from `R0` to `R1`:

```text
R(alpha) = R0 + alpha * (R1 - R0), alpha in [0, 1]
```

The per-atom attribution is:

```text
A_i = - dR_i dot integral_0^1 F_i(R(alpha)) d alpha
```

The sum should approximate `E(R1) - E(R0)`. The residual is a numerical quadrature check:

```text
residual = sum_i A_i - (E1 - E0)
```

## Initial Structures

- `diamond_si`: model and environment smoke test.
- `cubic_cspbi3`: perovskite baseline.
- `cspbi3_iodine_displacement`: same atom count and same cell, used for attribution before vacancy or defect formation energies.

## Volcengine Run Policy

- Use the existing Volcengine GPU server after CLI login and resource discovery.
- One experiment process may use at most two GPUs.
- Independent sweeps can run in parallel if the server has idle GPUs.
- Start with one GPU for MACE/CHGNet inference and only request two GPUs for batched attribution throughput.

## Gates Before Edge-Level Claims

- Do not interpret message norms as physical bond energies without fidelity tests.
- Edge attribution requires periodic edge identities including lattice shifts.
- Vacancy cases need a separate formation-energy protocol with chemical potentials; they are not part of the first fixed-composition path.
