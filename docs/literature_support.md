# Literature and Repository Support

This project starts from four supported building blocks.

## Pretrained Force Fields

- MACE: higher-order equivariant message passing with tensor product structure. Paper: https://arxiv.org/abs/2206.07697. Repository: https://github.com/ACEsuit/mace.
- CHGNet: pretrained universal neural network potential for charge-informed crystalline materials. Paper: https://www.nature.com/articles/s42256-023-00716-3. Repository: https://github.com/CederGroupHub/chgnet.
- NequIP: E(3)-equivariant neural network interatomic potentials, useful as a conceptual baseline. Paper: https://www.nature.com/articles/s41467-022-29939-5. Repository: https://github.com/mir-group/nequip.

## Attribution Methods

- Integrated Gradients gives a path-integral attribution baseline with a completeness property. Paper: https://arxiv.org/abs/1703.01365.
- GNNExplainer and PGExplainer support the broader graph-explanation framing, but direct masking can break atomistic geometry and periodic edges. They are literature context before being used as baselines.

## Hamiltonian Extension

- DeepH-E3: E(3)-equivariant deep-learning electronic Hamiltonian. Paper: https://www.nature.com/articles/s41467-023-38468-8. Repository: https://github.com/Xiaoxun-Gong/DeepH-E3.
- HamGNN: graph neural network tooling for machine-learning Hamiltonians. Repository: https://github.com/QuantumLab-ZY/HamGNN.

## Working Constraints

- The first phase should compare model-output attribution, not claim a unique physical bond energy decomposition.
- MACE and CHGNet use different training targets and energy conventions. Use within-model energy differences under fixed composition when comparing attribution patterns.
- Version pins, checkpoint names, CUDA version, GPU count, and random seeds must be written to each result manifest.
