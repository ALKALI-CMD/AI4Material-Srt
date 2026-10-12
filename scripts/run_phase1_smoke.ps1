$ErrorActionPreference = "Stop"

python -m pip install -e ".[dev]"
python -m pytest -q
atomlens smoke --calculator harmonic --steps 65
atomlens manifest --out results/manifest.json
atomlens attribute --calculator harmonic --structure-pair cspbi3_iodine_displacement --steps 65 --device cpu --out results/phase1_attribute_harmonic.json
atomlens convergence --calculator harmonic --structure-pair cspbi3_iodine_displacement --steps 17 33 65 129 --device cpu --out results/phase1_convergence_harmonic.json
atomlens validate-results --attribution results/phase1_attribute_harmonic.json --convergence results/phase1_convergence_harmonic.json --out results/phase1_validation_harmonic.json
atomlens plot-attribution --input results/phase1_attribute_harmonic.json --out results/phase1_attribute_harmonic.png
atomlens plot-convergence --input results/phase1_convergence_harmonic.json --out results/phase1_convergence_harmonic.png
atomlens export-structure --attribution results/phase1_attribute_harmonic.json --structure-pair cspbi3_iodine_displacement --which end --out results/phase1_attribute_harmonic.extxyz
atomlens report --attribution results/phase1_attribute_harmonic.json --convergence results/phase1_convergence_harmonic.json --attribution-figure results/phase1_attribute_harmonic.png --convergence-figure results/phase1_convergence_harmonic.png --structure-file results/phase1_attribute_harmonic.extxyz --out results/phase1_report.md
atomlens compare-attributions --left results/phase1_attribute_harmonic.json --right results/phase1_attribute_harmonic.json --out results/phase1_compare_self.json
atomlens plot-comparison --input results/phase1_compare_self.json --out results/phase1_compare_self.png
