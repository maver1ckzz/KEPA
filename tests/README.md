# Tests and contribution notes

Tests use synthetic NIfTI volumes or small metadata fixtures. Do not add patient images, names, identifiers, private paths, weights, or real run logs to this directory.

Run the suite from the repository root:

```bash
python -m pip install -e . pytest
python -m pytest -q
kepa workflow validate configs/workflows/cervical_scs_dual_view.json
kepa workflow validate configs/workflows/rsna_lumbar_scs_soft_distance.json
```

The geometry tests cover coordinate transforms and oblique planes. Paired-runner tests cover workflow validation, typed connections, generated artifacts, manifest serialization, no-overwrite run IDs, and a de-identified slice-index GT regression fixture. The latter stores indices only; it is a software regression fixture, not a clinical performance estimate.

When contributing an operation, add descriptor/option validation, synthetic input/output tests, error-path tests, provenance checks and a workflow integration test. Validate that writes remain under the supplied temporary/run root. See the [tool contract](../src/kepa/tools/README.md).
