# Tool author contract

This document is the common authoring guide for KEPA-compatible algorithms. The stable concept is a typed operation with explicit ports and provenance. The current repository is still a prototype: the operation catalog and executor are static, and the Python protocol below is a target contract rather than an importable public SDK today.

## Responsibilities

A tool operation owns one well-defined transformation, for example “segment this series” or “rank target planes for this anatomy.” It should not parse an entire workflow, choose unrelated downstream tools, or write into another step's directory.

### Inputs

Each input port declares:

- artifact type and `schema_version`;
- whether it is required or optional;
- view and coordinate-space expectations;
- geometry/grid requirements and whether resampling is allowed;
- label-map semantics if a segmentation is consumed.

Use artifact handles rather than bare arrays in the public API. An artifact envelope should identify its URI, content hash, media type, geometry, coordinate convention, and semantic metadata. Large NIfTI/mask arrays belong in files, not inside workflow JSON.

### Options

Options must be JSON-serializable and described by a JSON Schema. Include units for physical values (`mm`, degrees, seconds), valid ranges, defaults and ties/fallback behavior. A tool must reject unsupported or ambiguous options rather than silently ignore them.

### Outputs

Return exactly the declared output ports. Write files beneath `context.step_dir`; return paths relative to the run or registered artifact URIs. Preserve source series and transforms so downstream tools can determine which image grid the artifact belongs to. Do not mutate source inputs.

### Provenance and errors

Record implementation/version, parameters, input/output SHA-256, model/checkpoint identity and license/source where applicable, coordinate conversions, elapsed time, warnings, and failure reason. Errors should identify the operation and invalid port; do not replace missing masks or failed matching with an arbitrary result.

## Target operation protocol

The future SDK is intended to provide an `OperationDescriptor`, typed `ArtifactPort`s, `ToolContext`, logging, artifact writers, and a registry. Conceptually:

```python
class MyOperation:
    operation_id = "organization.package.operation"

    def describe(self):
        return OperationDescriptor(
            inputs={"series": ArtifactPort("MRIImageSeries", "1.0")},
            outputs={"mask": ArtifactPort("SegmentationMask", "1.0")},
            options_schema={"type": "object", "properties": {}},
        )

    def execute(self, inputs, options, context):
        # inputs are resolved artifact objects; context.step_dir is isolated.
        return {"mask": context.artifacts.write_nifti(...)}
```

This snippet is intentionally illustrative; these SDK names are not yet exported by `kepa`. Until the plugin API is implemented, a new operation must be added to the static registry/validator and executor dispatch, and to the workflow schema if it introduces a new tool family.

## Third-party tools and namespacing

The public plugin design should permit workflows made entirely from third-party tools. Use a reverse-DNS or organization-qualified ID (`org_name.tool_name.operation_name`) for external catalog entries to avoid collisions. A future package should publish its descriptor and runtime implementation together through a Python entry point; validation must be able to inspect descriptors without running the tool. The executor must resolve only explicitly installed/allow-listed plugins and should not execute arbitrary module paths from workflow JSON.

An external operation is compatible when its port types and schema versions match the workflow connections—not because its implementation belongs to one of KEPA's current four tool families. A third-party tool may define a new artifact type, but downstream consumers must explicitly declare support for that type/schema.

## Required tests for a contributed tool

1. Descriptor/schema tests: required ports, option rejection, version and capabilities.
2. Synthetic execution test: deterministic output and no writes outside the supplied step directory.
3. Geometry tests: affine, coordinate convention, view, orientation, spacing and resampling where relevant.
4. Provenance test: output artifact hash and source/input linkage are recorded.
5. Workflow integration test: a small JSON workflow validates and passes the output to a consumer.
6. Failure test: empty/invalid data produces a clear failed step, not a plausible-looking result.

See the family guides: [segmentation](segmentation/README.md), [matching](matching/README.md), [slice selection](slice_selection/README.md), and [ROI extraction](roi_extraction/README.md).
