"""Structural and semantic validation for declarative KEPA workflows."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from .registry import REGISTRY, OperationSpec


@dataclass(frozen=True)
class ValueInfo:
    type_name: str
    view: str | None = None
    structures: frozenset[str] = frozenset()


STEP_REF = re.compile(r"^\$steps\.([A-Za-z0-9_-]+)\.([A-Za-z0-9_-]+)$")
VERTEBRA_REGION = re.compile(r"^[CTLS]\d+$")
DISC_REGION = re.compile(r"^[CTLS]\d+/[CTLS]\d+$")
ANATOMICAL_REGION = re.compile(r"^[CTLS]\d+(?:/[CTLS]\d+)?$")
KNOWN_STRUCTURES = {
    "vertebrae",
    "intervertebral_regions",
    "intervertebral_discs",
    "spinal_canal",
    "spinal_cord",
    "neural_foramen",
}
MODEL_CAPABILITIES = {
    ("sct_cli", "spinalcord"): {"spinal_cord"},
    ("sct_cli", "totalspineseg"): {"vertebrae", "intervertebral_discs", "spinal_canal"},
    ("totalspineseg_cli", "totalspineseg"): {"vertebrae", "intervertebral_discs", "spinal_canal"},
    ("totalspineseg_python", "totalspineseg"): {"vertebrae", "intervertebral_discs", "spinal_canal"},
    ("axial_nnunet_python", "canal_seg"): {"spinal_canal"},
    ("axial_nnunet_python", "sc_epi"): {"spinal_cord"},
    ("totalspineseg", "totalspineseg"): {"vertebrae", "intervertebral_discs", "spinal_canal"},
    ("local_unet", "cervical_vertebra_unet"): {"vertebrae"},
}

SEGMENTATION_TOOLS = {
    "TotalSpineSeg": {
        "view": "sagittal_t2",
        "backend": "totalspineseg_python",
        "model_id": "totalspineseg",
    },
    "AxialCanalSeg": {
        "view": "axial_t2",
        "backend": "axial_nnunet_python",
        "model_id": "canal_seg",
    },
}


def normalize_workflow(document: Any) -> Any:
    """Translate the readable workflow surface syntax into the typed execution form.

    Legacy workflows using ``exam``/``query`` inputs and generic tool/operation
    pairs pass through unchanged. The readable form binds the ``sag`` and
    ``axl`` input slots to workflow-local names, which steps can reference
    directly without a ``$inputs.`` prefix.
    """
    if not isinstance(document, dict) or not isinstance(document.get("inputs"), dict):
        return document
    if document["inputs"] == {"exam": "MRIExam", "query": "DiagnosticQuery"}:
        # Keep the established workflow JSON backward-compatible while making
        # the runtime consume geometry from the mask artifact itself.
        normalized = deepcopy(document)
        for step in normalized.get("steps", []):
            if step.get("tool") == "segmentation" and step.get("operation") == "postprocess":
                step.get("inputs", {}).pop("series", None)
            if step.get("tool") == "slice_selection" and step.get("operation") == "select_sagittal_reference":
                step.get("options", {}).pop("view", None)
                step.get("options", {}).pop("structure", None)
        return normalized

    bindings: dict[str, str] = {}
    input_slots = {"sag": "sagittal_t2", "axl": "axial_t2"}
    for slot, view in input_slots.items():
        alias = document["inputs"].get(slot)
        if alias is None:
            continue
        if not isinstance(alias, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", alias):
            raise ValueError(f"input alias for {slot!r} must be a non-empty identifier or null")
        if alias in bindings:
            raise ValueError(f"duplicate workflow input alias {alias!r}")
        bindings[alias] = view

    unknown_slots = set(document["inputs"]) - set(input_slots)
    if unknown_slots:
        raise ValueError(f"unknown input slots: {sorted(unknown_slots)}")

    normalized = deepcopy(document)
    normalized["inputs"] = {"exam": "MRIExam", "query": "DiagnosticQuery"}
    requested_regions = document.get("requested_regions", {})
    if not isinstance(requested_regions, dict):
        raise ValueError("requested_regions must map input aliases to arrays of anatomical regions")
    normalized_regions: dict[str, list[str]] = {}
    for alias, regions in requested_regions.items():
        if alias not in bindings:
            raise ValueError(f"requested_regions key {alias!r} is not a bound workflow input alias")
        if not isinstance(regions, list) or not regions:
            raise ValueError(f"requested_regions[{alias!r}] must be a non-empty array")
        if any(not isinstance(region, str) or not ANATOMICAL_REGION.fullmatch(region) for region in regions):
            raise ValueError(f"requested_regions[{alias!r}] contains an invalid anatomical region name")
        if len(set(regions)) != len(regions):
            raise ValueError(f"requested_regions[{alias!r}] contains duplicate regions")
        view = bindings[alias]
        if view in normalized_regions:
            raise ValueError(f"requested_regions declares {view!r} more than once through input aliases")
        normalized_regions[view] = list(regions)
    normalized["requested_regions"] = normalized_regions
    source_refs = {
        alias: f"$input.exam.series.{view}"
        for alias, view in bindings.items()
    }
    label_maps: dict[str, dict[str, Any]] = {}
    named_outputs: dict[str, str] = {}
    named_output_views: dict[str, str | None] = {}

    def flatten_label_group(value: Any) -> list[int]:
        if isinstance(value, int) and not isinstance(value, bool):
            return [value]
        if isinstance(value, list):
            return [int(item) for item in value]
        if isinstance(value, dict):
            return [int(item) for item in value.values()]
        raise ValueError("label group must contain integer label IDs")

    def resolve_label_groups(view: str, groups: Any, step_id: str) -> list[int]:
        mapping = label_maps.get(view)
        if mapping is None:
            raise ValueError(f"step {step_id!r}: no segmentation label map is declared for {view}")
        names = [groups] if isinstance(groups, str) else groups
        if not isinstance(names, list) or not names or any(not isinstance(name, str) for name in names):
            raise ValueError(f"step {step_id!r}: labels must be a name or non-empty list of names")
        direct: dict[str, list[int]] = {}
        derived: dict[str, list[int]] = {
            "vertebral_bodies": [], "vertebrae": [],
            "intervertebral_discs": [], "discs": [],
        }
        for key, value in mapping.items():
            if key in {"vertebral_bodies", "vertebrae", "intervertebral_discs", "discs"} and isinstance(value, dict):
                group_ids = flatten_label_group(value)
                derived[key].extend(group_ids)
                if key in {"vertebral_bodies", "vertebrae"}:
                    derived["vertebral_bodies"].extend(group_ids)
                    derived["vertebrae"].extend(group_ids)
                else:
                    derived["intervertebral_discs"].extend(group_ids)
                    derived["discs"].extend(group_ids)
                for region, label_value in value.items():
                    direct[region] = flatten_label_group(label_value)
            else:
                ids = flatten_label_group(value)
                direct[key] = ids
                if VERTEBRA_REGION.fullmatch(key):
                    derived["vertebral_bodies"].extend(ids)
                    derived["vertebrae"].extend(ids)
                elif DISC_REGION.fullmatch(key):
                    derived["intervertebral_discs"].extend(ids)
                    derived["discs"].extend(ids)
        known = set(direct) | {key for key, ids in derived.items() if ids}
        unknown = set(names) - known
        if unknown:
            raise ValueError(f"step {step_id!r}: unknown labels {sorted(unknown)} for {view}")
        return sorted({label_id for name in names for label_id in direct.get(name, derived.get(name, []))})

    for step in normalized.get("steps", []):
        tool = step.get("tool")
        if tool in SEGMENTATION_TOOLS:
            if "operation" in step or "inputs" in step:
                raise ValueError(f"step {step.get('id')!r}: {tool} uses 'input' and 'labels'")
            tool_spec = SEGMENTATION_TOOLS[tool]
            alias = step.get("input")
            if alias not in bindings:
                raise ValueError(
                    f"step {step.get('id')!r}: input {alias!r} is not bound by workflow inputs"
                )
            if bindings[alias] != tool_spec["view"]:
                raise ValueError(
                    f"step {step.get('id')!r}: {tool} requires the {tool_spec['view']} input, "
                    f"but {alias!r} is bound to {bindings[alias]}"
                )
            labels = step.pop("labels", None)
            if not isinstance(labels, dict) or not labels:
                raise ValueError(f"step {step.get('id')!r}: {tool} requires a non-empty 'labels' map")
            for group, value in labels.items():
                if isinstance(value, dict):
                    if any(not isinstance(level, str) or not level for level in value):
                        raise ValueError(f"step {step.get('id')!r}: labels for {group!r} require named levels")
                    id_values = list(value.values())
                elif isinstance(value, list):
                    id_values = value
                elif isinstance(value, int) and not isinstance(value, bool):
                    id_values = [value]
                else:
                    raise ValueError(
                        f"step {step.get('id')!r}: labels for {group!r} must be a positive ID, "
                        "a list of positive IDs, or a level-to-ID map"
                    )
                if not id_values or any(
                    isinstance(item, bool) or not isinstance(item, int) or item <= 0
                    for item in id_values
                ):
                    raise ValueError(f"step {step.get('id')!r}: labels for {group!r} contain invalid IDs")
            structure_aliases = {
                "vertebral_bodies": "vertebrae",
                "vertebrae": "vertebrae",
                "intervertebral_discs": "intervertebral_discs",
                "discs": "intervertebral_discs",
                "spinal_canal": "spinal_canal",
                "spinal_cord": "spinal_cord",
                "neural_foramen": "neural_foramen",
            }
            structures_found: set[str] = set()
            unknown_labels: set[str] = set()
            for label_name, label_value in labels.items():
                if label_name in structure_aliases:
                    structures_found.add(structure_aliases[label_name])
                elif VERTEBRA_REGION.fullmatch(label_name):
                    structures_found.add("vertebrae")
                elif DISC_REGION.fullmatch(label_name):
                    structures_found.add("intervertebral_discs")
                else:
                    unknown_labels.add(label_name)
            structures = sorted(structures_found)
            if unknown_labels:
                raise ValueError(
                    f"step {step.get('id')!r}: unknown semantic label groups {sorted(unknown_labels)}"
                )
            label_maps[tool_spec["view"]] = labels
            step["tool"] = "segmentation"
            step["operation"] = "segment"
            step["inputs"] = {"series": source_refs[alias]}
            step["options"] = {
                "view": tool_spec["view"],
                "backend": tool_spec["backend"],
                "model_id": tool_spec["model_id"],
                "structures": structures,
                "label_map": labels,
            }
            step.pop("input", None)
            output_alias = step.pop("output", None)
            if output_alias is not None:
                if not isinstance(output_alias, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", output_alias):
                    raise ValueError(f"step {step.get('id')!r}: output must be a non-empty identifier")
                if output_alias in named_outputs:
                    raise ValueError(f"duplicate workflow output name {output_alias!r}")
                named_outputs[output_alias] = f"$steps.{step['id']}.mask"
                named_output_views[output_alias] = tool_spec["view"]
        else:
            step_inputs = step.get("inputs", {})
            if step.get("tool") == "roi_extraction":
                if "slice_selection" in step_inputs and "selection" in step_inputs:
                    raise ValueError(f"step {step.get('id')!r}: use only slice_selection")
                if "slice_selection" in step_inputs:
                    step_inputs["selection"] = step_inputs.pop("slice_selection")
            input_view = None
            if step.get("tool") == "segmentation" and step.get("operation") == "postprocess":
                # Post-processing consumes only the named mask artifact. Its
                # spatial metadata travels with that artifact, so repeating the
                # source image series is unnecessary.
                source_alias = step.pop("input", None)
                output_alias = step.pop("output", None)
                if step_inputs:
                    raise ValueError(
                        f"step {step.get('id')!r}: segmentation.postprocess uses 'input' and 'output'"
                    )
                if not isinstance(source_alias, str) or source_alias not in named_outputs:
                    raise ValueError(
                        f"step {step.get('id')!r}: input {source_alias!r} is not a previously named output"
                    )
                if not isinstance(output_alias, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", output_alias):
                    raise ValueError(f"step {step.get('id')!r}: output must be a non-empty identifier")
                if output_alias in named_outputs:
                    raise ValueError(f"duplicate workflow output name {output_alias!r}")
                input_view = named_output_views.get(source_alias)
                step["inputs"] = {"mask": named_outputs[source_alias]}
                step_inputs = step["inputs"]
                named_outputs[output_alias] = f"$steps.{step['id']}.mask"
                named_output_views[output_alias] = input_view

            # Bare workflow input aliases identify MRI series; named artifact
            # aliases identify outputs from prior steps. Legacy $steps refs are
            # still accepted internally and for backwards compatibility.
            series_refs = [
                ref for input_name, ref in step_inputs.items()
                if input_name in {"series", "source_series", "target_series"}
            ]
            input_view = input_view or next(
                (bindings[ref] for ref in series_refs if isinstance(ref, str) and ref in bindings),
                None,
            )
            for input_name, ref in step_inputs.items():
                if ref in source_refs:
                    step_inputs[input_name] = source_refs[ref]
                elif isinstance(ref, str) and ref in named_outputs:
                    step_inputs[input_name] = named_outputs[ref]

            options = step.get("options", {})
            if step.get("tool") == "roi_extraction":
                if "save_preview" in options and "render_overlay" in options:
                    raise ValueError(f"step {step.get('id')!r}: use only save_preview")
                if "save_preview" in options:
                    options["render_overlay"] = options.pop("save_preview")
            requested_view = None
            if step.get("tool") == "3d_matching":
                target_ref = step_inputs.get("target_series")
                requested_view = next((bindings[alias] for alias, ref in source_refs.items() if ref == target_ref), None)
            elif step.get("tool") == "roi_extraction":
                series_ref = step_inputs.get("series")
                requested_view = next((bindings[alias] for alias, ref in source_refs.items() if ref == series_ref), None)
            if requested_view in normalized_regions:
                options["requested_regions"] = normalized_regions[requested_view]
                if step.get("tool") == "3d_matching" and step.get("operation") == "match_anatomical_levels":
                    declared_map = label_maps.get("sagittal_t2", {})
                    options["region_label_ids"] = {
                        name: flatten_label_group(value)
                        for name, value in declared_map.items()
                        if ANATOMICAL_REGION.fullmatch(name)
                    }
            if step.get("tool") == "segmentation" and step.get("operation") == "postprocess":
                if "target_labels" in options:
                    options["target_label_ids"] = resolve_label_groups(
                        input_view, options.pop("target_labels"), step["id"]
                    )
                if "reference_labels" in options:
                    options["reference_label_ids"] = resolve_label_groups(
                        input_view, options.pop("reference_labels"), step["id"]
                    )
            elif step.get("tool") == "slice_selection" and step.get("operation") == "select_sagittal_reference":
                declared_view = options.pop("view", None)
                if declared_view is not None and input_view is not None and declared_view != input_view:
                    raise ValueError(
                        f"step {step.get('id')!r}: option view {declared_view!r} conflicts with input view {input_view!r}"
                    )
                declared_structure = options.pop("structure", None)
                if "labels" in options:
                    label_names = [options["labels"]] if isinstance(options["labels"], str) else options["labels"]
                    if declared_structure is not None and declared_structure not in label_names:
                        raise ValueError(
                            f"step {step.get('id')!r}: legacy structure must be included in labels"
                        )
                    options["label_ids"] = resolve_label_groups(input_view, options.pop("labels"), step["id"])
                output_alias = step.pop("output", None)
                if output_alias is not None:
                    if not isinstance(output_alias, str) or not re.fullmatch(
                        r"[A-Za-z][A-Za-z0-9_-]*", output_alias
                    ):
                        raise ValueError(f"step {step.get('id')!r}: output must be a non-empty identifier")
                    if output_alias in named_outputs or output_alias in bindings:
                        raise ValueError(f"duplicate workflow output name {output_alias!r}")
                    named_outputs[output_alias] = f"$steps.{step['id']}.slice"
                    named_output_views[output_alias] = input_view
            elif step.get("tool") == "slice_selection" and step.get("operation") == "select_sagittal_regions":
                regions = options.get("regions")
                if not isinstance(regions, dict) or not regions:
                    raise ValueError(f"step {step.get('id')!r}: regions must be a non-empty object")
                options["region_label_ids"] = {
                    region: resolve_label_groups(input_view, source_labels, step["id"])
                    for region, source_labels in regions.items()
                }
                options.pop("regions")
                output_alias = step.pop("output", None)
                if not isinstance(output_alias, str) or not re.fullmatch(
                    r"[A-Za-z][A-Za-z0-9_-]*", output_alias
                ):
                    raise ValueError(f"step {step.get('id')!r}: select_sagittal_regions requires an output name")
                if output_alias in named_outputs or output_alias in bindings:
                    raise ValueError(f"duplicate workflow output name {output_alias!r}")
                named_outputs[output_alias] = f"$steps.{step['id']}.slices"
                named_output_views[output_alias] = input_view
            elif step.get("tool") == "roi_extraction" and step.get("operation") in {"mask_bbox", "mask_centered_bbox"}:
                if "labels" in options:
                    options["label_ids"] = resolve_label_groups(input_view, options.pop("labels"), step["id"])

            output_alias = step.pop("output", None)
            if output_alias is not None:
                if not isinstance(output_alias, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", output_alias):
                    raise ValueError(f"step {step.get('id')!r}: output must be a non-empty identifier")
                if output_alias in named_outputs or output_alias in bindings:
                    raise ValueError(f"duplicate workflow output name {output_alias!r}")
                if step.get("tool") == "3d_matching" and step.get("operation") in {"match_anatomical_levels", "tra2sag_regions"}:
                    port = "matches"
                elif step.get("tool") == "roi_extraction":
                    port = "rois"
                else:
                    raise ValueError(f"step {step.get('id')!r}: this operation does not support a public output alias")
                named_outputs[output_alias] = f"$steps.{step['id']}.{port}"
                named_output_views[output_alias] = requested_view or input_view

    # Final workflow artifacts/evidence may also use the public output names.
    final_outputs = normalized.get("outputs", {})
    for output_kind in ("evidence", "artifacts", "results"):
        if output_kind in final_outputs:
            refs = final_outputs[output_kind]
            final_outputs[output_kind] = [named_outputs.get(ref, ref) for ref in refs]
    normalized["requested_regions"] = normalized_regions
    return normalized


def _schema() -> dict[str, Any]:
    resource = files("kepa").joinpath("workflow.schema.json")
    return json.loads(resource.read_text(encoding="utf-8"))


def _input_ref(ref: str) -> ValueInfo | None:
    if ref == "$input.exam":
        return ValueInfo("MRIExam")
    if ref == "$input.query":
        return ValueInfo("DiagnosticQuery")
    if ref == "$input.query.level":
        return ValueInfo("AnatomicalLevel")
    if ref == "$input.query.task":
        return ValueInfo("DiagnosticTask")
    match = re.fullmatch(r"\$input\.exam\.series\.(sagittal_t2|axial_t2)", ref)
    if match:
        return ValueInfo("MRIImageSeries", view=match.group(1))
    return None


def _validate_option_value(name: str, kind: str, value: Any, choices: dict[str, set[str]]) -> str | None:
    if kind in {"string", "view", "backend"}:
        if not isinstance(value, str) or not value:
            return f"option '{name}' must be a non-empty string"
        allowed = choices.get(name)
        if allowed is not None and value not in allowed:
            return f"option '{name}' must be one of {sorted(allowed)}, got {value!r}"
    elif kind == "strings":
        if not isinstance(value, list) or not value or any(not isinstance(x, str) or not x for x in value):
            return f"option '{name}' must be a non-empty array of strings"
    elif kind in {"positive_integer", "nonnegative_integer"}:
        minimum = 1 if kind == "positive_integer" else 0
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            return f"option '{name}' must be an integer >= {minimum}"
    elif kind == "positive_integers":
        if not isinstance(value, list) or not value or any(
            isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in value
        ):
            return f"option '{name}' must be a non-empty array of positive integers"
    elif kind == "number_pair":
        if not isinstance(value, list) or len(value) != 2 or any(
            isinstance(x, bool) or not isinstance(x, (int, float)) for x in value
        ):
            return f"option '{name}' must be an array of two numbers"
    elif kind == "positive_size_pair":
        if not isinstance(value, list) or len(value) != 2 or any(
            isinstance(x, bool) or not isinstance(x, int) or x <= 0 for x in value
        ):
            return f"option '{name}' must be a pair of positive integers [width, height]"
    elif kind == "positive_number":
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            return f"option '{name}' must be a positive number"
    elif kind == "nonnegative_number":
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            return f"option '{name}' must be a non-negative number"
    elif kind == "json_object":
        if not isinstance(value, dict):
            return f"option '{name}' must be an object"
    elif kind == "json_array":
        if not isinstance(value, list) or not value:
            return f"option '{name}' must be a non-empty array"
    elif kind == "boolean":
        if not isinstance(value, bool):
            return f"option '{name}' must be a boolean"
    return None


def validate_workflow(document: Any) -> list[str]:
    """Return all schema and cross-tool workflow errors; empty means valid."""
    errors: list[str] = []
    try:
        document = normalize_workflow(document)
    except (TypeError, ValueError) as exc:
        return [f"workflow input/tool configuration: {exc}"]
    schema_validator = Draft202012Validator(_schema())
    for error in sorted(schema_validator.iter_errors(document), key=lambda item: list(map(str, item.path))):
        location = ".".join(str(part) for part in error.absolute_path) or "$"
        errors.append(f"schema {location}: {error.message}")
    if errors:
        return errors

    if document["inputs"] != {"exam": "MRIExam", "query": "DiagnosticQuery"}:
        errors.append("inputs must declare exam: MRIExam and query: DiagnosticQuery")
        return errors

    produced: dict[str, dict[str, ValueInfo]] = {}
    id_to_step: dict[str, dict[str, Any]] = {}
    operation_count: dict[str, int] = {}

    def resolve(ref: str, current_index: int) -> ValueInfo | None:
        external = _input_ref(ref)
        if external is not None:
            return external
        match = STEP_REF.fullmatch(ref)
        if not match:
            errors.append(f"invalid reference {ref!r}; use $input... or $steps.<id>.<output>")
            return None
        step_id, output_name = match.groups()
        source_step = id_to_step.get(step_id)
        if source_step is None:
            errors.append(f"reference {ref!r} points to an unknown or later step; steps must be ordered")
            return None
        value = produced.get(step_id, {}).get(output_name)
        if value is None:
            errors.append(f"reference {ref!r} names an output that {step_id!r} does not produce")
        return value

    for index, step in enumerate(document["steps"]):
        step_id = step["id"]
        if step_id in id_to_step:
            errors.append(f"duplicate step id {step_id!r}")
            continue
        id_to_step[step_id] = step
        key = (step["tool"], step["operation"])
        spec: OperationSpec | None = REGISTRY.get(key)
        if spec is None:
            errors.append(f"step {step_id!r}: unsupported operation {key[0]}.{key[1]}")
            continue
        operation_count[step["tool"]] = operation_count.get(step["tool"], 0) + 1

        missing_inputs = set(spec.inputs) - set(step["inputs"])
        extra_inputs = set(step["inputs"]) - set(spec.inputs)
        for name in sorted(missing_inputs):
            errors.append(f"step {step_id!r}: missing input '{name}'")
        for name in sorted(extra_inputs):
            errors.append(f"step {step_id!r}: unknown input '{name}'")

        input_values: dict[str, ValueInfo] = {}
        for name, expected_type in spec.inputs.items():
            ref = step["inputs"].get(name)
            if not isinstance(ref, str):
                continue
            actual = resolve(ref, index)
            if actual is None:
                continue
            input_values[name] = actual
            compatible = actual.type_name == expected_type
            if name == "selection" and key[0] == "roi_extraction":
                compatible = actual.type_name in {"ImageSlice", "SliceMatchSet"}
                if key[1] in {"composite_bbox", "mask_centered_bbox"}:
                    compatible = compatible or actual.type_name == "SliceSelectionSet"
                if key[1] == "point_bbox":
                    compatible = actual.type_name == "SliceMatchSet" if step["options"].get("points_from_match") else actual.type_name == "ImageSlice"
            if not compatible:
                errors.append(
                    f"step {step_id!r}: input '{name}' expects {expected_type}, got {actual.type_name} from {ref!r}"
                )

        options = step["options"]
        allowed_option_names = set(spec.required_options) | set(spec.optional_options)
        for name in sorted(set(spec.required_options) - set(options)):
            errors.append(f"step {step_id!r}: missing option '{name}'")
        for name in sorted(set(options) - allowed_option_names):
            errors.append(f"step {step_id!r}: unknown option '{name}'")
        for name, kind in {**spec.required_options, **spec.optional_options}.items():
            if name not in options:
                continue
            option_error = _validate_option_value(name, kind, options[name], spec.choices)
            if option_error:
                errors.append(f"step {step_id!r}: {option_error}")

        output_values: dict[str, ValueInfo] = {}
        if key == ("segmentation", "segment"):
            series = input_values.get("series")
            view = options.get("view")
            structures = options.get("structures", [])
            if series and view != series.view:
                errors.append(f"step {step_id!r}: segmentation view {view!r} does not match series view {series.view!r}")
            valid_structures = isinstance(structures, list) and all(isinstance(x, str) for x in structures)
            requested_structures = set(structures) if valid_structures else set()
            unknown = requested_structures - KNOWN_STRUCTURES
            if unknown:
                errors.append(f"step {step_id!r}: unknown anatomical structures {sorted(unknown)}")
            model_id = options.get("model_id")
            capabilities = MODEL_CAPABILITIES.get((options.get("backend"), model_id))
            if capabilities is None:
                errors.append(
                    f"step {step_id!r}: model_id {model_id!r} is not registered for backend {options.get('backend')!r}"
                )
            elif not requested_structures.issubset(capabilities):
                unsupported = requested_structures - capabilities
                errors.append(f"step {step_id!r}: model {model_id!r} does not provide {sorted(unsupported)}")
            output_values["mask"] = ValueInfo("SegmentationMask", view, frozenset(requested_structures))

        elif key == ("segmentation", "postprocess"):
            mask = input_values.get("mask")
            method = options.get("method")
            if method == "anterior_largest_component":
                if mask and mask.view != "sagittal_t2":
                    errors.append(f"step {step_id!r}: anterior_largest_component requires sagittal_t2")
                if not options.get("target_label_ids") or not options.get("reference_label_ids"):
                    errors.append(
                        f"step {step_id!r}: anterior_largest_component requires target_labels and reference_labels"
                    )
            elif method == "center_nearest_largest_component":
                if mask and mask.view != "axial_t2":
                    errors.append(f"step {step_id!r}: center_nearest_largest_component requires axial_t2")
                if not options.get("target_label_ids"):
                    errors.append(f"step {step_id!r}: center_nearest_largest_component requires target_labels")
                fraction = options.get("minimum_area_fraction", 0.5)
                if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0 < fraction <= 1:
                    errors.append(f"step {step_id!r}: minimum_area_fraction must be in (0, 1]")
            output_values["mask"] = ValueInfo(
                "SegmentationMask", mask.view if mask else None,
                mask.structures if mask else frozenset(),
            )

        elif key == ("3d_matching", "match"):
            source = input_values.get("source_series")
            target = input_values.get("target_series")
            mask = input_values.get("source_mask")
            direction = options.get("direction")
            expected_pair = {
                "sagittal_to_axial": ("sagittal_t2", "axial_t2"),
                "axial_to_sagittal": ("axial_t2", "sagittal_t2"),
            }.get(direction)
            if source and target and expected_pair and (source.view, target.view) != expected_pair:
                errors.append(f"step {step_id!r}: direction {direction!r} conflicts with source/target series views")
            if mask and source and mask.view != source.view:
                errors.append(f"step {step_id!r}: source_mask and source_series must use the same view")
            output_values["match"] = ValueInfo("SliceMatch", target.view if target else None)

        elif key == ("3d_matching", "tra2sag_regions"):
            source = input_values.get("source_series")
            target = input_values.get("target_series")
            mask = input_values.get("source_mask")
            source_slices = input_values.get("source_slices")
            if source and target and (source.view, target.view) != ("axial_t2", "sagittal_t2"):
                errors.append(f"step {step_id!r}: tra2sag_regions requires axial_t2 -> sagittal_t2")
            if mask and source and mask.view != source.view:
                errors.append(f"step {step_id!r}: source_mask and source_series must use the same view")
            if source_slices and source and source_slices.view != source.view:
                errors.append(f"step {step_id!r}: source_slices must select source_series slices")
            point_rules = options.get("point_rules", [])
            rule_regions = []
            if not isinstance(point_rules, list) or not point_rules:
                errors.append(f"step {step_id!r}: point_rules must be a non-empty array")
            else:
                for rule_index, rule in enumerate(point_rules):
                    if not isinstance(rule, dict):
                        errors.append(f"step {step_id!r}: point_rules[{rule_index}] must be an object")
                        continue
                    regions = rule.get("regions")
                    if not isinstance(regions, list) or not regions or any(not isinstance(region, str) for region in regions):
                        errors.append(f"step {step_id!r}: point_rules[{rule_index}].regions must be a non-empty string array")
                    else:
                        rule_regions.extend(regions)
                    for fraction_name in ("alpha", "beta"):
                        fraction = rule.get(fraction_name)
                        if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0 <= fraction <= 1:
                            errors.append(f"step {step_id!r}: point_rules[{rule_index}].{fraction_name} must be in [0, 1]")
                    shift = rule.get("outward_shift")
                    if isinstance(shift, bool) or not isinstance(shift, int) or shift < 0:
                        errors.append(f"step {step_id!r}: point_rules[{rule_index}].outward_shift must be a non-negative integer")
            if len(rule_regions) != len(set(rule_regions)):
                errors.append(f"step {step_id!r}: point_rules assign a region more than once")
            requested = options.get("requested_regions", [])
            missing_rules = sorted(set(requested) - set(rule_regions))
            if missing_rules:
                errors.append(f"step {step_id!r}: requested regions have no point rule: {missing_rules}")
            output_values["matches"] = ValueInfo("SliceMatchSet", target.view if target else None)

        elif key == ("3d_matching", "match_anatomical_levels"):
            source = input_values.get("source_series")
            target = input_values.get("target_series")
            mask = input_values.get("source_mask")
            reference = input_values.get("reference_slice")
            if source and target and (source.view, target.view) != ("sagittal_t2", "axial_t2"):
                errors.append(f"step {step_id!r}: anatomical level matching requires sagittal_t2 -> axial_t2")
            if mask and source and mask.view != source.view:
                errors.append(f"step {step_id!r}: source_mask and source_series must use the same view")
            if reference and reference.view != "sagittal_t2":
                errors.append(f"step {step_id!r}: reference_slice must be sagittal_t2")
            if reference and source and reference.view != source.view:
                errors.append(f"step {step_id!r}: reference_slice and source_series must use the same view")
            requested = options.get("requested_regions", [])
            label_map = options.get("region_label_ids", {})
            if requested and isinstance(label_map, dict):
                missing_map_entries = sorted(set(requested) - set(label_map))
                if missing_map_entries:
                    errors.append(
                        f"step {step_id!r}: requested regions have no declared source label mapping: {missing_map_entries}"
                    )
            if options.get("method", "ratio") == "ratio":
                if isinstance(label_map, dict) and label_map:
                    missing_anatomy = sorted({
                        name
                        for region in requested
                        for name in (region.split("/") if "/" in region else [region])
                        if name not in label_map
                    })
                    if missing_anatomy:
                        errors.append(
                            f"step {step_id!r}: requested regions cannot be resolved from sagittal vertebra labels: {missing_anatomy}"
                        )
                else:
                    ratio_regions = {
                        *(f"C{level}" for level in range(3, 8)),
                        *(f"C{level}/C{level + 1}" for level in range(3, 7)),
                    }
                    unsupported = sorted(set(requested) - ratio_regions)
                    if unsupported:
                        errors.append(
                            f"step {step_id!r}: declare vertebra label aliases for non-cervical ratio matching: {unsupported}"
                        )
            output_values["matches"] = ValueInfo("SliceMatchSet", target.view if target else None)

        elif key == ("slice_selection", "select_sagittal_reference"):
            series = input_values.get("series")
            mask = input_values.get("mask")
            if mask and series and mask.view != series.view:
                errors.append(f"step {step_id!r}: mask and series must use the same view")
            if options.get("selector") == "max_label_union_area" and not options.get("label_ids"):
                errors.append(f"step {step_id!r}: max_label_union_area requires positive option 'label_ids'")
            output_values["slice"] = ValueInfo("ImageSlice", series.view if series else None)

        elif key == ("slice_selection", "select_sagittal_regions"):
            series = input_values.get("series")
            mask = input_values.get("mask")
            if series and series.view != "sagittal_t2":
                errors.append(f"step {step_id!r}: select_sagittal_regions requires sagittal_t2")
            if mask and series and mask.view != series.view:
                errors.append(f"step {step_id!r}: mask and series must use the same view")
            regions = options.get("region_label_ids")
            if not isinstance(regions, dict) or not regions:
                errors.append(f"step {step_id!r}: region_label_ids must be a non-empty object")
            elif any(not isinstance(ids, list) or not ids or any(
                isinstance(label_id, bool) or not isinstance(label_id, int) or label_id <= 0
                for label_id in ids
            ) for ids in regions.values()):
                errors.append(f"step {step_id!r}: every region must map to positive label IDs")
            output_values["slices"] = ValueInfo("SliceSelectionSet", series.view if series else None)

        elif key == ("slice_selection", "select_matched_slice"):
            series = input_values.get("series")
            match_value = input_values.get("match")
            view = options.get("view")
            if series and series.view != view:
                errors.append(f"step {step_id!r}: selected view does not match input series")
            if match_value and series and match_value.view != series.view:
                errors.append(f"step {step_id!r}: matching result targets {match_value.view!r}, not {series.view!r}")
            output_values["slice"] = ValueInfo("ImageSlice", series.view if series else view)

        elif key == ("slice_selection", "export_reference_slice"):
            series = input_values.get("series")
            selected = input_values.get("slice")
            view = options.get("view")
            if series and series.view != view:
                errors.append(f"step {step_id!r}: export view does not match input series")
            if selected and series and selected.view != series.view:
                errors.append(f"step {step_id!r}: selected slice and series must use the same view")
            output_values["evidence"] = ValueInfo("EvidenceItem", view)

        elif key == ("slice_selection", "export_matched_slices"):
            series = input_values.get("series")
            matches = input_values.get("matches")
            view = options.get("view")
            if series and series.view != view:
                errors.append(f"step {step_id!r}: export view does not match input series")
            if matches and series and matches.view != series.view:
                errors.append(f"step {step_id!r}: matched slices and series must use the same view")
            output_values["evidence"] = ValueInfo("EvidenceBundle", view)

        elif key == ("roi_extraction", "construct"):
            image = input_values.get("image")
            mask = input_values.get("mask")
            if image and mask and image.view != mask.view:
                errors.append(f"step {step_id!r}: ROI image and mask must use the same view")
            rule = options.get("rule")
            if rule in {"mask_bbox", "shifted_mask_bbox"}:
                structure = options.get("structure")
                if not structure:
                    errors.append(f"step {step_id!r}: rule {rule!r} requires option 'structure'")
                elif mask and structure not in mask.structures:
                    errors.append(f"step {step_id!r}: ROI structure {structure!r} is absent from the input mask")
            elif rule == "adjacent_structure_union":
                structures = options.get("structures", [])
                if not structures:
                    errors.append(f"step {step_id!r}: adjacent_structure_union requires option 'structures'")
                elif mask and isinstance(structures, list) and all(isinstance(x, str) for x in structures) and not set(structures).issubset(mask.structures):
                    errors.append(f"step {step_id!r}: union structures are not all present in the input mask")
            output_values["evidence"] = ValueInfo("EvidenceItem", image.view if image else None)

        elif key[0] == "roi_extraction" and key[1] in {"mask_bbox", "mask_centered_bbox", "composite_bbox", "point_bbox"}:
            series = input_values.get("series")
            mask = input_values.get("mask")
            selection = input_values.get("selection")
            if series and selection and selection.view != series.view:
                errors.append(f"step {step_id!r}: ROI selection and image series must use the same view")
            if mask and series and mask.view != series.view:
                errors.append(f"step {step_id!r}: ROI mask and image series must use the same view")
            if key[1] in {"mask_bbox", "composite_bbox", "mask_centered_bbox"} and mask is None:
                errors.append(f"step {step_id!r}: {key[1]} requires a segmentation mask")
            if key[1] == "point_bbox":
                if options.get("points_from_match"):
                    if not isinstance(options.get("points_from_match"), bool):
                        errors.append(f"step {step_id!r}: points_from_match must be boolean")
                    if selection and selection.type_name != "SliceMatchSet":
                        errors.append(f"step {step_id!r}: points_from_match requires a SliceMatchSet selection")
                elif not isinstance(options.get("points"), list) or not options.get("points"):
                    errors.append(f"step {step_id!r}: point_bbox requires points or points_from_match=true")
            view = series.view if series else (selection.view if selection else None)
            output_values["rois"] = ValueInfo("ROIBundle", view)
            output_values["evidence"] = ValueInfo("EvidenceBundle", view)

        produced[step_id] = output_values

    if len(document["query"]["required_views"]) > 1 and operation_count.get("3d_matching", 0) == 0:
        errors.append("multi-view query requires at least one 3d_matching step")

    output_refs = document["outputs"].get("evidence", [])
    output_views: set[str] = set()
    used_steps: set[str] = set()
    pending: list[str] = []
    for ref in output_refs:
        value = resolve(ref, len(document["steps"]))
        match = STEP_REF.fullmatch(ref)
        if match and match.group(1) in id_to_step:
            pending.append(match.group(1))
        if value is None:
            continue
        if value.type_name not in {"EvidenceItem", "EvidenceBundle"}:
            errors.append(
                f"workflow output {ref!r} must reference EvidenceItem or EvidenceBundle, got {value.type_name}"
            )
        elif value.view:
            output_views.add(value.view)
    for ref in document["outputs"].get("results", []):
        value = resolve(ref, len(document["steps"]))
        match = STEP_REF.fullmatch(ref)
        if match and match.group(1) in id_to_step:
            pending.append(match.group(1))
        if value is not None and value.type_name != "ROIBundle":
            errors.append(f"workflow result {ref!r} must reference ROIBundle, got {value.type_name}")
        elif value is not None and value.view:
            output_views.add(value.view)
    for ref in document["outputs"].get("artifacts", []):
        value = resolve(ref, len(document["steps"]))
        match = STEP_REF.fullmatch(ref)
        if match and match.group(1) in id_to_step:
            pending.append(match.group(1))
        if value is not None and value.type_name not in {"SegmentationMask", "ROIBundle", "SliceSelectionSet", "SliceMatchSet"}:
            errors.append(
                f"workflow artifact {ref!r} must reference SegmentationMask, ROIBundle, SliceSelectionSet, or SliceMatchSet, "
                f"got {value.type_name}"
            )
    missing_views = set(document["query"]["required_views"]) - output_views
    if missing_views:
        errors.append(f"workflow outputs do not include required views: {sorted(missing_views)}")

    while pending:
        step_id = pending.pop()
        if step_id in used_steps:
            continue
        used_steps.add(step_id)
        for ref in id_to_step[step_id]["inputs"].values():
            match = STEP_REF.fullmatch(ref)
            if match and match.group(1) in id_to_step:
                pending.append(match.group(1))
    disconnected = set(id_to_step) - used_steps
    if disconnected:
        errors.append(f"workflow contains steps that do not contribute to outputs: {sorted(disconnected)}")

    return errors
