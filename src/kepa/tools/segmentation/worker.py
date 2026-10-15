"""Line-delimited worker protocol for a persistent model runtime.

Run with the Python that has the model dependencies installed. This process
loads a registered backend once and accepts sequential segmentation requests.
Only response lines beginning with KEPA_RESPONSE are part of the protocol;
third-party diagnostic stdout is ignored by the client.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from .contract import SegmentationRequest
from .factory import create_backend


PREFIX = "KEPA_RESPONSE\t"


def _reply(value: dict) -> None:
    sys.stdout.write(PREFIX + json.dumps(value, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-workers", type=int, default=1)
    args = parser.parse_args()
    try:
        backend = create_backend(
            args.backend, data_dir=args.data_dir, device=args.device,
            max_workers=args.max_workers,
        )
        backend.start()
        preflight = getattr(backend, "preflight_device", None)
        if callable(preflight):
            preflight()
    except Exception as exc:
        _reply({"ok": False, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
        return 2
    _reply({"ok": True, "ready": True, "backend_id": backend.backend_id,
            "keeps_model_resident": backend.keeps_model_resident})
    try:
        for line in sys.stdin:
            try:
                message = json.loads(line)
                if message.get("action") == "close":
                    _reply({"ok": True, "closed": True})
                    break
                if message.get("action") != "segment":
                    raise ValueError("unsupported worker action")
                payload = message["request"]
                result = backend.segment(SegmentationRequest(
                    input_image=Path(payload["input_image"]),
                    output_dir=Path(payload["output_dir"]),
                    view=payload["view"], structures=tuple(payload["structures"]),
                    model_id=payload["model_id"], case_id=payload.get("case_id"),
                    options=payload.get("options", {}),
                ))
                _reply({"ok": True, "result": {
                    "label_map": str(result.label_map), "view": result.view,
                    "model_id": result.model_id, "backend_id": result.backend_id,
                    "backend_version": result.backend_version,
                    "geometry": {
                        "shape_ijk": list(result.geometry.shape_ijk),
                        "voxel_to_world": [list(row) for row in result.geometry.voxel_to_world],
                        "coordinate_system": result.geometry.coordinate_system,
                    },
                    "label_map_semantics": result.label_map_semantics,
                    "command": list(result.command),
                    "log_paths": [str(path) for path in result.log_paths],
                    "metadata": result.metadata,
                }})
            except Exception as exc:
                _reply({"ok": False, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
    finally:
        backend.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
