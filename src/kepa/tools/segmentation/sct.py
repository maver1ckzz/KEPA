"""SCT command-line adapter for TotalSpineSeg."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TotalSpineSegBackend:
    executable: str = "sct_deepseg"

    def run(self, input_nii: Path, output_dir: Path) -> tuple[Path, list[str]]:
        """Run SCT once and return its combined, instance-labeled NIfTI output."""

        input_nii = input_nii.absolute()
        output_dir = output_dir.absolute()
        if not input_nii.is_file():
            raise FileNotFoundError(f"input NIfTI does not exist: {input_nii}")
        executable = shutil.which(self.executable) or (
            self.executable if Path(self.executable).is_file() else None
        )
        if executable is None:
            raise FileNotFoundError(
                f"cannot find SCT executable {self.executable!r}; pass --sct-deepseg or add it to PATH"
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        # SCT 7.1 expects a NIfTI output path (including the extension). Passing
        # a bare prefix can leave the CLI reporting success without saving the
        # final combined label map.
        output_base = output_dir / "totalspineseg.nii.gz"
        command = [
            str(executable),
            "totalspineseg",
            "-i",
            str(input_nii),
            "-o",
            str(output_base),
            "-r",
            "1",
        ]
        completed = subprocess.run(
            command,
            cwd=output_dir,
            check=False,
            capture_output=True,
            text=True,
        )
        (output_dir / "totalspineseg.stdout.log").write_text(completed.stdout, encoding="utf-8")
        (output_dir / "totalspineseg.stderr.log").write_text(completed.stderr, encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(
                "SCT TotalSpineSeg failed with exit code "
                f"{completed.returncode}. See {output_dir / 'totalspineseg.stderr.log'}"
            )
        candidates = sorted(output_dir.glob("*_step2_output.nii.gz"))
        if not candidates:
            candidates = sorted(output_dir.glob("*_totalspineseg_all.nii.gz"))
        if not candidates:
            raise RuntimeError(
                "could not find SCT combined labels (*_step2_output.nii.gz or "
                f"*_totalspineseg_all.nii.gz) in {output_dir}"
            )
        if len(candidates) > 1:
            raise RuntimeError(f"found multiple SCT combined label files in {output_dir}: {candidates}")
        return candidates[0], command
