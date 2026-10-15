# Third-party software and model assets

## TotalSpineSeg

- Upstream: <https://github.com/neuropoly/totalspineseg>
- Upstream code repository identifies its source code as GNU LGPL v3.0.
- KEPA provides an independently written CLI adapter and a resident Python adapter in `src/kepa/tools/segmentation/`. The latter imports the separately installed TotalSpineSeg and nnU-Net Python packages and reuses their preprocessing/postprocessing. KEPA does not copy or bundle upstream source files.
- Upstream pretrained nnU-Net checkpoints are separate downloaded assets. The public repository's source-code license alone is not treated here as an explicit checkpoint redistribution grant. KEPA does not bundle or mirror checkpoints; users must obtain them from the official TotalSpineSeg distribution and comply with any asset-specific terms.
- Upstream installation, model setup, and citation instructions: <https://github.com/neuropoly/totalspineseg#installation> and <https://github.com/neuropoly/totalspineseg#inference>.
- The KEPA adapter preflights both checkpoint sets and passes the model-data root directly to TotalSpineSeg, which supports versioned release folders. No checkpoint files are copied, staged, or modified by KEPA.

## Axial canal and spinal-cord models

- [model-canal-seg](https://github.com/ivadomed/model-canal-seg): upstream repository identifies its source license as MIT. KEPA tested its `r20241126` T2w checkpoint on one axial case. A newer contrast-agnostic release exists; KEPA does not silently substitute it for the pinned test release.
- [fmri-segmentation / EPISeg](https://github.com/sct-pipeline/fmri-segmentation): upstream repository identifies its source license as MIT. KEPA tested the `v0.2` checkpoint as an **out-of-domain experiment** on anatomical axial T2; the model was developed for gradient-echo EPI/fMRI.
- The axial adapter in `src/kepa/tools/segmentation/axial_nnunet.py` adapts the model-loading, orientation-conversion, and prediction workflow described by Spinal Cord Toolbox (SCT) 7.1, particularly `spinalcordtoolbox/deepseg/nnunet.py`, `spinalcordtoolbox/deepseg/inference.py`, and SCT image-orientation helpers. It calls the separately installed nnU-Net v2 package directly and **does not require SCT at runtime**. No SCT source file or complete SCT module is included. Because this adapter is being distributed as an adaptation of SCT behavior, KEPA marks this file as LGPL-3.0-only; see its SPDX/modification header and [`LICENSE-LGPL-3.0-only`](LICENSE-LGPL-3.0-only). SCT is distributed under GNU LGPL-3.0; this attribution does not imply endorsement by the SCT project.
- Neither external checkpoint ZIP nor its extracted weights are included in the KEPA source repository. Users should obtain exact releases from their official repositories and verify asset-specific permissions before redistribution.

This notice is an engineering inventory, not legal advice. Re-check the exact source and checkpoint terms for the pinned upstream release before redistributing software or model assets.
