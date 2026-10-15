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
- The axial adapter in `src/kepa/tools/segmentation/axial_nnunet.py` is a small KEPA implementation of the model loading, orientation conversion and prediction behavior studied in [Spinal Cord Toolbox](https://github.com/spinalcordtoolbox/spinalcordtoolbox) 7.1 (`spinalcordtoolbox/deepseg/nnunet.py`, `spinalcordtoolbox/deepseg/inference.py`, and its image-orientation helpers). It calls the separately installed nnU-Net v2 package directly and **does not require SCT at runtime**. SCT is distributed under GNU LGPL-3.0; preserve this attribution and review derivative-work obligations before publication. No entire SCT source module is bundled.
- Neither external checkpoint ZIP nor its extracted weights are included in the KEPA source repository. Users should obtain exact releases from their official repositories and verify asset-specific permissions before redistribution.

This notice is an engineering inventory, not legal advice. Re-check the exact source and checkpoint terms for the pinned upstream release before publishing or redistributing a combined product.
