# Caption-generation clients

This folder contains the two caption-generation clients and sanitized prompt templates taken from the sagittal and axial generation prompts:

- `generate_caption.py` uses `sagittal_prompt_templates.json`.
- `generate_caption_tra.py` uses `axial_prompt_templates.json`.

The original prompt instructions and formatting are retained. The embedded knowledge-base material is blanked, and sample-specific input/label values are replaced by the `{{INPUT_DATA}}` placeholder. No source prompt records, patient identifiers, labels, generated captions, or datasets are included. Because the knowledge base is intentionally omitted, these public prompts are not the complete knowledge-guided method and are not expected to reproduce the full original caption behavior.

Each template JSON contains one or more variants. Use its `template_id` value in the request file. A request's `input_data` is inserted at the original `# Input Data` position. The scripts send the rendered prompt to the inference endpoint; an optional opaque `request_id` is used only to link the output locally and is never sent to the model.

## Inference service

The clients use the `InferClient` API from [ms-swift](https://github.com/modelscope/ms-swift). In the reference setup, ms-swift serves the model with vLLM as its inference backend. The clients do not start/configure that service and do not include checkpoints, model configuration, or deployment commands. Install a ms-swift version compatible with the separate service. This optional dependency is not part of KEPA's core requirements.

## Request format

UTF-8 JSONL, one object per line:

```json
{"template_id":"sag_01","request_id":"sample-0001","input_data":"Synthetic example input only."}
```

Use an axial template ID such as `axl_01` with the axial client. The template files list all available IDs. Do not place identifying information in `input_data`, `request_id`, logs, or outputs. Use synthetic or properly de-identified information and follow applicable institutional privacy rules.

## Run

Start the separately managed inference service, then run:

```bash
python caption_generation/generate_caption.py --input requests.jsonl --output sagittal_captions.jsonl --host 127.0.0.1 --port 8000
python caption_generation/generate_caption_tra.py --input requests.jsonl --output axial_captions.jsonl --host 127.0.0.1 --port 8000
```

Defaults match the original client settings: batch size 20, `max_tokens=512`, and `temperature=1.0`. Existing output files are preserved unless `--overwrite` is supplied; `--append` explicitly appends. Use `--validate-only` to validate IDs and render prompts without contacting the model service. Captions are research outputs and require appropriate review; the scripts do not assess clinical correctness.
