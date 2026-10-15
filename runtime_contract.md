# Workflow 输入/输出契约与运行目录规则

状态：paired sagittal/axial NIfTI 执行器 v0.1。此文档描述当前已实现并回归验证的行为；不代表所有 registry operation 均已接入运行时。

## 1. 执行输入

paired NIfTI runner 接收四项必需输入：

| 输入 | 契约 |
|---|---|
| `--sag-nii` | 3D sagittal NIfTI；header affine 必须有效且数据非空。 |
| --ax-nii | One or more 3D axial NIfTIs from the same exam and patient coordinate frame; each volume retains its own affine and is never concatenated or assumed to share a voxel grid. |
| `--workflow` | 符合 `workflow.schema.json` 并通过 cross-tool validator 的 JSON 工作流。 |
| `--runs-root` | 运行产物根目录；runner 在其下新建唯一 case/run 子目录。 |

可选项：`--case-id`、`--run-id`、`--sct-deepseg`。`--reuse-sag-segmentation` 是显式回归/复用选项：将既有 sagittal 标签图复制进本次 run，并跳过 SCT；生产执行默认不提供此项，即运行配置声明的分割 tool。

NIfTI affine 被当作 voxel-to-RAS+。与 DICOM `ImagePositionPatient` 等 LPS 坐标直接比较前，必须应用 `diag(-1,-1,1)` 的 RAS/LPS 变换。方向轴由 affine 推断，并将推断轴及偏离标准视图角度记入 manifest；不静默重排或覆盖原始影像。

When multiple axial paths are supplied, the runner executes the workflow independently for each NIfTI, reuses the sagittal segmentation, and keeps resident segmentation backends alive across candidate volumes. Per-region matches are then compared across series. The final result records source_series_id, source_nifti, and the series-local slice index. Equal best evidence is marked ambiguous/null instead of being resolved by input order; each candidate run and match diagnostic is retained under candidates/. The workflow must declare any axial segmentation step to have every input NIfTI segmented.

## 2. 工具数据类型与接口

工具只通过 `$input...` 或 `$steps.<step-id>.<output>` 引用传递产物，不在 workflow JSON 中内嵌影像或 mask 数组。

| 类型 | 当前运行时内容 |
|---|---|
| `MRIImageSeries` | 3D 数据、源文件路径、voxel-to-RAS affine、view、推断切片轴。 |
| `SegmentationMask` | SCT instance-label NIfTI，重采样到输入 series 网格（必要时最近邻）；当前 SCT adapter。 |
| `ImageSlice` | source series + slice index + selector 信息。 |
| `SliceMatchSet` | level -> axial NIfTI index、匹配算法、score/details、切片平面几何。 |
| `EvidenceItem` | 一张导出的 sagittal PNG 及 view/index/plane/selection provenance。 |
| `EvidenceBundle` | 每节段一张 axial PNG 的 EvidenceItem 列表。 |

当前 cervical SCS paired workflow 的固定逻辑：

1. 在 sagittal volume 上运行 TSS，使用硬标签 `1 ∪ 2`（spinal cord + spinal canal）按面积选最大 sagittal slice。
2. 将该 sagittal index 与其两侧各 1 张（边界自动截断）作为 matching 源切片；根据 TSS vertebra labels 和 paired axial affine，复用 `build_tra_map.py` plane-side balance 与 inter-vertebra gap 选择逻辑，产生 C3–C7 与 4 个椎间的 axial index。
3. 导出 sagittal reference PNG 与 9 张 axial PNG；原始 NIfTI 只读引用，不复制、不修改。

另有一个可对照配置 `cervical_scs_paired_soft_distance.json`。它对 TSS 的五个椎体标签（13–17）和四个椎间盘标签（64–67）分别计算 sagittal 分割的三维质心，转换到 RAS+ 毫米坐标，再计算质心到每张 axial 切片平面的垂直距离 `d`；以 `exp(-0.5 * (d / sigma_mm)^2)` 作为软分数，取距离最小（分数最高）的平面。默认 `sigma_mm=3.0`，同时将每个候选切片的距离写进 matches JSON。该策略使用完整 3D anatomy label，不依赖最大椎管参考层及其相邻切片；与 `ratio`（比例计算）是两个独立选项，不会互相覆盖。

静态 validator 负责检查 schema、operation 参数与有序引用、类型/view/label 兼容、required views 和最终输出连接性。执行期间任何 step 失败则 run 标记为 `failed`，step 错误写入 manifest；不输出“部分成功”作为完整结果。

## 3. 运行目录与防覆盖规则

每次调用创建新的：

```text
<runs-root>/
└── <case-id>/
    └── <run-id>/
        ├── manifest.json
        ├── steps/<step-id>/          # 每个 step 的中间文件/日志
        └── outputs/
            ├── sagittal/             # sagittal PNG
            └── axial/                # level-specific PNG
```

- `case-id` 与 `run-id` 只允许英数、点、下划线、连字符；未显式提供时分别取 sagittal 文件名与 UTC 时间 + 随机后缀。
- `mkdir(exist_ok=False)` 保证同名 run 不会覆盖已有结果；冲突时立即失败，调用者应使用新 run id。
- step 输出只写入 `steps/<step-id>/`，公开 evidence 写入 `outputs/`。filename/prefix 不允许路径分隔符或 `..`，防止逃出 run 目录。
- NIfTI 输入按绝对路径记录并写入 SHA256、大小、shape、affine、推断 view 轴；默认不复制医学影像到 run。
- 每个派生文件写入相对路径、类型、大小、SHA256 与结构化 metadata。

## 4. Manifest

`manifest.json` 至少包含 `manifest_version`、workflow/case/run 标识、状态与时间、坐标规则、输入 provenance、按序 step 状态/输入/参数/命令/产物、最终 workflow evidence 引用、运行依赖版本。其 `workflow_outputs[].artifacts[]` 使用 run 内相对 PNG 路径，不保存 Python 对象或绝对临时路径。

run 的成功条件是所有 step 成功、最终 evidence view 覆盖 query 所需 views、manifest 可序列化并写盘。当前输出 evidence 是 PNG 预览；原始 NIfTI 和完整 label volume 均保留为源/中间 artifact，不由调用者手工拼路径来猜选片索引。

## 5. 命令示例

```bash
kepa workflow validate configs/workflows/cervical_scs_paired_series.json
kepa evidence build-paired \
  --sag-nii /data/case_sag.nii.gz \
  --ax-nii /data/case_ax.nii.gz \
  --workflow configs/workflows/cervical_scs_paired_series.json \
  --runs-root /data/kepa-runs \
  --case-id case-001
```

当前病例 GT 回归 fixture 只保存匿名化的切片索引与方向换算，不含姓名、原始路径或扫描影像。真实检查数据仍只存在于受控数据目录。
