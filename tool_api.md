# KEPA MRI Tool API 与联动约定

状态：接口设计草案 v0.1。本文定义工具边界和 workflow 校验要求；不是已完成的影像算法实现。正式实现前，具体策略名、模型能力及数值参数需与论文、旧代码和临床审核结果逐项核对。

## 1. 总体约定

MRI Tool Set 包含四个工具族：Segmentation、3D Matching、Slice Selection、ROI Extraction。每个工具族可包含多个 operation。JSON workflow 描述调用哪个 operation、参数是什么、输入来自何处；step 顺序和引用共同形成依赖图。

推荐数据流：

```text
MRIExam / DiagnosticQuery
       │
       ├── Segmentation ─────── AnatomyMask(s)
       │                            │
       ├── 3D Matching ◀───────────┘ + target series geometry
       │       │
       │       └── SliceMatchScores
       │                  │
       ├── Slice Selection ◀┘ + candidate series / anatomy criteria
       │       │
       │       └── SelectedImageSlice
       │                  │
       └── ROI Extraction ◀┴── corresponding-view AnatomyMask(s)
                  │
                  └── EvidenceItem(s)
```

工具按职责拆分：3D matching 计算源解剖与目标图像切片之间的几何对应分数；slice selection 根据任务策略从候选中选出最终切片。不能把“匹配得分”和“选中切片”混成一个不可检查的黑盒。

## 2. 公共数据类型

### `MRIExam`

- `exam_id`: 去标识化检查标识。
- `series`: `SeriesRef` 列表或按 view 命名的映射。
- 可用 view、影像模态及采集信息；不要求每项检查都有所有 view。

### `SeriesRef`

- `series_id`, `view`, `image_uri`。
- 3D 几何：NIfTI affine 或 DICOM 的 ImagePositionPatient、ImageOrientationPatient、PixelSpacing、切片位置/UID。
- `voxel_to_patient`, `coordinate_system`, `spacing`, `orientation`。
- 任何方向变换、重采样、像素翻转都必须有可组合的 transform 记录。

### `AnatomyMask`

- `mask_uri` 与 `source_series_id`；mask 必须明确在哪个图像网格/坐标系中。
- canonical anatomy labels 和外部 label mapping。
- 每个结构的体素/点集、边界框、面积或中心点等可派生量。
- segmentation backend、模型/checkpoint 标识、软件版本、参数和输入/输出 hash。

### `SliceMatchScores`

- `source_series_id`, `target_series_id`, `anatomical_level`, `direction`。
- 每个候选目标切片的 `SOPInstanceUID`/切片索引、patient-space 平面、距离、score、排名和拒绝原因。
- matcher 名称、单位、参数和方向约定。它保留候选，不自行丢弃排名信息。

### `SelectedImageSlice`

- `series_id`, `SOPInstanceUID`/slice index、图像引用和 2D 几何。
- `selection_policy`, 被选 score、候选排名和 tie-break 说明。

### `ROIResult` / `EvidenceItem`

- `image_slice_id`, ROI 的 `xyxy` 像素坐标、坐标约定（建议左上原点、半开区间）。
- 解剖目标、侧别、产生 ROI 的 rule 和参数。
- 可选 mask overlay、box-overlay image 或 crop artifact；canonical evidence 始终保留原图引用。

所有类型带稳定的 `type`/schema version，工具间通过类型化引用连接，不在 JSON 中传数组或模型对象本身。

## 3. 四类工具族 API

### 3.1 Segmentation

**职责：** 从指定 series（必要时是指定切片集合）获得后续匹配/选片/构框所需的解剖结构。

**初始 operation：** `segment`。

| 项 | 约定 |
|---|---|
| 必需输入 | `series: SeriesRef` |
| 输出 | `mask: AnatomyMask`，与源 series 几何对应 |
| 必需配置 | `view`, `backend`, `model_id`, `structures` |
| 可选配置 | `preprocessing`, `device`, `batch_size`, `cache_policy`, `model_options` |

`backend` 只选择执行入口，例如 `sct_cli`、独立上游 Python API 或本地模型 adapter；`model_id` 指定具体任务/模型。模型能力由 versioned model catalog 声明：支持的 view、结构标签、输入维度、软件/权重版本。workflow 不能请求模型未提供的结构。

### 3.2 3D Matching

**职责：** 把来源 view 的解剖目标映射到目标 series 的候选切片平面，并计算可审计的候选分数。此工具使用分割输出，也使用 DICOM/NIfTI patient-space 几何。

**初始 operation：** `match`，返回候选分数，不直接返回最终图片。

| 项 | 约定 |
|---|---|
| 必需输入 | `source_series`, `target_series`, `source_anatomy`, `level`；按策略可另需 `reference_slice` 或 `reference_point` |
| 输出 | `matches: SliceMatchScores` |
| 必需配置 | `direction`, `strategy`, `candidate_filter` |
| 可选配置 | `distance_unit`, `sigma_mm`, `side`, `position_prior`, `max_candidates` |

首批方向包括 `sagittal_to_axial` 和 `axial_to_sagittal`。strategy 区分论文中的解剖匹配方法：例如用目标椎体分割点在候选轴位平面两侧的分布评价椎体中心层面；或把目标区域映射到候选平面并计算距离加权软匹配。侧方矢状位可由轴位分割产生侧别参考点，再按其 patient-space 到各矢状位平面的距离排序。

匹配策略的输入需求必须在 registry 中显式声明。例如 `vertebral_plane_balance` 需要带 level 的 vertebra mask；`point_to_plane` 需要 patient-space reference point。不能只通过自由文本 strategy 名称猜需要哪些输入。

首个可运行的几何原型现位于 `src/kepa/tools/matching/matching3d.py`：`match_sag2tra(...)` 与 `match_tra2sag(...)`。它们接收已经按目标节段筛选的 3D mask、mask 的 voxel-to-patient affine 和目标序列的 `ImageSeriesGeometry`，输出所有目标层面的距离、Gaussian score、排序及可选的距离门控结果。当前原型支持 `centroid` 与 `max_support` 两种策略；尚未读取 DICOM/NIfTI 文件、调用分割，也未接入 workflow executor。输入 affine 必须处于同一个明确的 patient coordinate convention；不能直接把未经验证的 RAS 与 LPS 几何混用。

单个 3D NIfTI 的首个 workflow runner 位于 `src/kepa/execution/single_nifti.py`，命令为 `kepa evidence build --input-nii ... --level ... --workflow ... --out ...`。它按 JSON step 调用 SCT TotalSpineSeg（相同输入/模型只执行一次并缓存）、最大椎管面积 selector、`sag2tra` 和最佳匹配 selector，最终导出两张 PNG 与 `evidence.json`。此 runner 将原 NIfTI 解释为一个有物理坐标的 3D volume，所选 axial plane 可能是从该 volume 提取的重格式，不代表独立采集的 axial series；输出记录 plane normal 和 spacing。当前实现暂不支持 DICOM、多序列输入、SCT 以外分割后端或重新采样到标准正交 MPR 网格。

当前 SCS 基础 workflow 的分割职责先固定为两种：`local_unet/cervical_vertebra_unet` 在 sagittal T2 上提供节段级 vertebra mask，`sct_cli/totalspineseg` 请求 `spinal_canal` mask，供矢状位最大面积选片；横轴位若要构建椎管 ROI，也对 axial series 调用 TotalSpineSeg 并只向下游暴露所需 canal mask。独立 SCT `spinalcord` task 暂不用于此 workflow，因为 spinal cord 与 spinal canal 是不同结构。

### 3.3 Slice Selection

**职责：** 将候选影像与 anatomy/matching score 按 task policy 排序、过滤并选择最终切片。

**初始 operation：** `select`；也可保留便于阅读的策略别名，如 `select_sagittal_reference` 与 `select_matched_slice`，但都归入同一 tool family。

| 项 | 约定 |
|---|---|
| 必需输入 | `series: SeriesRef`；以及 selector 所需的 `AnatomyMask` 或 `SliceMatchScores` |
| 输出 | `selected: SelectedImageSlice`，可附 `alternatives[]` |
| 必需配置 | `view`, `selector`, `level`（目标级别适用时） |
| 可选配置 | `side`, `tie_break`, `min_score`, `fallback_policy`, `top_k` |

初始 selector 例子：

- `max_structure_area`：中央矢状位参考层面，基于指定结构（例如 spinal canal）的逐层面积。
- `best_match`：按 3D matching 得分选目标轴位层面，并遵循级别区间/切片覆盖约束。
- `nearest_side_reference`：在侧别约束下，从 axial-to-sagittal 匹配候选中选侧方矢状位。

selector 返回选择理由和候选排名。低分、目标级别不在覆盖范围或左右侧无法判定时，按明确的 fallback/error policy 输出，不静默选中任意切片。

### 3.4 ROI Extraction

**职责：** 在已选择切片上，根据目标解剖 mask 和任务规则产生 ROI；不负责选片，也不执行分割。

**初始 operation：** `construct`。

| 项 | 约定 |
|---|---|
| 必需输入 | `image: SelectedImageSlice`, 同 view 的 `AnatomyMask`，目标 `level`/`query` |
| 可选输入 | `SliceMatchScores`、其他相关结构 mask、左右侧 reference |
| 输出 | `evidence: EvidenceItem`，包含原图引用及 bbox |
| 必需配置 | `rule`, `target_structure` 或 `target_structures` |
| 可选配置 | `padding_mm`, `offset_mm`, `side`, `clip_to_image`, `render_overlay`, `export_crop` |

ROI rule 可分为：目标 mask 外接框、相关结构框并集、相对参考结构偏移框、按解剖定位的固定/比例窗口。padding/offset 优先用毫米定义，再根据当前切片 PixelSpacing 换算到像素；侧别要按 patient orientation 转换，不能假设图像左侧永远是患者左侧。

### 3.5 工具间公共配置

所有 operation 统一接受可审计的 `runtime` 配置：`timeout_sec`, `device`, `cache_key`, `random_seed`（仅适用时）。workflow 层还可声明 `on_error`（`fail`/`skip_query`）和 provenance 输出策略。用户/患者数据路径、checkpoint 路径与密钥不写进公开 workflow JSON，由运行环境变量或私有配置注入。

## 4. 工作流如何联动

### 4.1 中央管狭窄：sagittal 定位 axial level

```text
Segmentation(sagittal T2: vertebrae/discs/canal)
         ├── SliceSelection(max canal area) ──> 中央矢状位证据
         └── 3D Matching(sagittal anatomy + axial series geometry + level)
                         └── SliceSelection(best match) ──> 目标 axial slice
Segmentation(axial T2: spinal canal) ────────────────────────┘
ROI Extraction(sagittal slice + sagittal anatomy + level) ──> sagittal ROI
ROI Extraction(axial slice + axial canal mask + level) ─────> axial ROI
```

“通过 sagittal 找对应 axial”中，分割提供**目标是哪一个解剖节段/该节段在何处**；3D matching 用患者坐标和候选 axial 平面建立对应并给出 score；slice selection 结合 level 和 task policy 决定最终 axial 图。ROI extraction 只在选好的图上画出目标区域。若 target axial segmentation 可以在整组序列一次运行，就复用其 mask；若模型只适合单切片，则需在该 query 的 workflow 中先选片再分割，并由静态类型校验确认顺序。

### 4.2 左/右椎间孔：轴位定位后回到侧方矢状位

先由 sagittal 解剖分割与 `sagittal_to_axial` matching 找目标 level 的 axial slice；再用 axial canal/anatomy mask 产生左右侧 reference point，分别执行 `axial_to_sagittal` matching；Slice Selection 按 side 选择左/右 lateral sagittal slice；最后对 sagittal 与 axial evidence 分别构框。左右任务可以共享已完成的 segmentation 与 level matching，只在 side-specific reference、selector 和 ROI rule 上分支。

这体现 workflow 应表示为依赖图：可以共享计算结果、分支处理左右侧，最终合并 EvidenceItems。JSON 文件仍可用有序 `steps` 保存；每个输入端口引用生产者输出，validator 据引用构图和检查可执行顺序。

## 5. Validator 要检查的规则

### 静态校验：读 workflow 时即可完成

1. JSON Schema、版本、未知字段、重复 step ID。
2. tool/operation/model/selector/rule 均已注册，参数在对应 operation 的 schema 中且值合法。
3. 所有输入引用均存在；引用只指向外部输入或已定义 producer；输出端口名正确。
4. producer 输出类型与 consumer 输入类型兼容，view/side/space constraints 满足。
5. 某个 selector 使用 matching scores 前必须有 matching producer；ROI 必须取得已选 image 和兼容空间的 anatomy mask。
6. 分割 backend/model 对请求的 view 和 structures 有能力；匹配 strategy 拿到所需 mask、level 或 reference point。
7. 无环、无重复 ID、无未连接/不影响最终 evidence 的孤立步骤。
8. workflow 输出类型是 EvidenceItem；声明 required views 的 query，最终输出覆盖所需 views。多 view 跨 series query 必须有相应 3D matching 链路。

### 运行前校验：拿到实际 MRI 后完成

1. query 需要的 series/view/目标 level 是否存在，series 是否覆盖对应解剖范围。
2. DICOM/NIfTI 的几何字段完整；source 与 target 能映射到同一 patient coordinate frame。
3. 对应 segmentation model/checkpoint 已安装且版本/hash 符合配置；硬件、显存、SCT/上游依赖可用。
4. mask 非空、标签映射有效、matching 有合格候选、selection 未触发禁止的 fallback。
5. ROI 未越界、非空、侧别/spacing 换算有效，派生 evidence 保留 source series/slice 关系。

静态校验通过不等于这名患者一定可生成证据；运行前校验和执行后的 QC 仍需保留。

## 6. 建议的 JSON 引用形态

```json
{
  "id": "match_sagittal_to_axial",
  "tool": "3d_matching",
  "operation": "match",
  "inputs": {
    "source_series": "$input.exam.series.sagittal_t2",
    "target_series": "$input.exam.series.axial_t2",
    "source_anatomy": "$steps.segment_sagittal.mask",
    "level": "$input.query.level"
  },
  "options": {
    "direction": "sagittal_to_axial",
    "strategy": "vertebral_plane_balance",
    "candidate_filter": "between_adjacent_vertebrae"
  }
}
```

该片段说明接口意图，暂不代表当前 validator registry 已实现这些最终名字。正式定稿后需要同步更新 `workflow.schema.json`、tool registry、示例 workflow 和 validator，避免文档与可运行配置漂移。

## 7. 建议实现顺序

1. 锁定公共数据对象、坐标规范和 `AnatomyMask`/`SliceMatchScores`/`SelectedImageSlice`/`EvidenceItem` schema。
2. 将 tool registry 从当前示例版扩展为版本化 operation catalog，包含端口、选项 schema、能力和 side/view/space constraints。
3. 完成 `sagittal_to_axial` 这条 SCS 基础链路，先以已有分割结果或小型 fixture 检查几何匹配，再接入真实 segmentation backend。
4. 实现 ROI rule 和证据 provenance/QC。
5. 再增加 `axial_to_sagittal` 侧方 NFN 分支及左右共享缓存。

