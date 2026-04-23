# viz_final — Pipeline final figures

8 张交付图的 matplotlib 生成脚本。每张图都产出 **PNG (300 DPI)** 和 **SVG**，
输出到项目 `env_field_pipeline_output/viz_final/`。

## 运行

```bash
cd metaagent_run/steps/env_field_pipeline/viz_final/
python3 01_pipeline_funnel.py
python3 02_main_list_top30.py
python3 03_signature_4env.py
python3 04_mixs_alignment.py
python3 05_threshold_calibration.py
python3 06_umap_family_hexbin.py
python3 07_umap_A_subtypes.py
python3 08_traceability_example.py
```

每个脚本独立运行，只依赖 pandas / numpy / matplotlib（已在 llm conda env 里）。

## 共享文件

| 文件 | 内容 |
|---|---|
| `viz_palette.py` | 配色（家族/环境/category/MIxS）+ `install_style()` 字体字号 + `save_png_svg(fig, stem)` |
| `viz_io.py`      | `load_csv(name)` / `load_json(name)` 从 `env_field_pipeline_output/` 读文件 |

## 图目清单

| 编号 | 图名 | 数据源 | 脚本 |
|---|---|---|---|
| C | `viz_final_pipeline_funnel` | `env0_stats.json` / `env1_stats.json` / `env2_eval_report.json` / `env4_canonicals.csv` / `env5_canonical_classified.csv` / `env5_main_list.csv` | `01_pipeline_funnel.py` |
| B | `viz_final_main_list_heatmap` | `env5_main_list.csv` | `02_main_list_top30.py` |
| B-sig | `viz_final_signature_4env_comparison` | 4× `env5_signature_{Open_ocean,Coastal_waters,Lake,Wetlands}.csv` | `03_signature_4env.py` |
| D | `viz_final_mixs_alignment` | `env4_mixs_alignment_log.csv` + `env4_canonicals.csv` | `04_mixs_alignment.py` |
| E | `viz_final_threshold_calibration` | `env5_threshold_calibration.csv` + `env5_thresholds.json` | `05_threshold_calibration.py` |
| A | `viz_final_umap_family_hexbin` | `viz/env3_viz_coords.csv` + `viz_v2/env3_knn_family_confusion.csv` | `06_umap_family_hexbin.py` |
| A-sup | `viz_final_umap_A_subtypes` | `viz/env3_viz_coords.csv` | `07_umap_A_subtypes.py` — 3×4，11 A 子类 + all-A |
| F | `viz_final_traceability_example` | `env5_main_list.csv` + `env4_canonical_to_raw_key.csv` + `env3_final_annotations.csv` + `env4_canonicals.csv` | `08_traceability_example.py` — **单面板 salinity**，展示 Step 3 → Step 4 → Step 5 的收敛（65 raw_keys → 7 canonicals → 1 target）|
| A-sup-B | `viz_final_umap_B_subtypes` | `viz/env3_viz_coords.csv` | `09_umap_B_subtypes.py` — 2×3，4 B 子类 + all-B |
| A-sup-C | `viz_final_umap_C_subtypes` | `viz/env3_viz_coords.csv` | `10_umap_C_subtypes.py` — 3×3，8 C 子类 + all-C |

## 配色说明

- **Family**：A（红 `#E63946`）/ B（青绿 `#2A9D8F`）/ C（深蓝 `#1D3557`）/ D（灰 `#6C757D`）
- **Env**：Open_ocean 深蓝 `#0D3B66`、Coastal_waters 青绿 `#2A9D8F`、Lake 浅蓝 `#5DADE2`、Wetlands 棕褐 `#8B5A2B`
- **Category**：Universal 深青 / Cross-biome 青绿 / Signature 金 / Niche 灰

## 关键数字口径约定

- **主清单 93.23% PMID 覆盖率**：分母是 **phase 4 canonical 池 (154,155 PMID×env)**，不是 env0 的 231,340
- **Stage 1 76.94%**：分母是 env0 的 231,340（和 stage 1 产出文件 `env1_stats.json` 一致）
- **Niche = 1,201**：来自 `env5_canonical_classified.csv` 的真实分类计数
- **Signature 总数 140 vs 原始 145 canonical**：经过 Safety Net 合并后写入 4 份文件的总行数 140；145 是 raw canonical 数

## 数据更新后重跑

```bash
# 如果只是 step 4/5 重跑了，全部 8 张图都可以直接重跑
cd metaagent_run/steps/env_field_pipeline/viz_final/
for f in 0*.py; do python3 "$f"; done
```

Figure A / A-sup 依赖 `viz/env3_viz_coords.csv`（UMAP 坐标）——这份是 step 3 embedding 流程产出的静态文件，只在 step 3 embedding 重跑时需要更新。

## 已知限制

- **Figure F**：当前只画 `salinity` 一个 target，作为 3 阶段收敛的教学示例。若要换成别的 target，改 `08_traceability_example.py` 里 `TARGET = "salinity"` 即可；注意新 target 的 canonical 数会影响整图高度（现在 `figsize=(15, 12)` 是给 7 canonical × ~4 raw_key/canonical 设计的）。
- **Figure A kNN 纯度**：图例底部小框的 A/B/C 数字会优先从 `viz_v2/env3_knn_family_confusion.csv` 读取；若该 CSV 缺列或不在则回退到硬编码（97.4% / 79.8% / 96.5%）。
