# Phase 6 Decisions — `hydrosphere_metadata_norm` 目标清单定版

> **Produced**: 2026-04-24
> **Scope**: env_field_pipeline 主清单 + 4 份 Signature 清单的 target schema 决策
> **Upstream**: env2 展开后 + phase3 / 3a-norm / phase4 (post-bugfix) / phase5 (granularity_mode=coarse, 403 main + 135 sig)
> **Downstream**: `metaagent_run/steps/step5/upstream_loader.py` 消费的 `env6_extraction_targets.json`

---

## 1. 项目目的与核心矛盾

### 1.1 流水线最终目的

为 `metaagent_run/steps/step5` 的**靶向元数据抽取**提供一份权威的 "**抽取目标清单**"——告诉 LLM 对于每篇水圈环境文献，应该从原文中抽出哪些**具体字段**的值，并与 biosample accession 关联。

### 1.2 核心矛盾

> *"元数据字段的粒度本身没有客观正确答案。`depth` vs `water_depth` vs `tot_depth_water_col` vs `sediment_depth` 的拆分粒度取决于下游用途——做全局 PMID 覆盖统计就想合并，做海洋 vs 湖泊对比就想拆分。MIxS 本身没给明确的对齐层级，所以无论是 LLM、专家、embedding 还是规则，都只是在一个本质上无唯一解的问题上做一次投票。这个投票一旦写死成单层 canonical 映射，就永远欠下游场景的债。"*

### 1.3 两个具体痛点（驱动 phase6 必要性）

1. **主清单颗粒度不对齐**：Top 30 同时出现 `collection_date / sampling_date / date / year / sampling_time` 5 条同概念字段，读者无法一眼看清"水圈样本关键元数据是哪几类"
2. **无法制定 step5 抽取清单**：env5 的 443 条主清单是"字段聚类审计产物"，不是"抽取 schema"——规模和结构都不匹配 step5 的 `upstream_loader` 消费口

---

## 2. 设计哲学

### 2.1 原则 1：承认粒度是**选择**而非真理

粒度无唯一解 ≠ 不能对齐。**选定一个层级 + 透明披露选择**即可成立。本清单明确宣告：**主清单粒度 = "对下游环境样本抽取任务下，subtype 内部最能代表该测量类的具体 canonical quantity_kind"**。

### 2.2 原则 2：两个产物分离，各自一致

| 产物 | 粒度哲学 | 承载物 |
|---|---|---|
| **env5_main_list.csv** (443 条) | 保留细节，展示数据丰富度 | 论文附录 / 透明度报告 |
| **env6_extraction_targets.json** (~66 主 + 100+ sig) | 激进合并，服务下游抽取 | step5 upstream_loader 输入 |

两者**共用同一 pipeline 中间产物**（env3/4/5 canonicals），但**渲染成不同视图**。回溯 env6 的每条 target 都能映射到 env5 若干 canonical（通过 traceability 表）。

### 2.3 原则 3：锚点是 **subtype**，不是 MIxS

- Phase 3 数据驱动归纳出的 **24 个 subtype**（21 非兜底 + 3 catch-all）就是颗粒度对齐的层级
- **每个 subtype 在主清单里的 target 数是受控的**，由 subtype 的内部语义结构决定：
  - **单一测量 subtype**（salinity / temperature / ph_alkalinity / conductivity_tds / chlorophyll_pigment 等）→ 主清单 1-3 条
  - **多化学种并列 subtype**（nutrient_chemistry / trace_chemistry / oxygen）→ 按分子/离子/元素锚点，各独立成 target
  - **定位/时间/分类 subtype**（time_point / sampling_site / habitat_biome 等）→ 合并为 1 条
  - **Catch-all subtype**（other_chemistry / other_categorical / other）→ 不进主清单（归 R3 剔除或 signature）
- **MIxS 89 slot 仅作命名参考**（当 target 名有 MIxS exact 对齐时优先使用 MIxS 命名），**不作为过滤条件**——真实研究字段（peat_depth / water_table_depth / tidal_stage / reef_zone / iron / manganese / cadmium 等 UNMAPPED 字段）完整保留

### 2.4 原则 4：路线 Y——step5 prompt 带 aliases 展示

主清单每条 target = `{target_name, subtype, aliases[], signature_env?}`。**step5 的 section_extract prompt 要改为**：

```
## Target Metadata Fields
- collection_date (aliases: sampling_date, date, year, month, sampling_time, ...)
- water_depth (aliases: depth, sampling_depth, bottom_depth, ...)
- ...
```

→ 允许更激进合并（只要 aliases 覆盖全），抽取召回率更高。

### 2.5 原则 5：合并/拆分的 3 判据

1. **语义一致**：都描述同一环境属性
2. **量纲/数据结构一致**：值的单位相同、结构相同（路线 Y 下放宽，LLM 通过 aliases 提示能区分不同单位）
3. **LLM 常识易识别**：不加 aliases 提示 LLM 常识能认出（路线 Y 下仍建议，避免 prompt 过长）

**不能合并的例子**：
- `dissolved_oxygen` (mg/L) ≠ `biochemical_oxygen_demand` (mg/L O2 当量) —— 语义不同
- `turbidity` (NTU) ≠ `secchi_depth` (m) —— 量纲不同
- `nitrate` ≠ `nitrite` —— 不同分子

### 2.6 原则 6：R1-R4 剔除理由规范

每条不进 env6 schema 的字段必须归入四类理由之一（记录在 trace 表）：

| 代码 | 理由 | 论文表述 |
|---|---|---|
| **R1** | **PMID 覆盖度阈值** | 覆盖度低于 ~30-50 篇论文，无代表性 |
| **R2** | **量纲/值结构冲突** | 值结构与主 target 不兼容（即使语义相关） |
| **R3** | **语义偏离样本环境元数据** | 是采样设备/方法/处理参数/研究设计/地质学 |
| **R4** | **组合字段** | 是多值组合（coordinates = lat+lon），由分量 target 覆盖 |

### 2.7 原则 7：Signature 字段细粒度保留

4 份 Signature 清单（Open_ocean / Coastal_waters / Lake / Wetlands）**不做合并**，每个 signature canonical 独立成 target。理由：
- Signature 本身就是"某 env 特有 + 在其他 env 鲜见"
- 合并会破坏特征性（lake_name 合入通用 sampling_location 就丢失了"每个湖分组查询"的能力）
- Signature 规模有限（每 env 10-25 条），不致臃肿

### 2.8 原则 8：展开表（raw_key normalization）—— 表述规范化≠粒度决策

基于 step2 evidence 保留后的 kept pool，引入 115 条 "100% 无歧义" 展开表（化学元素 20 / 化学式 40 / 氮种 15 / 水化学缩写 25 / 温度位置 17 / 色素比值 等）。这是**表述规范化**（bod = biochemical_oxygen_demand 是同物两写），**不引入语义判断**，不违背"数据驱动归纳"的原则。效果：
- 44 条历史 manual_shortcode_fix 补丁 100% 自动退役
- Phase 3 LLM 看全称字段判 subtype 更准
- MIxS exact 对齐 +118（350 → 468）

---

## 3. Pipeline 整体流程

```
┌──────────────────────────────────────────────────────────────┐
│  env_field_pipeline (hydrosphere_metadata_norm)              │
│                                                              │
│  Phase 0: raw_key × env × pmid 三元组聚合 (53,068 raw_keys)  │
│  Phase 1: total_pmid ≥ 3 切分 (6,140 main / 46,928 orphan)   │
│  Phase 2: LLM is_env_metadata 二元判断 (4,132 kept)          │
│       ↓ [apply_raw_key_expansion.py: 397 rows 全称化]        │
│  Phase 3: 四槽位结构化标注 (family/subtype/qk/bag)           │
│  Phase 3a-norm: qk 规范化 (2151 → 1450 unique qk)            │
│  Phase 4: 桶内聚类 + EDC verify + MIxS 对齐 (1,815 canonicals)│
│  Phase 5: 环境熵分类 + Rule A collapse (405 main, 135 sig)   │
│       ↓                                                      │
│  【Phase 6 (本文档)】: target schema 渲染                    │
│       ├─ 主清单 ~66 target × {name, subtype, aliases[]}      │
│       ├─ 4 sig × ~20-60 target × {name, aliases[]}           │
│       └─ env6_extraction_targets.json (step5 直接读)         │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## 4. 批次 1：无歧义 subtype (17 主 target)

### 4.1 salinity subtype (1)

| Target | Aliases (主要) |
|---|---|
| **salinity** | water_salinity, sal |

### 4.2 temperature subtype (1)

| Target | Aliases |
|---|---|
| **temperature** | water_temperature, surface_water_temperature, sea_water_temperature, air_temperature, in_situ_temperature, ambient_temperature, sea_surface_temperature, mean_annual_temperature, maximum_temperature, minimum_temperature, bottom_temperature, potential_temperature, temperature_range, temp, temperature_c, temperature_deg_c, t_c, t°c, temp_c |

**R2 剔除**：temperature_gradient (17) — °C/m 与 °C 量纲不同

### 4.3 ph_alkalinity subtype (2)

| Target | Aliases |
|---|---|
| **ph** | pH |
| **alkalinity** | total_alkalinity, bicarbonate, hco3, ta, at, alk |

### 4.4 conductivity_tds subtype (2)

| Target | Aliases |
|---|---|
| **conductivity** | ec, EC, electrical_conductivity, electric_conductivity, specific_conductance, conductance, ec25 |
| **total_dissolved_solids** | tds, TDS, total_dissolved_solid (拼写变体), total_dissolved_salts, dry_residue |

**跨 subtype 移入**：`total_suspended_solids` → 移到 turbidity_transparency 的 `suspended_particulate_matter`

### 4.5 chlorophyll_pigment subtype (2)

| Target | Aliases |
|---|---|
| **chlorophyll_a** | chlorophyll_a_concentration, chlorophyll_concentration, photosynthetic_pigments, chl_a, chl-a, chla, chlorophyll-a, chlorophyll-a_concentration |
| **fluorescence** | chlorophyll_fluorescence, chlorophyll_a_fluorescence |

**R1 剔除**：pheophytin (50) / phaeopigment (22) — 叶绿素降解副产物，非主状态指标；**R1+R2 剔除**：primary_productivity / productivity (25+11) — gC/m²/d 通量量纲不同

### 4.6 turbidity_transparency subtype (3)

| Target | Aliases |
|---|---|
| **turbidity** | visibility, colour, water_clarity |
| **secchi_depth** | transparency, secchi, secchidepth, secci_depth, secchi_disc_depth, secchi_disk_depth |
| **suspended_particulate_matter** | tss, suspended_solids, total_suspended_solids, spm, spm_concentration, tsm, total_suspended_matter, suspended_sediment_concentration |

**R3 剔除**：filter_diameter (29) — 过滤器规格，属采样工具

### 4.7 geo_coord subtype (2)

| Target | Aliases |
|---|---|
| **latitude** | coordinates, geographic_coordinates, latitude_range, geographical_position, utm_coordinates, gps_coordinates, co-ordinates, gps_co-ordinates |
| **longitude** | coordinates, longitude_e, longitude_range, latitude_longitude, utm_coordinates, gps_coordinates |

**R4 剔除**：coordinates (组合字段，值含 lat+lon，由 latitude/longitude 分量 target 覆盖)

### 4.8 oxygen subtype (4)

| Target | Aliases |
|---|---|
| **dissolved_oxygen** | oxygen, o2, do, DO, oxygenation, oxygen_concentration, dissolved_o2 |
| **oxygen_saturation** | dissolved_oxygen_saturation, do_saturation, oxygen_sat, do_percent_saturation |
| **biochemical_oxygen_demand** | bod, bod5, five_day_biochemical_oxygen_demand, biological_oxygen_demand |
| **chemical_oxygen_demand** | cod |

⚠️ **phase6 必须做**：phase4 的 EDC verify 未成功拆分 BOD/COD（它们被归到同一 canonical），phase6 要手工按字面匹配把 `chemical_oxygen_demand` / `cod` 的 raw_keys 从 BOD canonical 拆出来作独立 target。

---

## 5. 批次 2 第一组（23 主 target）

### 5.1 vertical_position subtype (3)

| Target | Aliases |
|---|---|
| **water_depth** | depth, sampling_depth, bottom_depth, water_depth, max_depth, mean_depth, depth_range, average_depth, collection_depth, water_column_depth, thermocline_depth, maximum_depth, water_layer, soil_depth, water_column, depth_interval, bathymetry, core_depth, mixed_layer_depth, depth_layer, depth_zone, maximum_water_depth, minimum_depth_in_meters, depth_strata, mean_water_depth, euphotic_zone_depth, sampling_layer, photic_zone_depth, depth_max, sampled_depth, min_depth, depth_gradient, oxygen_penetration_depth, nitracline_depth, bot_depth, stratigraphic_depth, halocline_depth, sample_depths, sediment_depth, position, layer |
| **elevation** | altitude, sea_level, elevation_range |
| **ice_snow_thickness** (NEW) | ice_thickness, snow_depth |

**跨 subtype 调整**：
- `secchi_disk_depth` → 移到 turbidity_transparency 的 `secchi_depth` alias
- `tidal_elevation` / `tide_level` / `groundwater_level` / `groundwater_table` → 移到 physical_env_driver 的 `water_level`

### 5.2 time_point subtype (1)

| Target | Aliases |
|---|---|
| **collection_date** | sampling_date, date, year, sampling_time, month, time, collection_year, collection_time, day, sampling_month, sample_date, event_date, start_date, survey_date, year_sampled, time_of_day, collection_month, end_date, date_time, time_point, start_time, capture_date, local_time, survey_year, hour, month_year, end_time, date_deployed, datetime, observation_date, day_of_year, catch_date, timestamp, time_utc, day_night, event_time, local_date, date_time_utc, coring_date, years, time_of_year |

**R3 剔除**：
- `age_ma` (25) — 地质年代，非采样时间
- `isolation_date` (25) — 菌株分离时间，生物学概念
- `week` (17) — 时间单位（合入 collection_date）

### 5.3 sampling_site subtype (1)

| Target | Aliases |
|---|---|
| **sampling_location** | sampling_site, sampling_station, sampling_locality, sampling_point, site_name, transect, site_code, study_site, station_name, station_id, sampling_areas, type_locality, city, river_system, bay, survey_site |

**R3 剔除**：
- `core` (32) — 样品容器（岩心/土心），非地点
- `sampling_device` (19) — 采样器具
- `stranding_location` (16) — 搁浅地点（海洋哺乳动物专用）
- `release_location` (20) — 标记放生地点

### 5.4 habitat_biome subtype (1)

| Target | Aliases |
|---|---|
| **habitat** | vegetation, environment, zone, isolation_source, vegetation_type, site_type, sampling_zone, estuary, biome, intertidal_zone, wetland, pond_type, biotope, environment_feature, vegetation_zone, environment_biome, littoral_zone |

**R3 剔除**：
- `vegetation_cover` (43) / `macrophyte_cover` (12) / `macroalgae_cover` (11) — "覆盖度数值"非"生境类型"

---

## 6. 批次 2 第二组（6 主 target）

### 6.1 time_duration subtype (2)

| Target | Aliases |
|---|---|
| **season** | season, seasonal_period, summer_season, winter_season, growing_season |
| **sampling_period** | collection_period, sampling_time_period, sampling_date_range |

**合入批次 2 第一组 collection_date**：time_of_year, years
**R3 剔除**：day_length (22), flooding_duration (15), ice_cover_duration (13)

### 6.2 geo_region subtype (1)

| Target | Aliases |
|---|---|
| **geographic_location** | region, collection_country, province, sampling_region, basin, island, coast, geographic_area, geo_loc_name, subregion, geography, sub_region, region_of_origin, higher_geography, sea_area, river_basin |

**R3 剔除**：latitudinal_gradient (15)

### 6.3 material_medium_type subtype (3)

| Target | Aliases |
|---|---|
| **sediment_type** | sediment_type, soil_type, sediment_layer, type_of_sample, environment_material |
| **size_fraction** | size_fraction, mesh_size, filter_size |
| **grain_size_class** (NEW) | clay, clay_content, sand_content, fine_sand |

**R3 剔除**：
- rock_type (27), mineralogy (14), lithology (83), formation (23), quartz (13), mineral_composition (11) — 地质学概念
- net_type (11) — 采样网具
- color (63) — 混合 water/sediment color 歧义
- seawater (27) — 是值不是字段名

---

## 7. 批次 2 第三组（22 主 target）

### 7.1 nutrient_chemistry subtype (12)

| Target | Aliases |
|---|---|
| **nitrate** | no3, no3-, no3−, NO3, nitrate_nitrogen (如果下游出现), n_no3, no3-n, n-no3, nitrate-n, nitrate-nitrogen |
| **nitrite** | no2, no2-, no2−, NO2, nitrite_nitrogen, no2-n, n_no2, nitrite-n, nitrite-nitrogen |
| **ammonium** | nh4, nh4+, NH4, ammonium_nitrogen, nh4-n, nh4+-n, n_nh4, n-nh4, ammonium-nitrogen, ammonia, ammonia_nitrogen, nh3, nh3-n, ammonium-n |
| **phosphate** | po4, po43-, po43, PO4, phosphate_phosphorus, po4-p, po43-p, phosphate-p |
| **silicate** | sio2, sio4, sio3, sio32-, silicic_acid, sioh4 |
| **sulfate** | so4, so42-, so42, SO42−, sulfate_concentration, so4-2 |
| **methane** | ch4 |
| **total_organic_carbon** | toc, TOC, total_organic_c, org_c, oc, organic_c, soc, sediment_organic_carbon |
| **organic_matter_content** | om, OM, organic_matter, som, tom, total_organic_matter |
| **carbon_nitrogen_ratio** | c:n, c_n, cn, C:N, c:n_ratio, carbon/nitrogen_ratio, c/n |
| **nitrogen_to_phosphorus_ratio** | n:p, n_p, n/p, N:P_ratio, n:p_ratio, n:p_ratios |
| **total_sulfur** | — |

**合并规则**（批次 2 第三组 Q8）：`ammonium ⇔ ammonium_nitrogen`，`phosphate ⇔ phosphate_phosphorus`，`nitrate ⇔ nitrate_nitrogen` 各合 1 条（路线 Y 下 LLM 通过 aliases 识别不同单位表述）

**R3 剔除**：dissolved_inorganic_nutrients (495), inorganic_nutrients (69), macronutrient_concentrations (13), nutrient_availability (12), carbon (112 太泛), pn (15 歧义), phosphorus_pentoxide (11), urea (11)

### 7.2 trace_chemistry subtype (10)

| Target | Aliases |
|---|---|
| **iron** | fe, Fe, fe2+, fe3+, dfe, dFe, fet |
| **copper** | cu, Cu |
| **arsenic** | as, As |
| **chromium** | cr, Cr |
| **potassium** | k, K, k+, K+, total_potassium |
| **sulfide** | h2s, hs, hs-, hydrogen_sulfide, hydrogen_sulfide_concentration, h2s_concentration |
| **carbonate_content** | — |
| **total_hardness** | — |
| **sediment_grain_size** | — |
| **sediment_density** | bulk_density |
| **sediment_porosity** | — |
| **sediment_grain_size_stats** (NEW, 合并 4 条) | silt_content, mud_content, sorting, skewness |

**同位素类 (6)**：δ13c (153), δ15n (60), δ18o (52), 87sr_86sr (12), ωaragonite (24 + ωarag 合), ωca (13) — 全保留独立 target，Greek 记号保留

**ORP 跨 subtype 调整**：`oxidation_reduction_potential` (67) 从 trace_chemistry 归出独立主 target（或并入物化类）。orp_mv / eh / eh_mv 作 alias

**R3 剔除**：
- heavy_metals (64) / sediment_metals (63) — 泛指非具体元素
- mineral_composition (11) — 地质矿物学
- ion_concentration (13) — 泛指
- cs (11) / ct (15) — 歧义未消
- potassium_oxide (14), calcium_oxide (13), aluminum_oxide (12), titanium_oxide (12), magnesium_oxide (11) — 地质主量氧化物 XRF 分析，非水圈核心
- filter_pore_size (1895) — ⚠️ 需跨 subtype 归属确认，实际是采样工具/turbidity 类

### 7.3 trace/nutrient Signatures 保留

- **OO**: h2, poc_flux, particulate_organic_carbon, total_dissolved_inorganic_carb, mg_ca, nd, oxygen_isotope_composition, fepy_fehr, dFe_concentration, potential_density_anomaly, fehr_fet
- **Coa**: ωarag, total_suspended_matter
- **Lake**: major_ions, total_salts
- **Wet**: total_potassium

---

## 8. 批次 3：边缘 subtype (15 主 target)

### 8.1 spatial_metric subtype (3)

| Target | Aliases |
|---|---|
| **river_discharge** | discharge, river_flow |
| **slope** | — |
| **sedimentation_rate** | — |

**R3 剔除（26 条）**：
- `surface_area`, `distance` 系列 (distance_from_shore/coast/land/offshore/river_mouth/coastline 8 变体), sampling_area, catchment_area, watershed_area, width, water_area, perimeter, sediment_thickness, drainage_area, survey_area, sediment_core_length, total_core_length, shoreline, forest, speed, distance_from_vent, site_area, coastal_area, ap (歧义)

### 8.2 physical_env_driver subtype (8，含 water_level 移入)

| Target | Aliases |
|---|---|
| **precipitation** | total_precipitation, runoff |
| **wind_speed** | — |
| **wind_direction** | — |
| **light_intensity** | photoperiod, light_dark_cycle, light_attenuation, photosynthetic_active_radiation, light_penetration, daylength, par, PAR, par_irradiance, light_conditions |
| **pressure** | — |
| **soil_moisture** | soil_moisture_content, soil_water_content, mc, moisture_content, water_content, wc |
| **water_flow** | current_velocity, flow_rate, velocity, current_direction, water_velocity, wave_height, wave_action, wind (边缘变体), river_flow |
| **humidity** | humidity, relative_humidity |
| **water_level** (从 vertical_position 移入) | tidal_elevation, tide_level, groundwater_level, groundwater_table, wl, rsl |

**R3 剔除**：water_column_stratification (15), phase (11)

### 8.3 water_body_descriptor subtype (1)

| Target | Aliases |
|---|---|
| **water_body_type** | lake, surface_water, water_body, ocean, river, water, sea, source_water, pond (从 Lake sig 合入), inflow, outflow |

**R3 剔除**：surface (120 泛指), hydrology (13 学科名), coastline (11 地理概念)

### 8.4 env_state_context subtype (4)

| Target | Aliases |
|---|---|
| **tidal_stage** | tide, tide_condition, tidal_phase, tidal_zone, tidal_influence, tidal_condition, tidal_regime |
| **water_quality_status** | water_quality |
| **land_use** | land_use_type, land_cover |
| **stratification** | thermal_stratification, water_column_stratification |

**R3 剔除（18 条）**：
- **泛指**：environmental_variables (150), climate (117), condition (57), environmental_conditions (55), water_source (49)
- **研究设计/扰动**：exposure (40), wave_exposure (37), water_masses (32), protection_level (21), geomorphology (21), sea_state (23), human_impact (13)
- **其他**：depth_category (17), vegetation_coverage (15), sea_ice (11), snow_cover (13), ice_cover (23), inundation_frequency (12), canopy_cover (11)

---

## 9. Signature 清单定版

**原则**：每个 signature 字段保留独立 target，不做合并；主 canonical 的代表名（phase4 v3 规则选出）直接作 signature target_name。

### 9.1 Open_ocean Signature (~20 条)

| target | 批次来源 subtype |
|---|---|
| station_number (含 station_no 合并 alias) | sampling_site |
| hole / verbatim_locality / vent_site / pop_up_location | sampling_site |
| time_utc / day_night / event_time / local_date / date_time_utc | time_point (已合入主 collection_date) |
| water_mass / vent_field / vent / biogeographical_province / marine_biome / marine_region | habitat_biome |
| ocean_region / ocean_basin / ocean_and_sea_region / fao_fishing_area / longhurst_province | geo_region |
| h2 / poc_flux / particulate_organic_carbon / total_dissolved_inorganic_carb | nutrient_chemistry |
| mg_ca / nd / oxygen_isotope_composition / fepy_fehr / dFe_concentration / potential_density_anomaly / fehr_fet | trace_chemistry |
| sea_ice_concentration / surface_par / stratification_index | physical_env_driver |
| sea_ice_coverage / sea_ice_cover / sea_ice_conditions | env_state_context |
| sea_ice_thickness / size_fraction_upper_threshold / size_fraction_lower_threshold / bin_size | spatial_metric |
| trench | water_body_descriptor |

### 9.2 Coastal_waters Signature (~15 条)

| target | 批次来源 subtype |
|---|---|
| reef_location | sampling_site |
| beach / reef_type / reef / reef_zone / benthic_cover / habitat_complexity / structural_complexity | habitat_biome |
| shelf_position | geo_region |
| beach_type | material_medium_type |
| ωarag / total_suspended_matter | trace_chemistry |
| wave_energy | physical_env_driver |
| fishing_pressure | env_state_context |
| area_covered / beach_width | spatial_metric |

### 9.3 Lake Signature (~20 条)

| target | 批次来源 subtype |
|---|---|
| lake_name | sampling_site |
| water_residence_time (主清单 residence_time 56 合入) | time_duration |
| lake_region | geo_region |
| permafrost | material_medium_type |
| major_ions / total_salts | trace_chemistry |
| lake_area / lake_size / pond_area / lake_surface_area / river_width / river_length / total_storage_capacity / shoreline_development | spatial_metric |
| lake_type / mixing_type / impervious | env_state_context |
| secci_depth | turbidity_transparency (作 secchi_depth alias 合入主) 或 Lake sig 保留？**需最终确认** |

### 9.4 Wetlands Signature (~15 条)

| target | 批次来源 subtype |
|---|---|
| peat_depth | vertical_position |
| water_table_depth (含 water_table_level alias) | vertical_position |
| intertidal_elevation (含 shore_elevation alias) | vertical_position |
| inundation | vertical_position |
| wetland_type / dominant_vegetation / peatland_type / microhabitat / forest_type / marsh_type / marsh | habitat_biome |
| wetland_area | spatial_metric |
| total_potassium | trace_chemistry |
| peat_type | material_medium_type |
| microtopography / flood_frequency | env_state_context |

---

## 10. 规模汇总

| 清单 | 数量 |
|---|---|
| 主清单主 target | **~66** |
| Open_ocean signature | ~20 |
| Coastal_waters signature | ~15 |
| Lake signature | ~20 |
| Wetlands signature | ~15 |
| **合计 step5 可消费 target** | **~136** |
| R3 剔除（全量） | ~90 |

主清单单 env 可见量：每篇 paper 的 tier1 = 主清单 66 条 + 该 env signature ~15-20 条 ≈ **80-90 条 tier1**（与旧 `step4b_env_extraction_targets.json` 的 40-50 tier1 规模同阶）

---

## 11. Phase 6 实现规范

### 11.1 输入

1. `env5_main_list.csv` (403 rows, bugfix 后)
2. 4 × `env5_signature_{env}.csv` (68/15/34/18 rows)
3. `env4_canonical_to_raw_key.csv` (4132 raw_key → canonical_id)
4. `env3_final_annotations.csv` (4132 raw_key + subtype/qk/bag + raw_key_original)
5. `raw_key_expansion_table.csv` (115 条展开映射)
6. **本文档**的决策（需要编码为 YAML 或嵌入 py）

### 11.2 核心动作

按 subtype 组装：

```python
PHASE6_SCHEMA = {
    "main_list": [
        {"target": "salinity", "subtype": "salinity",
         "aliases": [...], "source_canonical_ids": [...]},
        {"target": "temperature", "subtype": "temperature",
         "aliases": [...]},
        # ... 66 条
    ],
    "signatures": {
        "Open_ocean": [
            {"target": "station_number", ...},
            # ...
        ],
        "Coastal_waters": [...],
        "Lake": [...],
        "Wetlands": [...],
    },
    "excluded": [
        {"raw_key": "...", "reason_code": "R1|R2|R3|R4", "detail": "..."},
        # ...
    ],
}
```

### 11.3 需要解决的 4 个特殊处理

1. **跨 subtype 归属调整**：
   - turbidity_transparency 收 `secchi_disk_depth`（from vertical_position）
   - physical_env_driver 收 `water_level` group（from vertical_position）
   - water_body_descriptor 合入 `pond/inflow/outflow`（from Lake signature）

2. **BOD/COD 强制拆分**（phase4 EDC 未成功）：
   - 遍历 oxygen subtype canonicals 的 member_raw_keys
   - 按字面匹配把含 `chemical_oxygen_demand` / `cod` 的 raw_keys 拆出来作独立 `chemical_oxygen_demand` target

3. **R3 剔除字段清单**嵌入决策 YAML，trace 表记录归类

4. **输出 step5 兼容 JSON**：
   ```json
   {
     "metadata": {"generated_at": "...", "source": "phase6_decisions.md v1"},
     "global_fields": {
       "Universal": [...],     // 主清单核心字段
       "Shared": [...]         // 主清单扩展字段（或留空）
     },
     "per_environment": {
       "Open_ocean": {"fields": [
         {"field": "collection_date", "tier": 1, "aliases": [...]},
         {"field": "station_number", "tier": 2, "aliases": [...]},
         ...
       ]},
       "Coastal_waters": {...},
       "Lake": {...},
       "Wetlands": {...}
     }
   }
   ```
   其中 tier1 = 主清单 target，tier2 = 该 env signature target

### 11.4 输出产物

1. `env_field_pipeline_output/env6_extraction_targets.json` (step5 直接读)
2. `env_field_pipeline_output/env6_main_schema.csv` (主清单 66 条，人审/论文附录用)
3. `env_field_pipeline_output/env6_signature_schema.csv` (4 env signature 合并版)
4. `env_field_pipeline_output/env6_excluded_trace.csv` (所有 R1-R4 剔除字段 + 理由)

### 11.5 审计指标（phase6 必须输出）

- 主清单 target 数
- 4 env signature target 数
- R1/R2/R3/R4 分别剔除数量
- 主清单 PMID 覆盖率（相对于 env4_canonicals 总 pmid）
- 路线 Y 所需 step5 prompt 改动示意（生成一个 sample prompt 片段）

---

## 12. 未完成项（列为 phase6 之后的 TODO）

1. **step5 prompt 改动**：section_extract_v1.txt 加 aliases 展示（路线 Y 要求）
2. **phase4 EDC prompt 升级**：加 BOD/COD 反例后，下一次全流程重跑时 BOD/COD 自动拆开（当前版本还是 phase3 LLM 把 chemical_oxygen_demand 判成 qk=biochemical_oxygen_demand 导致同桶；彻底解决需在 phase3 prompt few-shot 加反例并重跑 phase3）
3. **filter_pore_size 归属确认**：当前在 trace_chemistry，实际是"采样工具粒度分级"。phase6 内部决定归属
4. **MIxS checkpoint 错位修复**：phase4 bugfix 后 canonical_id 变了但 MIxS checkpoint 没清，导致 env5 的 mixs_slot 列信息错位。不影响 phase6（不依赖 MIxS），但 viz 04 mixs_alignment 图不可信，需要清 MIxS checkpoint 重跑 phase4（约 +10-15min）
5. **同位素 Greek 记号 ASCII 化**（可选）：δ13c / ωaragonite 等当前保留 Greek；若需跨工具兼容考虑 ASCII 化（delta_c13 / omega_aragonite）
6. **Lake signature `secci_depth` 最终归属**：作 `secchi_depth` (turbidity_transparency 主) 的 Lake sig 变体 alias，还是 Lake sig 独立 target？本文档默认为后者

---

## 13. 版本历史

| 版本 | 日期 | 变更 |
|---|---|---|
| v1 | 2026-04-24 | 首版，批次 1/2/3 全部决策 + 路线 Y + 3 判据 + R1-R4 规范 |

---

*End of document.*
