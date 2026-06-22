# RelSAR Analysis

本目录是 RelSAR / PC-SAR 项目的分析层，不负责模型训练。它读取处理后的
Qilin 搜索推荐行为数据，以及上游 `RelSAR/test.py` 导出的模型特征，完成现象
发现、intent 内部验证、案例筛选和 component 内部验证。

## 研究主线

```text
2-Problem-identify
  从真实搜索与推荐行为中发现 exploration、consolidation 和跨渠道转移现象
        |
        v
3-Intent
  验证 PC-SAR 学到的 latent intent 是否呈现这些内部结构
        |
        v
4-SelectSample
  找出适合人工检查 intent 变化的用户轨迹案例
        |
        v
5-Validation
  验证产生这些 intent 表征的模型 component 和消融效果
```

`1-Compare-baseline/` 是独立的排序基线比较模块，不属于上述 2-5 的主要验证
链路。

## 当前目录结构

```text
Analysis/
  Data/                         # 本地处理数据，Git 忽略
    Step1/
    Step4/
    dataset/
    vocab/

  1-Compare-baseline/           # TrustSAR 与 UNISAR 排序比较
    Code/
    New_results/
    Old_results/

  2-Problem-identify/           # 发现用户行为现象
    scripts/
    figures/
    cache/                      # 可重建缓存，Git 忽略
    intermediate/               # 可重建中间结果，Git 忽略

  3-Intent/                     # latent intent 内部验证
    script/
      intent_pattern_analysis.py
      intent_tsne.py
    output/
      Part1/
      Part2/
      Part3/
      AuxiliaryAnalysis/

  4-SelectSample/               # intent 变化案例筛选
    select_sample.py
    output/

  5-Validation/                 # PC-SAR component 内部验证
    Features/
    Old_features/
    script/
      mechanism_validation.py
      ablation_analysis.py
    output_mechanism/
    output_ablation/
```

## 数据与上游关系

### `Data/`

`Data/` 保存已经重编号、分词、切分并转换为模型可读格式的 Qilin 数据，不是
原始日志。

- `src` / `S`：搜索行为，文本表示是解码后的 query。
- `rec` / `R`：推荐行为，文本表示是被点击内容的 caption。
- `Data/Step1/`：词表映射、item 和 user 特征、曝光数据。
- `Data/Step4/`：完整搜索和推荐事件表。
- `Data/dataset/`：train、validation、test 样本。
- `Data/vocab/`：用于恢复 query、item、user 等标识的词表。

### `RelSAR/`

上游 `RelSAR` 负责模型训练、测试和特征导出。本目录只消费这些结果：

```text
RelSAR/test.py
  -> PC-SAR intent、uncertainty、attribution、prediction 和 embedding 特征
  -> 3-Intent/ 与 5-Validation/
```

## 1. Baseline Comparison

### 目的

比较 TrustSAR 与固定 UNISAR baseline 的排序差异，回答模型把哪些样本提升或
降低，以及这些变化与用户近期行为相似度的关系。

### 主要流程

1. 解码 TrustSAR 与 UNISAR 排序结果。
2. 计算 `rank_diff = rank_trust - rank_unisar`。
3. 导出发生明显排序变化的用户时间线。
4. 按字符级 TF-IDF 相似度统计 rank difference。

### 代码与结果

- `1-Compare-baseline/Code/2-compare_ranks.py`
- `1-Compare-baseline/Code/3-export_timelines_txt.py`
- `1-Compare-baseline/Code/4-similarity_bucket_analysis.py`
- `1-Compare-baseline/New_results/`

详细说明见 `1-Compare-baseline/README.md`。

## 2. Phenomenon Discovery

### 定位

`2-Problem-identify/` 用行为文本 embedding 发现现象，不依赖 PC-SAR 的 latent
intent。它建立后续 intent 和 component 验证所要解释的经验事实。

### 核心问题

- 当前行为相对近期历史有多新，即 exploration 程度。
- 当前行为是否与用户近期未来一致。
- 连续停留在相近语义区域是否形成 preference consolidation。
- `R->R`、`R->S`、`S->R`、`S->S` 是否呈现不同的行为模式。
- 当前探索是否被未来行为采纳，还是只是短暂偏移。

### 主要输出

`2-Problem-identify/figures/` 当前包含：

- exploration 与 future consistency / dispersion
- consolidation quadrant
- transition-stratified exploration curves
- run length 与 consolidation / semantic radius

缓存和中间 embedding 位于 `cache/` 与 `intermediate/`，缺失时可由脚本重建。
详细定义和运行顺序见 `2-Problem-identify/README.md`。

## 3. Intent Internal Validation

### 定位

`3-Intent/` 验证 PC-SAR 学到的 latent intent 是否真实反映第 2 部分发现的行为
结构。这里关注的是 intent representation 本身，不做 component 消融。

### 主结果

- `output/Part1/`：exploration 与 intent ambiguity。
- `output/Part2/`：不同搜索推荐 transition 的 intent shift。
- `output/Part3/`：global intent posterior 的 t-SNE 结构。

### 辅助分析

- `output/AuxiliaryAnalysis/Part1/`：intent future consistency 与 expansion。
- `output/AuxiliaryAnalysis/Part2/`：stable run、intent consolidation 与 dispersion。

### 代码

- `script/intent_pattern_analysis.py`
- `script/intent_tsne.py`

其中 t-SNE 使用
`5-Validation/output_mechanism/event_level/model_events.csv` 作为 event-level 输入。
详细说明见 `3-Intent/README.md`。

## 4. Intent-Change Sample Selection

### 定位

`4-SelectSample/` 从用户时间线中筛选适合观察 intent 变化的案例，用于把第 2、
3 部分的统计现象对应到可人工阅读的真实轨迹。

### 筛选逻辑

- 同时包含 recommendation 与 search 行为。
- 具有足够的事件数量。
- 在 `rec` 和 `src` 之间多次切换。
- search session 合并为一条记录，避免同一 query 重复展示。

### 输出

- `output/selected_samples.txt`：中文可读时间线。
- `output/selected_samples.csv`：用户级筛选指标和排序分数。

脚本为 `4-SelectSample/select_sample.py`。详细参数见
`4-SelectSample/README.md`。

## 5. Component Internal Validation

### 定位

`5-Validation/` 验证 PC-SAR 内部 component 是否按预期工作，并通过消融确认
intent state 与 counterfactual attribution 的贡献。它验证的是模型机制，不是
第 3 部分的 intent representation 可视化。

### Mechanism validation

`script/mechanism_validation.py` 只分析完整模型：

- 构建 event-level intent trajectory。
- 验证 uncertainty / state 与未来偏好动态的关系。
- 验证跨渠道 attribution gate 与 intent dominance 的关系。
- 输出到 `output_mechanism/`。

主要图：

- `state_validation/state_validation_uncertainty.png`
- `attribution_validation/transition_gate_validation.png`

其中 `event_level/model_events.csv` 会继续提供给 `3-Intent` 的 t-SNE。

### Ablation validation

`script/ablation_analysis.py` 比较：

- `full`
- `no_intent_state`
- `no_counterfactual`

结果写入 `output_ablation/`。最终图为：

- `state_attribution_summary.png`

目录中的四个 CSV 是紧凑的中间汇总表，不保存可重复生成的大型事件明细。
详细输入、输出和运行命令见 `5-Validation/README.md`。

## 整体数据流

```text
Data/Step1 + Data/Step4
  -> 2-Problem-identify/              发现行为现象
  -> 4-SelectSample/                  筛选可解释案例

RelSAR/test.py exports
  -> 3-Intent/                        验证 latent intent 表征
  -> 5-Validation/output_mechanism/   验证完整模型内部机制
  -> 5-Validation/output_ablation/    验证 component 消融效果

5-Validation/output_mechanism/event_level/model_events.csv
  -> 3-Intent/output/Part3/           intent posterior t-SNE
```

## 运行环境

- Python 3.11
- 项目命令默认从 `Analysis/` 根目录运行。
- `Data/`、CSV、PKL、TXT、缓存和本地环境文件不会被 Git 跟踪。
- PNG 分析图允许被 Git 跟踪。

各模块的完整运行命令、参数和指标定义以对应目录内的 `README.md` 为准。
