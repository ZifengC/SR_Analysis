# PC-SAR Component Internal Validation

`5-Validation/` 用于验证 PC-SAR 内部 component 是否按预期工作。这里关注两类
问题：

1. 完整模型内部的 intent state、uncertainty 和 attribution gate 是否具有可解释
   的行为关系。
2. 移除 intent-state 或 counterfactual-attribution component 后，模型内部行为如何
   改变。

本目录不训练模型。所有模型特征都由上游 `RelSAR/test.py` 导出，再由本目录的
两个分析脚本读取。

## 与其他分析模块的边界

- `2-Problem-identify/`：从真实行为 embedding 中发现 exploration 和
  consolidation 现象。
- `3-Intent/`：验证 latent intent representation 是否呈现这些现象。
- `5-Validation/`：验证产生 intent representation 的模型 component 和内部机制。

因此，`3-Intent` 验证“intent 表征了什么”，本目录验证“模型 component 是否按
预期构造并使用这些 intent”。

## 目录结构

```text
5-Validation/
  Features/
    intermediate/
    intermediate_no_intent/
    intermediate_no_counterfactual/
  Old_features/
  script/
    mechanism_validation.py
    ablation_analysis.py
  output_mechanism/
    event_level/
    state_validation/
    attribution_validation/
  output_ablation/
```

## 输入 CSV 的来源

### Mechanism validation 导出

`mechanism_validation.py` 默认使用 `Features/` 下由 `RelSAR/test.py` 导出的
flattened mechanism feature table：

- `Features/intermediate/pcsar_intent_features_all_full_mechanism.csv`

该 CSV 包含 mechanism validation 所需的 intent posterior、uncertainty、gate、
prediction 和 metadata，但不包含 ablation analysis 所需的 embedding vectors。

`Features/` 还保存两个采用相同 131-column schema 的 component 变体：

| 模型版本 | CSV | 上游配置含义 |
| --- | --- | --- |
| `full` | `Features/intermediate/pcsar_intent_features_all_full_mechanism.csv` | 完整 PC-SAR，intent state 与 counterfactual attribution 均启用 |
| `no_intent_state` | `Features/intermediate_no_intent/pcsar_intent_features_all_full_mechanism_no_intent.csv` | 关闭 intent-state 相关 component |
| `no_counterfactual` | `Features/intermediate_no_counterfactual/pcsar_intent_features_all_full_mechanism_no_counterfactual.csv` | 关闭 counterfactual attribution |

`Features/intermediate/pcsar_user_trajectory.csv` 是同一导出流程产生的用户轨迹
文件，但当前两个验证脚本不直接读取它。

### Ablation validation 导出

`ablation_analysis.py` 当前使用 `Old_features/` 下由较完整 export schema 产生的
三份 test-set CSV：

- `pcsar_intent_features_test_full.csv`
- `pcsar_intent_features_test_no_intent_state.csv`
- `pcsar_intent_features_test_no_counterfactual.csv`

每份 CSV 有 836 columns，除 scalar diagnostics 与 intent posteriors 外，还包含
64 维的：

- `pos_item_emb_*`、`query_emb_*`
- `rec_user_feat_*`、`rec_history_mean_emb_*`
- `shared_user_feat_*`

这些向量是 semantic anchor、state resistance 和 future consistency 计算所必需
的，因此 ablation 不能使用缺少 embedding columns 的新 `Features/` CSV。
脚本会检查候选文件 schema，并自动跳过不满足这些要求的 CSV。

### 关键字段

两个脚本主要使用：

- 标识与顺序：`user_id`、`sample_index`、`timestamp`、`channel`
- 历史构成：`history_rec_share`、`history_src_share`
- intent posterior：`global_pi_*`、`rec_pi_*`、`src_pi_*`
- state diagnostics：`global_dominant_intent_prob`、`global_intent_entropy`、
  `global_posterior_uncertainty`
- attribution：`attribution_source_proxy`、gate 与 confidence 字段
- prediction：rec/src positive score、rank 和 score-gap 字段
- representation：item、history 和 user embedding vectors；若导出 top-k
  embedding 则优先使用，否则代码回退到 user feature

这些字段由 `RelSAR/test.py` 计算和导出，本目录只进行聚合、比较与绘图。

## Script 1: Mechanism Validation

### 目的

`script/mechanism_validation.py` 只读取完整模型：

```text
Features/intermediate/pcsar_intent_features_all_full_mechanism.csv
```

它验证完整 PC-SAR 内部机制，不比较消融版本。

### 计算流程

```text
full sample-level export
  -> 合并为 user event-level intent trajectory
  -> state / uncertainty 与未来 intent dynamics
  -> transition attribution gate 与 intent dominance
  -> output_mechanism/
```

1. 将 sample-level rows 合并为有时间顺序的 search/recommendation events。
2. 使用未来窗口计算 intent consistency 和 dispersion。
3. 将 uncertainty、entropy、confidence 分为 Low/Medium/High 状态。
4. 比较不同状态下的未来行为指标。
5. 构造 `R->R`、`R->S`、`S->R`、`S->S` transition，并验证 attribution gate。

### 输出 CSV 及其来源

| 输出 CSV | 来源与用途 |
| --- | --- |
| `output_mechanism/event_level/model_events.csv` | 从完整模型 sample-level export 合并得到的 event-level 表；同时是 `3-Intent/script/intent_tsne.py` 的输入 |
| `output_mechanism/state_validation/state_validation_summary.csv` | 从 event-level state events 按 state metric 和 Low/Medium/High 分组汇总 |
| `output_mechanism/state_validation/state_validation_high_low_differences.csv` | 由 state summary 进一步计算 High minus Low 差异 |
| `output_mechanism/attribution_validation/transition_summary.csv` | 从相邻 event transitions 按 transition type 汇总 attribution 与 prediction 指标 |
| `output_mechanism/attribution_validation/attribution_intent_dominance_by_target_summary.csv` | 从 transition events 按 target channel 和 intent-dominance 状态汇总 gate |

`model_events.csv` 由 `5-Validation/script/mechanism_validation.py` 生成。
读取者 `3-Intent/script/intent_tsne.py` 读取该文件，用于生成
`3-Intent/output/Part3/` 的 t-SNE 结果。

大型 state-event 和 transition-event 工作表只在内存中计算，不写入 CSV。

### 最终图

- `output_mechanism/state_validation/state_validation_uncertainty.png`
- `output_mechanism/attribution_validation/transition_gate_validation.png`

## Script 2: Component Ablation

### 目的

`script/ablation_analysis.py` 比较三个模型版本：

```text
full vs no_intent_state
full vs no_counterfactual
```

当前三个版本分别来自：

- `Old_features/pcsar_intent_features_test_full.csv`
- `Old_features/pcsar_intent_features_test_no_intent_state.csv`
- `Old_features/pcsar_intent_features_test_no_counterfactual.csv`

- state 分析只比较 `full` 与 `no_intent_state`。
- attribution 分析只比较 `full` 与 `no_counterfactual`。

脚本还读取 `Data/Step4/rec_all.pkl` 和 `Data/Step4/src_all.pkl`，用于构造完整
历史中的 nearest semantic anchors。这些 PKL 来自 Qilin 数据预处理流程，不是
`RelSAR/test.py` 的模型导出。

### 计算流程

```text
three aligned model exports + Data/Step4 histories
  -> 对齐 user_id + sample_index
  -> 在 full 数据上确定共同 state / transition events
  -> 在对应模型版本上重新评估相同事件
  -> component difference summaries
  -> output_ablation/
```

使用共同事件定义可以避免不同模型版本因样本选择不同而产生不可比结果。

### 输出 CSV 及其来源

| 输出 CSV | 来源与用途 |
| --- | --- |
| `output_ablation/state_summary.csv` | 同一批 state-shock events 在 `full` 与 `no_intent_state` 下的聚合比较 |
| `output_ablation/transition_summary.csv` | 同一批 anchor transitions 在 `full` 与 `no_counterfactual` 下按 transition type 聚合 |
| `output_ablation/transition_prediction_score_gap_summary.csv` | 从 transition evaluation 汇总 relevant-source 与 irrelevant-source prediction score，并计算两者的绝对 gap |
| `output_ablation/transition_gap.csv` | 从 transition evaluation 比较方向性差异，例如 `R->S` 与 `S->R` 的 gap |

事件级明细只在内存中使用，不写入 CSV。

### 最终图

- `output_ablation/state_attribution_summary.png`

该图同时展示 intent-state component 与 attribution component 的消融结果。

## 两个脚本的关系

两个脚本彼此独立：

- `mechanism_validation.py` 回答完整模型内部机制是否合理。
- `ablation_analysis.py` 回答移除 component 后结果是否发生预期变化。

它们可以单独运行，没有强制先后顺序。本目录输出的唯一跨目录下游读取关系是：

```text
output_mechanism/event_level/model_events.csv
  -> 3-Intent/script/intent_tsne.py
```

因此，若要重新生成 `3-Intent/output/Part3/`，需要先运行 mechanism validation。

## 运行方式

从 `Analysis/` 根目录运行，使用 Python 3.11：

```bash
cd "/Users/Brodie/Documents/Code-RelSAR/Results&Analysis/Analysis"
```

完整模型机制验证：

```bash
/Users/Brodie/miniconda3/bin/python3 5-Validation/script/mechanism_validation.py
```

component 消融验证：

```bash
/Users/Brodie/miniconda3/bin/python3 5-Validation/script/ablation_analysis.py
```

两个脚本均支持命令行参数覆盖默认输入和输出路径：

```bash
/Users/Brodie/miniconda3/bin/python3 5-Validation/script/mechanism_validation.py --help
/Users/Brodie/miniconda3/bin/python3 5-Validation/script/ablation_analysis.py --help
```

## 输出保留原则

- 保留最终 PNG。
- 保留解释最终图和下游分析所需的小型 summary CSV。
- 不保存可重复生成的大型 event-level working CSV，唯一例外是下游 t-SNE 依赖的
  `model_events.csv`。
