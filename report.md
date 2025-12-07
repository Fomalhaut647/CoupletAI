### 项目与改造目标概述

- **原始项目框架**：  
  - 任务：给定上联，生成下联。  
  - 原实现：利用“上下联等长”这一特点，将问题转成**序列标注**：输入上联序列 `x₁…xₙ`，输出与之等长的标签序列 `y₁…yₙ`（下联），模型一次性输出整句，无自回归解码。
- **本次目标**：  
  - 在不立刻重写成完整 encoder–decoder 的前提下，先引入**自回归生成机制**，改造成类似 LLM 的“单流条件语言模型”：  
    - 序列形式：`上联 + [SEP] + 下联`  
    - 训练时做 **next-token 预测**，只对“下联部分”计算损失；  
    - 推理时从 `上联 + [SEP]` 出发，自回归生成等长下联。  
  - 同时保留原始序列标注模式，便于对比与回退。

---

### 词表与 Tokenizer 改动（`preprocess/module/tokenizer.py`）

- **特殊符号设计**  
  - 预置特殊 token 映射：
    - `"[PAD]"`: 0  
    - `"[UNK]"`: 1  
    - `"[SEP]"`: 2  （上下联分隔符）
- **关键属性与接口**  
  - `pad_id` / `unk_id` / `sep_id`：提供常用特殊 token 的 ID 访问。  
  - `build(vocab_file)`：在上述初始表基础上，把数据集中的字/符号依次加入 `token_to_ix`。  
  - `encode` / `decode` 与原始实现保持一致：  
    - `encode(sent)`：将句子按**单字切分**编码为 `List[int]`。  
    - `decode(ids)`：将 ID 序列（忽略 PAD）还原为字符串。

**作用**：  
为新框架提供统一的 `[SEP]` 分隔符，并保证该 token 在所有阶段（预处理、训练、推理）ID 固定。

---

### 数据预处理改造（`preprocess/preprocess.py`）

#### 1. 样本表示结构

- `CoupletExample`：保存原始 token 级上下联：  
  - `seq: List[str]`（上联）  
  - `tag: List[str]`（下联）
- `CoupletFeatures`：保存 ID 级上下联：  
  - `input_ids: List[int]`（上联）  
  - `target_ids: List[int]`（下联）

#### 2. features 构造

```python
for example in examples:
    seq_ids = tokenizer.convert_tokens_to_ids(example.seq)   # 上联
    tag_ids = tokenizer.convert_tokens_to_ids(example.tag)   # 下联
    features.append(CoupletFeatures(seq_ids, tag_ids))
```

#### 3. 两种张量化模式

##### （1）原始序列标注模式 `convert_features_to_tensors`

- 输入：`features, tokenizer, max_seq_len`
- 输出四个张量：`(input_ids, masks, lens, target_ids)`  
  - `input_ids`：形状 `(N, max_seq_len)`，上联，右侧 PAD。  
  - `target_ids`：形状 `(N, max_seq_len)`，下联，右侧 PAD。  
  - `masks`：同形状，`0` 表示有效 token，`1` 表示 PAD。  
  - `lens`：上联真实长度（截断到 `max_seq_len`）。

> 这一部分保持原逻辑，用于 `mode="tagger"` 训练。

##### （2）单流自回归模式 `convert_features_to_tensors_concat`

- 目标：构造“上联 + [SEP] + 下联”的单流序列，并做 next-token 预测。  
- 定义最大长度：  
  - `max_total_len = max_seq_len * 2 + 1`（上联 `max_seq_len` + `[SEP]` + 下联 `max_seq_len`）  
  - 实际训练使用长度 `seq_len = max_total_len - 1`，因为输入与目标是 `concat[:-1]` 和 `concat[1:]`。
- 对每个样本：
  1. 截断上联、下联：  
     - `src = f.input_ids[:max_seq_len]`  
     - `tgt = f.target_ids[:max_seq_len]`
  2. 拼接单流序列：  
     - `concat = src + [sep_id] + tgt`，再截断到 `max_total_len`。
  3. 构造自回归输入/目标：  
     - `inp = concat[:-1]`  
     - `tar = concat[1:]`  
     - 实际长度 `real_len = min(len(inp), seq_len)`
  4. 写入张量：
     - `input_ids[i, :real_len] = inp[:real_len]`  
     - `target_ids[i, :real_len] = tar[:real_len]`  
     - `attn_mask[i, :real_len] = 0`（有效），其余为 `1`（PAD）。
  5. 构造 `loss_mask`：只对**下联部分**（即 `[SEP]` 之后的位置）计算 loss：
     - 拼接序列下标：`concat` 中  
       - `src` 区间：[0, len(src)-1]  
       - `[SEP]` 位置：`len(src)`  
       - 下联首 token：`len(src) + 1`
     - `tar[j]` 对应 `concat[j+1]`，因此：  
       - `token_pos_in_concat = j + 1`  
       - 当 `token_pos_in_concat >= len(src) + 1` 时，该位置属于“预测下联”，`loss_mask[i, j] = 1`。

- 最终返回四元组：  
  - `input_ids: (N, seq_len)`  
  - `attn_mask: (N, seq_len)`  
  - `loss_mask: (N, seq_len)`  
  - `target_ids: (N, seq_len)`

#### 4. 数据集创建与命令行参数

- `create_dataset(fdir, tokenizer, max_seq_len, mode)`：  
  - `mode == "tagger"`：调用 `convert_features_to_tensors`；  
  - `mode == "concat"`：调用 `convert_features_to_tensors_concat`；  
  - 最后用 `TensorDataset(*tensors)` 封装。
- CLI 新增参数：  
  - `--mode/-m`：`"tagger"` 或 `"concat"`，控制生成哪种数据格式。

> 这一步让**同一套代码**可以产出两种数据：旧模式与自回归模式。

---

### 训练脚本改造（`preprocess/main.py`）

#### 1. 模式开关与模型 max_seq_len 调整

- CLI 新增参数 `--mode/-m`：  
  - `"tagger"`：保持原训练逻辑；  
  - `"concat"`：走单流自回归训练。
- 在初始化模型前增加逻辑：
  - `if args.mode == "concat": args.max_seq_len = args.max_seq_len * 2`  
  - 对 Transformer 来说，这相当于把 `pos_embedding` 的最大位置扩到约两倍，以覆盖 `上联 + [SEP] + 下联` 的长度。

#### 2. concat 模式损失函数 `compute_concat_loss`

- 输入：`logits, targets, loss_mask, pad_id`
  - `logits`: `(B, L, V)`  
  - `targets`: `(B, L)`  
  - `loss_mask`: `(B, L)`，1 表示该位置计入损失。
- 步骤：
  1. 使用 `CrossEntropyLoss(reduction="none", ignore_index=pad_id)` 对每个位置计算 CE：  
     - `per_token = CE(logits.view(-1, V), targets.view(-1)).view_as(targets)`
  2. 利用 `loss_mask` 屏蔽掉上联和 `[SEP]`：  
     - `masked = per_token * loss_mask`
  3. 用有效 token 数归一化：  
     - `denom = loss_mask.sum().clamp(min=1)`  
     - 返回 `masked.sum() / denom`。

**意义**：模型虽然在整个序列上做 next-token 预测，但我们只对“预测下联的 token”积累梯度，实现条件语言建模。

#### 3. 训练循环分支

- 读取 batch 后统一搬到 device：
  ```python
  batch = tuple(t.to(device) for t in batch)
  ```
- `mode == "tagger"`：
  - `input_ids, masks, lens, target_ids = batch`  
  - `logits = model(input_ids, masks)`  
  - `loss = CrossEntropy(logits.view(-1, vocab), target_ids.view(-1))`
- `mode == "concat"`：
  - `input_ids, masks, loss_mask, target_ids = batch`  
  - `logits = model(input_ids, masks)`  
  - `loss = compute_concat_loss(logits, target_ids, loss_mask, tokenizer.pad_id)`

其他部分（优化器、scheduler、梯度裁剪等）保持不变。

---

### concat 模式的自动评估与 Demo（`preprocess/main.py`）

#### 1. 自回归生成函数 `concat_generate`

- 输入：上联的 token id 列表 `upper_ids`。
- 过程：
  1. 构造上下文：`ctx = upper_ids + [sep_id]`。  
  2. 生成步数：`max_len = len(upper_ids)`（与上联等长）。  
  3. 循环 `max_len` 次：
     - 构建当前输入 `inp = ctx`，`mask` 全 0（均为有效）。  
     - 前向：`logits = model(inp, mask)`。  
     - 当前步预测：取最后位置 `logits[0, len(ctx)-1]` 的 `argmax` 作为下一个 token。  
     - 把该 token 追加到 `ctx`。
  4. 最终下联部分为 `[SEP]` 之后的段：`lower_ids = ctx[len(upper_ids)+1:]`，去掉 `PAD` 后解码。

#### 2. concat 自动评估 `auto_evaluate_concat`

- 针对 `test_loader` 的每个 batch：
  1. 取 `input_ids, masks, target_ids`，搬到 GPU。  
  2. 对 batch 中每个样本：
     - `valid_len = (masks == 0).sum()`，找出有效长度。  
     - `seq = input_ids[:valid_len]`，`tar = target_ids[:valid_len]`。  
     - 找到 `sep_idx = seq.index(sep_id)`：  
       - 上联 ID：`upper_ids = seq[:sep_idx]`  
       - 目标下联（gold）：`gold_lower = tar[sep_idx:valid_len]`，过滤 PAD。
     - 用 `concat_generate` 得到预测下联 `pred_lower`。  
     - 若两者非空，计算 BLEU 和 ROUGE-L 并累加。
  3. 返回所有样本平均的 BLEU / ROUGE-L；若无有效样本，则返回 0。

#### 3. concat Demo `predict_demos_concat`

- 与原 `predict_demos` 用同一组固定上联。  
- 对每个上联：
  - `upper_ids = tokenizer.convert_tokens_to_ids(list(sent))`  
  - 调用 `concat_generate` 得到下联 ID，再 decode 为字符串输出。

#### 4. 训练时的评估分支

- `mode == "tagger"`：  
  - 调用原 `predict_demos` 和 `auto_evaluate`，输出 BLEU / Rouge-L。
- `mode == "concat"`：  
  - 调用 `predict_demos_concat` 和 `auto_evaluate_concat`，输出 `[concat] BLEU` / `Rouge-L`。

这样就保证两种模式在训练过程都有**定性样例展示**和**定量指标监控**。

---

### 命令行 Demo 适配（`preprocess/clidemo.py`）

- 新增参数 `--mode/-m`：`"tagger"` 或 `"concat"`，与训练和预处理保持一致。
- `mode == "tagger"`：
  - 沿用旧逻辑：上联编码成 `input_ids`，一次前向预测整句下联（argmax），再 decode。
- `mode == "concat"`：
  - 与训练时相同的生成逻辑：  
    - `upper_ids = tokenizer.convert_tokens_to_ids(list(upper))`  
    - `ctx = upper_ids + [sep_id]`，循环生成 `len(upper_ids)` 个 token。  
    - 输出 `[SEP]` 之后的部分作为下联。

---

### 小结与实验建议

- **框架变化本质**：  
  - 从“给定上联，整句标注下联（非自回归）”，扩展为“给定 `上联 + [SEP]`，自回归生成下联”的**单流条件语言模型**，引入了明确的解码时间依赖。  
  - 通过 `loss_mask`，模型只在“预测下联”位置上学习，兼顾了条件信息与语言流畅性。
- **实现特点**：  
  - 通过 `mode` 参数在同一代码中兼容两种训练/推理路径；  
  - 保持原有模型结构（如 Transformer 编码器），主要修改在数据与训练逻辑层；  
  - 训练/评估/CLI 三处逻辑统一，便于实验与展示。
- **可在报告中补充的实验内容**（需要你实际跑完再填数字）：  
  - 不同模式（tagger vs concat）的 BLEU/ROUGE 曲线、收敛速度对比；  
  - 对同一上联的生成示例对比（是否更加多样/自然）；  
  - 不同 `max_seq_len`、模型结构（如 GRU / 多层 RNN）对 concat 模式效果的影响。

这份整理可以直接作为报告中“方法与实现”章节的主体内容，根据需要再补上实验配置、参数表和结果图即可。