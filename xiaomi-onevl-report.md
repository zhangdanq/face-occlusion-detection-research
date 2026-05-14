# 小米 OneVL 开源模型深度研究报告

> 报告日期：2026年5月14日
> 所有信息均附来源链接，未标注来源的内容为基于官方资料的合理推断

---

## 一、模型概述

| 项目 | 内容 |
|------|------|
| 模型名称 | OneVL (One-Step Latent Reasoning and Planning with Vision-Language Explanations) |
| 开发方 | 小米具身智能团队 (Xiaomi Embodied Intelligence Team) |
| 模型类型 | Vision-Language-Action (VLA) 自动驾驶框架 |
| 基座模型 | Qwen3-VL-4B-Instruct |
| 参数规模 | 4B（主模型），另有5B视觉解码器变体 |
| 开源许可证 | Apache 2.0 |
| 发布时间 | 2026年4月（arXiv论文）/ 2026年5月正式开源 |
| 来源 | [GitHub仓库](https://github.com/xiaomi-research/onevl) |

---

## 二、核心创新

OneVL 解决了一个自动驾驶VLA领域的关键问题：**隐式Chain-of-Thought (CoT) 推理方法虽然快，但性能始终不如显式CoT**。

### 2.1 三种CoT范式对比

| 范式 | 速度 | 可解释性 | 性能 |
|------|------|----------|------|
| (a) 显式CoT | 慢（需生成完整推理链） | 高（语言可读） | 好 |
| (b) 隐式CoT | 快（压缩到潜变量） | 低（黑盒） | 差 |
| (c) OneVL（本文） | 快（等同于answer-only速度） | 高（视觉+语言双模态可解释） | 最好 |

来源：[OneVL项目主页](https://xiaomi-embodied-intelligence.github.io/OneVL/)

### 2.2 核心技术创新

1. **双模态辅助解码器 (Dual-Modal Auxiliary Decoders)**
   - **语言辅助解码器**：从语言潜变量中重建人类可读的CoT推理文本，将潜变量空间锚定在语义意图上（场景解读、目标分析、驾驶决策）
   - **视觉辅助解码器**：从视觉潜变量中预测未来帧的视觉token（t+0.5s和t+1.0s），充当世界模型，将潜变量空间锚定在物理场景动力学上
   - 推理时两个解码器被丢弃，不影响推理速度

2. **紧凑潜变量接口 (Compact Latent Token Interface)**
   - 4个视觉潜变量token + 2个语言潜变量token
   - 使用已有词汇表token，无需新增特殊token
   - 信息瓶颈迫使模型提炼因果结构，而非记忆

3. **预填充推理 (Prefill Inference)**
   - 所有潜变量token在一次并行前向传播中处理
   - 推理延迟等同于answer-only AR预测
   - NAVSIM上比显式CoT快1.5x，ROADWork上快2.3x

来源：[GitHub README](https://github.com/xiaomi-research/onevl)

---

## 三、模型架构

### 3.1 整体架构

```
输入图像 + 文本提示
       ↓
   Qwen3-VL-4B (主VLM)
       ↓
  ┌────┴────┐
  ↓         ↓
视觉潜变量  语言潜变量
(4 token)  (2 token)
  ↓         ↓
视觉辅助   语言辅助
解码器     解码器
  ↓         ↓
未来帧     CoT推理
(t+0.5s,   文本
 t+1.0s)
```

- **视觉tokenizer**：使用Emu3.5 IBQ VQ-VAE（131k码本）
- **训练时**：双解码器同时监督
- **推理时**：解码器丢弃，潜变量token预填充，仅轨迹自回归生成

### 3.2 三阶段训练流程

| 阶段 | 名称 | 操作 |
|------|------|------|
| Stage 0 | Main Model Warmup | 端到端训练主VLM，嵌入潜变量token，学习路由 |
| Stage 1 | Auxiliary Decoder Warmup | 冻结主模型，训练辅助解码器对齐潜变量表示 |
| Stage 2 | Joint End-to-End Fine-tuning | 联合微调所有组件，双向梯度优化瓶颈 |

来源：[OneVL项目主页 - Training Pipeline](https://xiaomi-embodied-intelligence.github.io/OneVL/)

---

## 四、性能基准测试

### 4.1 NAVSIM（轨迹预测）

| 方法 | 模型大小 | PDM-score ↑ | 延迟(s) ↓ | 可解释性 |
|------|:--------:|:-----------:|:---------:|:--------:|
| AdaThinkDrive | 8B | 86.20 | — | 语言 |
| LaST-VLA | 8B | 87.30 | — | — |
| AR Answer | 4B | 87.47 | 4.49 | — |
| AR CoT+Answer | 4B | 88.29 | 6.58 | 语言 |
| COCONUT | 4B | 84.84 | 5.93 | — |
| CODI | 4B | 83.92 | 8.62 | — |
| SIM-CoT | 4B | 84.21 | 10.86 | 语言 |
| **OneVL** | **4B** | **88.84** | **4.46** | **视觉+语言** |

来源：[GitHub README - Results](https://github.com/xiaomi-research/onevl)

### 4.2 ROADWork（施工区域导航）

| 方法 | ADE(px) ↓ | FDE(px) ↓ | 延迟(s) ↓ | 可解释性 |
|------|:---------:|:---------:|:---------:|:--------:|
| YNet | 22.68 | 80.78 | — | — |
| AR Answer | 15.98 | 40.29 | 4.74 | — |
| AR CoT+Answer | 13.18 | 29.98 | 10.74 | 语言 |
| COCONUT | 15.44 | 38.60 | 6.06 | — |
| CODI | 16.45 | 44.28 | 6.73 | — |
| SIM-CoT | 16.49 | 44.32 | 6.19 | 语言 |
| **OneVL** | **12.49** | **28.80** | **4.71** | **视觉+语言** |

### 4.3 Impromptu

| 方法 | ADE(m) ↓ | FDE(m) ↓ | 延迟(s) ↓ | 可解释性 |
|------|:--------:|:--------:|:---------:|:--------:|
| Impromptu VLA | 1.60 | 4.28 | 6.10 | — |
| AR Answer | 1.46 | 4.03 | 4.24 | — |
| AR CoT+Answer | 1.42 | 3.96 | 6.84 | 语言 |
| COCONUT | 1.49 | 4.07 | 5.27 | — |
| CODI | 1.86 | 5.18 | 5.24 | — |
| SIM-CoT | 2.43 | 6.10 | 5.09 | 语言 |
| **OneVL** | **1.34** | **3.70** | **4.02** | **视觉+语言** |

### 4.4 Alpamayo-R1 (APR1)

| 方法 | ADE(m) ↓ | FDE(m) ↓ | 延迟(s) ↓ | 可解释性 |
|------|:--------:|:--------:|:---------:|:--------:|
| Cosmos-Reason | 2.86 | 7.42 | — | 语言 |
| AR Answer | 3.27 | 9.59 | 3.06 | — |
| AR CoT+Answer | 2.99 | 8.54 | 3.51 | 语言 |
| COCONUT | 3.29 | 9.48 | 3.76 | — |
| CODI | 3.22 | 9.25 | 3.85 | — |
| SIM-CoT | 3.40 | 9.85 | 3.78 | 语言 |
| **OneVL** | **2.62** | 7.53 | **3.26** | **视觉+语言** |

### 4.5 面向实际部署的MLP变体

通过在Qwen3-VL backbone上附加紧凑的MLP head，OneVL可以在单次前向传播中预测轨迹：

| 指标 | 数值 |
|------|------|
| MLP延迟 | 0.24s (4.16 Hz) |
| PDM-score | 86.83 |
| 占AR延迟比例 | 5.4% |

### 4.6 消融实验（NAVSIM PDM-score）

| 模型变体 | 语言解码器 | 视觉解码器 | 分阶段训练 | PDM-score |
|----------|:----------:|:----------:|:----------:|:---------:|
| OneVL w/o 视觉解码器 | ✓ | — | ✓ | 87.97 |
| OneVL w/o 语言解码器 | — | ✓ | ✓ | 88.53 |
| OneVL w/o 分阶段训练 | ✓ | ✓ | — | 67.13 |
| **OneVL（完整）** | **✓** | **✓** | **✓** | **88.84** |

关键发现：分阶段训练至关重要，去掉后性能从88.84暴跌至67.13。

---

## 五、开源资源

### 5.1 模型权重（HuggingFace）

共6个模型权重：

| 模型 | 参数量 | 用途 |
|------|:------:|------|
| xiaomi-research/OneVL_NAVSIM | 14B | NAVSIM基准 |
| xiaomi-research/OneVL_ROADWork | 14B | ROADWork基准 |
| xiaomi-research/OneVL_Impromptu | 14B | Impromptu基准 |
| xiaomi-research/OneVL_AlpamayoR1 | 14B | Alpamayo-R1基准 |
| xiaomi-research/OneVL_visual_decoder_pt | 5B | 视觉解码器预训练 |
| xiaomi-research/OneVL_visual_decoder_pt_ar1 | 5B | 视觉解码器预训练(AR1) |

来源：[HuggingFace模型列表](https://huggingface.co/models?search=xiaomi+OneVL)

### 5.2 代码仓库

| 资源 | 链接 | Star数 |
|------|------|--------|
| 推理代码 | https://github.com/xiaomi-research/onevl | 238 |
| 训练代码 | https://github.com/GeorgeLuImmortal/OneVL_training | — |

### 5.3 论文

- **arXiv**: https://arxiv.org/abs/2604.18486
- **项目主页**: https://xiaomi-embodied-intelligence.github.io/OneVL/

---

## 六、技术依赖

| 组件 | 来源 | 说明 |
|------|------|------|
| 基座VLM | Qwen3-VL-4B-Instruct | 阿里通义千问视觉语言模型 |
| 视觉Tokenizer | BAAI/Emu3.5-VisionTokenizer | 131k码本IBQ VQ-VAE |
| CoT标注 | AdaThinkDrive | NAVSIM CoT标注来源 |
| 评估基准 | NAVSIM, ROADWork, Impromptu, Alpamayo-R1 | 四个自动驾驶benchmark |

运行要求：
- Python 3.10+
- CUDA GPU（推荐≥16GB VRAM）
- transformers ≥ 4.57.0

---

## 七、核心论文作者

```
Lu, Jinghui; Guan, Jiayi; Huang, Zhijian; Li, Jinlong; 
Li, Guang; Kong, Lingdong; Li, Yingyan; Wang, Han; 
Xu, Shaoqing; Luo, Yuechen and others
```

来源：[arXiv:2604.18486](https://arxiv.org/abs/2604.18486)

---

## 八、行业意义与评价

### 8.1 技术突破

1. **首个超越显式CoT的隐式方法**：此前所有隐式CoT方法（COCONUT、CODI、SIM-CoT）在驾驶任务上甚至不如简单的AR Answer基线，OneVL是第一个突破这一瓶颈的
2. **双模态可解释性**：同时提供语言推理（"为什么这样驾驶"）和视觉预测（"接下来场景会怎样"），这是之前方法无法实现的
3. **推理速度革命**：预填充推理使延迟等同answer-only，同时MLP变体可达4.16Hz实时部署

### 8.2 媒体评价

| 媒体 | 评价 |
|------|------|
| Gizmochina | "小米开源OneVL，统一VLA、世界模型和潜变量推理三大范式" |
| ChinaDailyBrief | "雷军的开源赌注：挑战特斯拉的AI霸权" |

来源：[Yahoo搜索结果](https://search.yahoo.com/search?p=xiaomi+OneVL+autonomous+driving+open+source)

### 8.3 与竞品对比

| 维度 | OneVL | 其他隐式CoT方法 | 显式CoT方法 |
|------|-------|-----------------|-------------|
| 性能 | SOTA | 低于AR Answer | 好 |
| 速度 | answer-only级 | 略快于显式CoT | 慢 |
| 可解释性 | 视觉+语言 | 无 | 仅语言 |
| 模型大小 | 4B | 4B | 4-8B |

---

## 九、应用前景

1. **自动驾驶**：核心应用场景，轨迹预测+可解释性+实时性三位一体
2. **具身智能**：VLA框架可扩展到机器人操控、导航等任务
3. **世界模型研究**：视觉辅助解码器的未来帧预测能力独立于主任务，具有研究价值
4. **实时推理系统**：MLP变体0.24s延迟证明了从研究到部署的可行性

---

## 十、局限性与挑战

1. **依赖闭源数据**：NAVSIM、ROADWork等基准的训练数据不完全公开
2. **模型规模限制**：目前仅开源4B版本，更大规模的性能未知
3. **视觉解码器额外开销**：训练时需要额外的Emu3.5-VisionTokenizer
4. **领域专用**：目前仅针对自动驾驶场景，泛化到其他VLA任务有待验证

---

## 十一、引用

```bibtex
@article{lu2026onevl,
  title={OneVL: One-Step Latent Reasoning and Planning with Vision-Language Explanation},
  author={Lu, Jinghui and Guan, Jiayi and Huang, Zhijian and Li, Jinlong and Li, Guang and Kong, Lingdong and Li, Yingyan and Wang, Han and Xu, Shaoqing and Luo, Yuechen and others},
  journal={arXiv preprint arXiv:2604.18486},
  year={2026},
  url={https://arxiv.org/abs/2604.18486}
}
```

---

## 参考来源汇总

1. GitHub仓库：https://github.com/xiaomi-research/onevl
2. 项目主页：https://xiaomi-embodied-intelligence.github.io/OneVL/
3. arXiv论文：https://arxiv.org/abs/2604.18486
4. HuggingFace模型：https://huggingface.co/collections/xiaomi-research/onevl-models/
5. HuggingFace模型列表：https://huggingface.co/models?search=xiaomi+OneVL
6. 训练代码：https://github.com/GeorgeLuImmortal/OneVL_training
7. Qwen3-VL：https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct
8. Emu3.5-VisionTokenizer：https://huggingface.co/BAAI/Emu3.5-VisionTokenizer
9. Gizmochina报道：https://www.gizmochina.com/2026/05/14/xiaomi-announces
10. ChinaDailyBrief报道：https://chinadailybrief.com/article/6a04aca381ad73ba8cd02968
