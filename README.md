# 面部遮挡检测 (Face Occlusion Detection) 技术调研

> 针对单张静态图片的人脸面部遮挡检测方案调研报告

## 一、问题定义

面部遮挡检测的目标是：对单张人脸图片，判断面部是否存在遮挡，以及遮挡的位置和类型。常见遮挡物包括：

- 口罩 (face mask)
- 墨镜/眼镜 (sunglasses/glasses)
- 围巾 (scarf)
- 帽子 (hat/cap)
- 手部遮挡 (hand occlusion)
- 头发遮挡 (hair occlusion)
- 其他物体遮挡

## 二、主流技术方案

### 方案1: 二分类判别 (Occluded vs. Not Occluded)

- **思路**: 用 CNN 分类器直接判断面部是否被遮挡
- **骨干网络**: ResNet / MobileNet / EfficientNet
- **优点**: 简单快速，适合实时场景
- **缺点**: 无法定位遮挡区域，无法区分遮挡类型
- **典型实现**: 标准图像分类 pipeline

### 方案2: 多类别遮挡类型分类

- **思路**: 将遮挡分为多个类别（无遮挡/口罩/墨镜/围巾等）
- **骨干网络**: 同上，输出层改为多分类
- **优点**: 能区分遮挡物类型
- **缺点**: 无法精确定位遮挡区域像素
- **应用**: 人脸识别前置判断，决定是否需要特殊处理

### 方案3: 人脸解析 (Face Parsing) ⭐推荐

- **思路**: 语义分割，对人脸每个像素进行分类（皮肤/眼睛/鼻子/嘴巴/头发/背景/遮挡物等）
- **核心模型**: BiSeNet / BiSeNetV2 / PIDNet
- **优点**: 像素级精确定位，可知道哪些面部区域被遮挡
- **缺点**: 计算量相对较大
- **代表项目**:
  - [zllrunning/face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch) (2.6k stars) — 基于 BiSeNet 的人脸解析 PyTorch 实现
  - [yakhyo/face-parsing](https://github.com/yakhyo/face-parsing) (276 stars) — 支持 PyTorch + ONNX Runtime 推理

### 方案4: 人脸属性识别 (Face Attribute Recognition)

- **思路**: 一次推理同时预测多个人脸属性（年龄/性别/是否戴眼镜/是否戴口罩/是否遮挡等）
- **代表项目**:
  - [InsightFace](https://github.com/deepinsight/insightface) (28.6k stars) — 包含 attribute 模块和 parsing 模块
  - [kby-ai/FaceAttribute-iOS](https://github.com/kby-ai/FaceAttribute-iOS) / [Android](https://github.com/kby-ai/FaceAttribute-Android) / [Flutter](https://github.com/kby-ai/FaceAttribute-Flutter) — 商业级方案，含活体检测+属性分析

### 方案5: 目标检测 (针对特定遮挡物如口罩)

- **思路**: 用 YOLO/Faster-RCNN 等检测口罩等遮挡物的位置
- **代表项目**:
  - [chandrikadeb7/Face-Mask-Detection](https://github.com/chandrikadeb7/Face-Mask-Detection) (1.6k stars) — 基于 OpenCV 和 Tensorflow/Keras 的口罩检测
  - [Spidy20/face_mask_detection](https://github.com/Spidy20/face_mask_detection) (199 stars)
- **优点**: 能给出遮挡物的 bounding box
- **缺点**: 需要针对每种遮挡物单独训练

## 三、关键数据集

| 数据集名称 | 特点 | 遮挡类型 | 来源 |
|---|---|---|---|
| CelebA / CelebA-HQ | 200K+ 图片，40 属性标注 | 有 Eyeglasses 属性 | [link](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) |
| MAFA | 35K 图片，专门的遮挡人脸数据集 | 口罩为主 | [link](https://sites.google.com/site/zfzhai/) |
| WIDER FACE | 32K 图片，有 occlusion 属性标注 | 各种遮挡 | [link](http://shuoyang1213.me/WIDERFACE/) |
| AR Face Database | 126 人，每人 26 张图 | 墨镜 + 围巾 | [link](https://www2.ece.ohio-state.edu/~aleix/ARdatabase.html) |
| CelebAMask-HQ | 30K 高质量图片，像素级标注 | 人脸解析用 | [link](https://github.com/switchablenorms/CelebAMask-HQ) |
| FaceOcc / FaceOcc2 | 经典遮挡测试集 | 随机遮挡块 | 常用人脸检测 benchmark |

## 四、工程流水线

```
输入图片
  ↓
人脸检测 (RetinaFace / MTCNN / SCRFD)
  ↓
人脸对齐 (Face Alignment)
  ↓
┌──────────────────────────────────────┐
│ 遮挡检测 (选以下一种或组合)          │
│                                      │
│  A. Face Parsing → 统计各区域像素     │
│     判断眼睛/嘴巴/鼻子区域被遮挡比例  │
│                                      │
│  B. 分类模型 → 直接输出遮挡类型       │
│                                      │
│  C. 目标检测 → 检测口罩/墨镜等物体    │
└──────────────────────────────────────┘
  ↓
输出: 是否遮挡 / 遮挡类型 / 遮挡区域
```

## 五、推荐实现路径

### 路径 A - 基于人脸解析（最通用）

1. 使用 InsightFace 做人脸检测和对齐
2. 使用 face-parsing.PyTorch (BiSeNet) 做语义分割
3. 解析结果中，统计 "左眼/右眼/鼻子/嘴巴" 区域被 "背景/遮挡物" 覆盖的比例
4. 超过阈值则判定该区域被遮挡

```python
# 伪代码示例
from insightface.app import FaceAnalysis
import cv2

app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0)

img = cv2.imread('test.jpg')
faces = app.get(img)

for face in faces:
    # face.parsing 包含像素级解析结果
    # 0: background, 1: skin, 2: left_eyebrow, 3: right_eyebrow
    # 4: left_eye, 5: right_eye, 6: nose, 7: upper_lip, 8: inner_mouth
    # 9: lower_lip, 10: hair, 11: hat, 12: eyeglass, 13: necklace
    parsing_map = face.parsing
    
    # 统计眼部区域是否被遮挡
    left_eye_mask = (parsing_map == 4)
    right_eye_mask = (parsing_map == 5)
    
    # 判断这些区域是否为背景/遮挡物
    # ...
```

### 路径 B - 轻量级分类（适合移动端）

1. 用 MobileNetV3 / EfficientNet-Lite 做 backbone
2. 训练多分类：无遮挡 / 口罩 / 墨镜 / 围巾 / 其他
3. 输出类别和置信度

### 路径 C - 使用 InsightFace 集成方案

1. InsightFace 的 attribute 模块直接包含属性预测
2. parsing 模块可做像素级分割
3. 开箱即用，社区活跃

## 六、关键论文

| 论文 | 会议/年份 | 关键词 |
|---|---|---|
| BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation | ECCV 2018 | 人脸解析基础网络 |
| Face Parsing with RoI Tanh-Warping | CVPR 2019 | 人脸语义分割 |
| Masked Face Recognition: Challenges and Solutions | 综述 | 口罩人脸识别 |
| Occluded Face Recognition: Methods, Datasets and Challenges | 综述 | 遮挡人脸识别 |

## 七、方案选择建议

| 场景 | 推荐方案 |
|---|---|
| 快速判断是否遮挡 | 二分类 CNN (MobileNet) |
| 区分遮挡类型 | 多分类 CNN |
| 精确定位遮挡区域 | Face Parsing (BiSeNet) |
| 集成人脸识别系统 | InsightFace attribute/parsing |
| 口罩专项检测 | YOLO 检测器 |

**总结**: 最通用且效果好的方案是 **Face Parsing**（方案3），它既能判断是否有遮挡，又能精确定位遮挡区域，还能区分遮挡物类型。

---

---

# 狗脸遮挡检测 (Dog Face Occlusion Detection) 技术调研

> 针对单张静态图片的狗脸面部遮挡检测方案调研

## 八、狗脸遮挡检测概述

### 与人脸遮挡检测的差异

| 维度 | 人脸遮挡检测 | 狗脸遮挡检测 |
|---|---|---|
| 遮挡物类型 | 口罩、墨镜、围巾等标准化物品 | 口套、帽子、毛发打结、脏污等，更随机 |
| 面部形态差异 | 差异相对较小 | 120+ 品种，脸型、毛色差异巨大 |
| 天然遮挡 | 较少 | 毛发本身就是天然"遮挡"，边界模糊 |
| 标注数据集 | 丰富（CelebA、MAFA 等） | 几乎没有专门的遮挡数据集 |
| 相关研究 | 成熟领域 | 非常冷门，几乎没有专门研究 |

### 狗脸常见的遮挡类型

- 口套/口罩 (muzzle)
- 帽子/头饰 (hat/headwear)
- 毛发遮挡/打结 (matted fur)
- 脏污/异物 (dirt/foreign objects)
- 眼罩/绷带 (eye patch/bandage)
- 其他物体遮挡

## 九、狗脸遮挡检测技术方案

### 方案1: VLM 大模型直接判断（零样本，最快上手）

- **思路**: 用 GPT-4o / Qwen-VL / LLaVA 等多模态模型直接判断
- **Prompt 示例**:
  ```
  请判断图片中狗的面部是否有遮挡。
  遮挡类型包括：口套/口罩、帽子/头饰、毛发遮挡、脏污、其他物体遮挡。
  请回答：是否有遮挡（是/否），遮挡类型，遮挡部位。
  ```
- **优点**: 无需训练数据，零样本泛化能力强，可处理任意遮挡类型
- **缺点**: 推理速度慢、API 成本高、可能产生幻觉
- **适用**: 冷启动验证、数据标注辅助、非实时分析场景

### 方案2: YOLO 检测 + 分类器（推荐落地方案）

- **思路**:
  1. YOLOv8/YOLO11 检测狗脸区域
  2. 裁剪狗脸 → 送入 ResNet/EfficientNet 分类器
  3. 输出类别: 无遮挡 / 眼部遮挡 / 口鼻遮挡 / 全遮挡
- **训练数据来源**:
  - 真实遮挡图片（口套、帽子、脏污等）
  - 合成数据（正常狗脸上叠加遮挡 mask）
  - 用 VLM 先标注一批作为种子数据
- **优点**: 准确率高、速度快、适合生产部署
- **缺点**: 需要标注数据，遮挡类型需预定义

### 方案3: 关键点可见性推断遮挡 ⭐最推荐

- **思路**:
  1. 训练狗脸关键点检测模型（眼睛、鼻子、嘴巴等）
  2. 关键点模型输出 (坐标, 置信度/可见性)
  3. 如果关键部位关键点置信度极低 → 判定该区域被遮挡
  4. 结合关键点几何关系异常进一步确认
- **优点**: 天然与遮挡关联，能精确定位遮挡区域
- **缺点**: 需要带关键点标注的狗脸数据
- **适用**: 需要细粒度遮挡定位的场景

### 方案4: 传统 CV 方法（轻量级）

- **思路**:
  1. Haar Cascade / HOG+SVM 检测狗脸
  2. 颜色直方图、边缘检测、纹理分析判断遮挡
  3. 检查眼部区域亮暗度（类似墨镜检测）
- **优点**: 无需 GPU，极轻量
- **缺点**: 泛化能力差、对光照角度敏感、准确率有限

### 方案5: SAM 分割 + 分析（最精细化）

- **思路**:
  1. YOLO 检测狗脸区域
  2. SAM / YOLOv8-Seg 对狗脸做语义分割
  3. 分析分割结果中面部区域的完整性
- **优点**: 可处理任意形状的遮挡
- **缺点**: 系统复杂度高

## 十、狗脸检测相关开源项目

| 项目 | 用途 | Stars | 链接 |
|---|---|---|---|
| kwjinwoo/Dog_face_landmark_detection | 狗脸关键点检测 | 11 | [GitHub](https://github.com/kwjinwoo/Dog_face_landmark_detection) |
| kskd1804/dog_face_haar_cascade | 狗脸检测（Haar 传统方法） | 5 | [GitHub](https://github.com/kskd1804/dog_face_haar_cascade) |
| DeepLabCut | 动物姿态追踪 | 5000+ | [GitHub](https://github.com/DeepLabCut/DeepLabCut) |
| MMPose | 统一姿态估计框架（支持动物关键点） | - | [GitHub](https://github.com/open-mmlab/mmpose) |
| ultralytics (YOLOv8-Pose) | 检测 + 关键点联合框架 | - | [GitHub](https://github.com/ultralytics/ultralytics) |
| menorashid/animal_human_kp | CVPR 2017 跨物种关键点 | 53 | [GitHub](https://github.com/menorashid/animal_human_kp) |

### 相关论文

| 论文 | 会议/年份 | 关键词 |
|---|---|---|
| Interspecies Knowledge Transfer for Facial Keypoint Detection | CVPR 2017 | 跨物种关键点，支持猫、狗等 |
| Automated Detection of Cat Facial Landmarks | IJCV 2024 | 猫面部 48 关键点，方法可迁移到狗 |
| Sheep Facial Pain Assessment Under Weighted Graph Neural Networks | FG 2025 | 图神经网络动物面部关键点 |
| AnimalWeb: A Hierarchical Dataset of Animal Faces | - | 大规模动物面部数据集，68 关键点 |

## 十一、狗脸相关数据集

| 数据集 | 规模 | 标注内容 | 来源 |
|---|---|---|---|
| Stanford Dogs | 20,580 张 / 120 品种 | bounding box | [link](http://vision.stanford.edu/aditya86/ImageNetDogs/) |
| Oxford-IIIT Pet | 7,349 张 / 37 品种 | 头部+身体部位像素级标注 | [link](https://www.robots.ox.ac.uk/~vgg/data/pets/) |
| AnimalWeb | 21,955 张 / 334 物种 | 68 个面部关键点 | [link](https://arxiv.org/abs/1811.07021) |
| StanfordExtra | 12,000 张 | 8 个身体关键点 | [GitHub](https://github.com/bbashirazizv/StanfordExtra) |
| AP-10K | 10,015 张 / 23 物种 | 17 个关键点 | [GitHub](https://github.com/AlexTheBad/AP-10K) |

## 十二、狗脸遮挡检测工程流水线

```
输入图片
  ↓
狗脸检测 (YOLOv8 / Haar Cascade)
  ↓
┌──────────────────────────────────────────┐
│ 遮挡判断 (选以下一种或组合)              │
│                                          │
│  A. 关键点检测 → 关键点置信度低=被遮挡    │
│                                          │
│  B. 分类模型 → 直接输出遮挡类型           │
│                                          │
│  C. VLM 推理 → 自然语言描述遮挡情况       │
│                                          │
│  D. 分割模型 → 分析面部区域完整性         │
└──────────────────────────────────────────┘
  ↓
输出: 是否遮挡 / 遮挡类型 / 遮挡区域
```

## 十三、推荐分阶段实施路径

### 阶段一：快速验证（1-2 天）

- 使用 VLM（GPT-4o / Qwen-VL）零样本判断狗脸是否遮挡
- 同时用 VLM 标注一批数据作为种子训练集
- 验证可行性，确定遮挡类型分类体系

### 阶段二：落地部署（1-2 周）

- YOLO 狗脸检测 + 关键点可见性判断遮挡
- 用 VLM 生成的标注 + 合成数据训练模型
- 轻量高效，适合生产环境

### 阶段三：持续优化

- 增加遮挡分类头 → 支持多类型遮挡识别
- 合成数据增强 + 真实数据迭代
- 建立专门的狗脸遮挡数据集

## 十四、狗脸遮挡检测方案选择建议

| 场景 | 推荐方案 |
|---|---|
| 快速验证可行性 | VLM 零样本推理 |
| 区分遮挡类型 | YOLO + 多分类 CNN |
| 精确定位遮挡区域 | 关键点可见性推断 |
| 极轻量嵌入式部署 | 传统 CV（Haar + 纹理分析） |
| 最精细遮挡分析 | SAM 分割 + 区域分析 |

**总结**: 狗脸遮挡检测目前没有专门研究，建议组合"狗脸检测 + 关键点可见性 + VLM 辅助标注"的方案。VLM 适合冷启动和数据标注，关键点方案适合生产部署。

---

*调研时间: 2026-05-04*
