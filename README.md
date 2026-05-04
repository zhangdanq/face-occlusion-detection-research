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

*调研时间: 2026-05-04*
