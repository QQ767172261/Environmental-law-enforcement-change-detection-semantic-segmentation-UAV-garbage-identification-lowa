1![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/acb3d531d8bf44afaf16186884414c8d.jpeg)
航拍无人机建筑垃圾分割检测数据集
无人机类型：DJI MAVIC 2
数据类型：Image+Json分割标签+Mask分割标签
内存大小：22.1G（4494张）
图片分辨率：4000*3000
采集高度：60m~120m
采集角度：90°
类别：建筑垃圾 
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/3ac7c126b2ff4164a43868f40a65aba2.png)


应用场景：用于环保执法、变化检测，语义分割、无人机垃圾识别低空目标检测等领域
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a6a7e570fba94a5e98b7cc0088a8a434.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/8c69df74c54445f3b40050bb76da8518.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/53f029ea834248c29cbbca531a2b71bf.png)
1
无人机建筑垃圾分割检测数据集】
无人机：DJI MAVIC 2
数据类型：Image+Json分割标签+Mask分割标签
总内存大小：22.1G（4494张）
图片分辨率：4000*3000
采集高度：60m~120m
采集角度：90°
类别：建筑垃圾


### 📊 无人机建筑垃圾分割检测数据集概览表

| **属性**            | **描述**                                                                 |
|---------------------|--------------------------------------------------------------------------|
| **数据集名称**       | 无人机建筑垃圾分割检测数据集                                              |
| **采集设备**         | DJI Mavic 2 无人机                                                       |
| **数据类型**         | - 原始图像（RGB，JPG/PNG）<br>- 分割标签（JSON 格式，COCO 风格）<br>- 二值掩码（Mask，PNG 格式） |
| **图像数量**         | 4,494 张                                                                 |
| **总数据大小**       | 22.1 GB                                                                  |
| **图像分辨率**       | 4000 × 3000 像素（1200 万像素）                                           |
| **采集高度**         | 60 米 ~ 120 米                                                           |
| **拍摄角度**         | 90°（垂直俯拍，正射影像）                                                 |
| **目标类别**         | **建筑垃圾**（单类语义分割）<br>（包括砖块、混凝土碎块、木材、塑料、金属废料等混合废弃物） |
| **标注格式**         | - **JSON**：符合 COCO 实例分割格式（含 polygon 坐标）<br>- **Mask**：与原图同名的 PNG 文件，白色（255）表示建筑垃圾区域，黑色（0）为背景 |
| **应用场景**         | 城市违建监测、非法倾倒识别、智慧城管、环境治理、AI+遥感分析                |
| **地理覆盖**         | 多城市城郊结合部、待开发地块、拆迁区域（具体位置可脱敏提供）               |
| **光照/天气条件**    | 晴天为主，部分多云；无雨雪雾霾                                            |
| **数据划分建议**     | 训练集 : 验证集 : 测试集 = 70% : 15% : 15%（约 3146 : 674 : 674）         |
| **兼容框架**         | 支持主流分割模型训练：<br>- Mask R-CNN<br>- U-Net<br>- YOLOv8-seg<br>- SAM + 微调 |

---



---
你提供的信息描述了一个**基于无人机航拍的建筑垃圾分割检测数据集**，以下是该数据集的关键信息整理以及配套的 **YOLOv8 实例分割训练代码示例**（适用于 YOLOv8-seg 模型）：

---

## 📦 数据集概览

- **采集设备**：DJI Mavic 2  
- **数据类型**：
  - 原始图像（Image）：4000×3000 分辨率 JPEG/PNG
  - 标注格式：
    - JSON（可能是 COCO 或自定义格式）
    - Mask 图像（与原图同名的二值分割掩码 PNG）
- **总规模**：4494 张图像，约 22.1 GB
- **飞行参数**：
  - 高度：60m ~ 120m
  - 拍摄角度：垂直向下（90°）
- **目标类别**：单类 —— `建筑垃圾`（Construction Waste）

---

## 🧠 推荐任务类型

由于你提供了 **Mask 分割标签**，适合做 **实例分割（Instance Segmentation）** 或 **语义分割（Semantic Segmentation）**。  
YOLOv8 支持 **实例分割（YOLOv8-seg）**，可同时输出边界框 + 分割掩码。

---

## 🛠️ YOLOv8 实例分割训练代码（PyTorch + Ultralytics）

### 第一步：准备数据目录结构

建议组织如下：

```
dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── dataset.yaml
```

> 注意：YOLOv8-seg 要求标签为 **TXT 格式**（每张图对应一个 .txt 文件），格式为：
>
> ```
> class_id x1 y1 x2 y2 ... xn yn
> ```
> 其中 `(xi, yi)` 是归一化后的多边形顶点坐标（0~1）。

如果你当前是 **Mask PNG + JSON**，需先转换为 YOLOv8 所需格式。可用脚本将 Mask 转为多边形轮廓（OpenCV + shapely）。

---

### 第二步：创建 `dataset.yaml`

```yaml
# dataset.yaml
train: ./dataset/images/train
val: ./dataset/images/val

nc: 1
names: ['construction_waste']
```

---

### 第三步：训练代码（Python）

```python
from ultralytics import YOLO

# 加载预训练的 YOLOv8-seg 模型（如 yolov8n-seg.pt）
model = YOLO('yolov8n-seg.pt')  # 可选：yolov8s-seg.pt, yolov8m-seg.pt 等

# 开始训练
results = model.train(
    data='dataset/dataset.yaml',
    epochs=100,
    imgsz=1280,          # 输入图像尺寸（建议 640~1280，原始图太大需 resize）
    batch=8,             # 根据 GPU 显存调整
    name='yolov8_construction_waste_seg',
    device=0,            # GPU ID，如 'cpu' 或 [0,1]
    augment=True,        # 启用数据增强
    cache=False,         # 若内存充足可设为 True 加速读取
    workers=8
)
```

---

### 第四步：推理示例

```python
results = model('test_image.jpg')
results[0].show()  # 显示带分割掩码的结果
```

---

## 🔁 数据格式转换提示（JSON/Mask → YOLO Seg）

若你的标注是 **二值 Mask PNG**（白色=垃圾，黑色=背景），可用以下伪代码提取轮廓并保存为 YOLO 格式：

```python
import cv2
import numpy as np
import os

def mask_to_yolo_txt(mask_path, txt_path, class_id=0):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    h, w = mask.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    with open(txt_path, 'w') as f:
        for cnt in contours:
            if len(cnt) < 3: 
                continue  # 至少3点才能构成多边形
            # 归一化坐标
            cnt = cnt.reshape(-1, 2)
            cnt_norm = cnt / np.array([w, h])
            line = [str(class_id)] + [f"{x:.6f}" for pt in cnt_norm for x in pt]
            f.write(" ".join(line) + "\n")
```

> ⚠️ 注意：YOLOv8 要求每个对象一行，且多边形必须闭合（但不需要重复首尾点）。

---

## ✅ 建议

1. **图像下采样**：原始 4000×3000 太大，训练时建议 resize 到 1280×960 或 640×480。
2. **数据划分**：按 8:2 或 7:3 划分 train/val，确保 val 集有代表性。
3. **增强策略**：可开启 HSV、旋转、Mosaic 等增强，提升小目标检测能力。
4. **评估指标**：关注 mAP50-95 和 mask mAP。

---

当然可以！以下是一个**完整的 Python 脚本**，用于将 **二值分割掩码图像（Mask PNG）** 转换为 **YOLOv8 实例分割所需的 TXT 标签格式**。

---

## ✅ 前提假设

- 原始图像和掩码图像**文件名一致**（如 `001.jpg` 对应 `001.png`）
- 掩码是**单通道灰度图或三通道图**，其中 **白色（255）表示建筑垃圾，黑色（0）表示背景**
- 每个连通区域视为一个独立实例（即一个对象）
- 输出目录结构与 YOLOv8 兼容（`images/` 和 `labels/` 平行）

---

## 📜 完整转换脚本：`mask_to_yolo_seg.py`

```python
import os
import cv2
import numpy as np
from pathlib import Path

def mask_to_yolo_txt(mask_path, txt_path, class_id=0, min_area=50):
    """
    将单张二值掩码图转换为 YOLO 分割格式的 .txt 标签文件
    :param mask_path: 掩码图像路径（PNG）
    :param txt_path: 输出的 .txt 路径
    :param class_id: 类别 ID（默认 0）
    :param min_area: 最小轮廓面积（过滤噪声）
    """
    # 读取掩码（强制转为单通道）
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Warning: Cannot read mask {mask_path}")
        return

    h, w = mask.shape

    # 二值化（确保只有 0 和 255）
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # 查找所有外部轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lines = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue  # 过滤小区域噪声

        # 提取轮廓点 (N, 1, 2) -> (N, 2)
        epsilon = 0.001 * cv2.arcLength(cnt, True)  # 可选：多边形近似简化
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        points = approx.reshape(-1, 2)

        if len(points) < 3:
            continue  # 至少3个点才能构成多边形

        # 归一化到 [0, 1]
        points_norm = points.astype(np.float32) / np.array([w, h])
        coords = points_norm.flatten().tolist()
        line = [str(class_id)] + [f"{x:.6f}" for x in coords]
        lines.append(" ".join(line))

    # 写入文件
    with open(txt_path, 'w') as f:
        f.write("\n".join(lines))

def convert_dataset(
    image_dir,
    mask_dir,
    output_image_dir,
    output_label_dir,
    class_id=0,
    min_area=50,
    img_exts=('.jpg', '.jpeg', '.png'),
    mask_ext='.png'
):
    """
    批量转换整个数据集
    """
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    output_image_dir = Path(output_image_dir)
    output_label_dir = Path(output_label_dir)

    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)

    image_files = [f for f in image_dir.iterdir() if f.suffix.lower() in img_exts]

    for img_path in image_files:
        stem = img_path.stem
        mask_path = mask_dir / (stem + mask_ext)

        if not mask_path.exists():
            print(f"Mask not found for {img_path}, skipping...")
            continue

        # 复制图像到输出目录（YOLO 需要 images/ 中有图）
        dst_img = output_image_dir / img_path.name
        if not dst_img.exists():
            import shutil
            shutil.copy(img_path, dst_img)

        # 生成标签
        txt_path = output_label_dir / (stem + '.txt')
        mask_to_yolo_txt(mask_path, txt_path, class_id=class_id, min_area=min_area)

    print(f"✅ 转换完成！共处理 {len(image_files)} 张图像。")

if __name__ == "__main__":
    # ================== 配置区 ==================
    # 原始数据路径
    RAW_IMAGE_DIR = "raw/images"      # 原图文件夹
    RAW_MASK_DIR = "raw/masks"        # 掩码文件夹（PNG）

    # YOLO 输出路径
    OUTPUT_BASE = "dataset"
    TRAIN_RATIO = 0.8  # 可后续划分 train/val，或先全部转再手动分

    # 创建 train 目录（你也可以分别处理 train/val）
    IMAGE_OUT = f"{OUTPUT_BASE}/images/train"
    LABEL_OUT = f"{OUTPUT_BASE}/labels/train"

    # 参数
    CLASS_ID = 0
    MIN_AREA = 100  # 像素面积，过滤小噪点
    # ==========================================

    convert_dataset(
        image_dir=RAW_IMAGE_DIR,
        mask_dir=RAW_MASK_DIR,
        output_image_dir=IMAGE_OUT,
        output_label_dir=LABEL_OUT,
        class_id=CLASS_ID,
        min_area=MIN_AREA
    )

    print("💡 提示：转换完成后，请按需划分 val 集，并更新 dataset.yaml")
```

---

## 📂 使用说明

1. **安装依赖**：
   ```bash
   pip install opencv-python numpy
   ```

2. **组织原始数据**：
   ```
   raw/
   ├── images/
   │   ├── 0001.jpg
   │   ├── 0002.jpg
   │   └── ...
   └── masks/
       ├── 0001.png
       ├── 0002.png
       └── ...
   ```

3. **修改脚本顶部配置**（如路径、类别ID等）

4. **运行脚本**：
   ```bash
   python mask_to_yolo_seg.py
   ```

5. **输出结果**：
   ```
   dataset/
   └── images/train/
   └── labels/train/
   ```

6. **划分验证集**（可选）：
   - 手动移动部分 `images/train` 和 `labels/train` 到 `images/val` 和 `labels/val`
   - 或使用 `splitfolders` 库自动划分

---

## ⚠️ 注意事项

- **轮廓简化**：脚本中使用了 `cv2.approxPolyDP` 简化多边形，减少点数（避免 YOLO 标签过大）。如需保留原始轮廓，可注释掉该行。
- **多类支持**：当前为单类（建筑垃圾）。若未来扩展多类，需根据 Mask 的不同灰度值或彩色通道区分类别。
- **大图处理**：4000×3000 图像在训练时建议 resize，但**标签转换必须用原始分辨率**（因坐标归一化基于原图尺寸）。

---


