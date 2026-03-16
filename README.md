# 细胞密度分层 Pipeline

基于4x和40x显微镜图像的大脑皮层细胞密度分层分析工具。

## 功能概述

该Pipeline实现以下功能：
1. **4x图像处理**：分割完整组织、灰质和白质区域，提取灰质外边界(GM)和白质边界(WM)
2. **坐标转换**：将4x坐标系下的边界和掩码转换到40x坐标系
3. **细胞分割**：使用StarDist模型在40x图像的灰质区域分割细胞核
4. **密度分析**：计算细胞深度-密度直方图，调用分层算法分出6层
5. **结果可视化**：在40x图像上绘制分层线

## 环境要求

```bash
pip install numpy opencv-python pandas matplotlib scikit-learn scikit-image scipy seaborn
pip install stardist csbdeep tensorflow
```

## 目录结构

```
layerpipeline/
├── main.py              # 主程序入口
├── input/               # 输入文件夹
│   ├── 4x.png          # 4x低倍镜图像
│   ├── 4x.json         # 4x元数据（包含物理坐标和像素比例）
│   ├── 40x.png         # 40x高倍镜图像
│   └── 40x.json        # 40x元数据
├── output/              # 输出文件夹
└── src/                 # 源代码模块
    ├── processImage.py      # 4x图像处理
    ├── segmentCells.py      # 细胞分割（StarDist）
    ├── boundaryTransfer.py  # 坐标转换
    ├── analyseDensity.py    # 密度分析与分层算法
    ├── layerVisualize.py    # 可视化
    └── register40to4.py     # 图像配准工具
```

## JSON元数据格式

4x.json 和 40x.json 文件应包含以下结构：

```json
{
    "scan_info": {
        "pixel_scale": 1.625,      // 每像素对应的物理距离（微米）
        "camera_resolution": [1376, 1024]  // 相机分辨率 [宽, 高]
    },
    "positions": [
        {"x": 12345.0, "y": 67890.0}  // 扫描起始位置的物理坐标
    ]
}
```

## 使用方法

### 方式1：完整Pipeline

```python
from main import LayerPipeline

# 创建Pipeline实例
pipeline = LayerPipeline(
    input_dir="input",
    output_dir="output"
)

# 运行完整Pipeline
pipeline.run_full_pipeline(
    n_layers=6,           # 分6层（皮层6层结构）
    method='kmeans',      # 分层方法: 'kmeans' 或 'gmm'
    use_roi_mask=True,    # 使用灰质掩码限制细胞分割区域
    enhance=False         # 是否对40x图像进行对比度增强
)
```

### 方式2：分步骤执行（便于调试）

```python
from main import LayerPipeline

pipeline = LayerPipeline("input", "output")

# 步骤0: 加载元数据
pipeline._load_metadata()

# 步骤1: 处理4x图像（分割组织、灰质、白质）
pipeline.step1_process_4x_image()

# 步骤2: 坐标转换（4x → 40x）
pipeline.step2_coordinate_transform()

# 步骤3: 细胞分割
pipeline.step3_segment_cells(use_roi_mask=True, enhance=False)

# 步骤4: 密度分析与分层
pipeline.step4_analyze_density_and_layer(n_layers=6, method='kmeans')

# 步骤5: 可视化
pipeline.step5_visualize()
```

### 方式3：命令行运行

```bash
python main.py
```

## 输出文件说明

| 文件名 | 说明 |
|--------|------|
| `tissueMask.png` | 完整组织掩码 |
| `grayMask.png` | 灰质区域掩码（4x坐标系） |
| `whiteMask.png` | 白质区域掩码 |
| `GM.csv` | 灰质外边界点（4x坐标系） |
| `WM.csv` | 白质边界点（4x坐标系） |
| `GM_40x.csv` | 灰质外边界点（40x坐标系） |
| `WM_40x.csv` | 白质边界点（40x坐标系） |
| `grayMask_40x.png` | 灰质掩码（40x坐标系） |
| `nuclei_centroids.csv` | 细胞核中心点坐标 |
| `nuclei_info.csv` | 细胞信息（含置信度） |
| `segmented_mask.png` | 细胞分割掩码 |
| `segmented_layers.csv` | 分层结果 |
| `layers_lines.png` | 分层可视化图像 |
| `depth_density_layers_kmeans.png` | 深度-密度分层曲线图 |

## 分层结果格式

`segmented_layers.csv` 包含以下列：

| 列名 | 说明 |
|------|------|
| layer | 层号 (1-6) |
| start | 层起始深度 (0=GM边界) |
| end | 层结束深度 (1=WM边界) |
| mean_density | 该层平均细胞密度 |

## 分层算法

支持以下分层方法：

1. **kmeans** (默认): KMeans聚类，支持先验初始化
2. **gmm**: 高斯混合模型聚类
3. **gradient**: 密度梯度分析法
4. **second_derivative**: 二阶导数方法
5. **multi_threshold**: 多阈值分割法
6. **dbscan**: DBSCAN密度聚类

## 坐标系统

- **深度定义**: 0 = 灰质外边界(pial surface)，1 = 白质边界
- **坐标转换**: 基于JSON元数据中的物理坐标和像素比例进行转换

## 注意事项

1. 确保4x和40x图像来自同一组织样本
2. JSON元数据中的物理坐标应准确
3. 大图像处理可能需要较长时间和较大内存
4. StarDist模型首次运行会自动下载预训练权重
