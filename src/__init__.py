# LayerPipeline 源代码包
"""
密度分层Pipeline模块

包含以下子模块：
- processImage: 4x图像处理，分割组织、灰质、白质
- segmentCells: 40x图像细胞分割（StarDist）
- boundaryTransfer: 坐标转换（4x ↔ 40x）
- analyseDensity: 密度分析与分层算法
- layerVisualize: 分层结果可视化
- register40to4: 图像配准工具
"""
