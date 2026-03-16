"""
密度分层Pipeline主程序

完整流程：
1. 处理4x图像：分割组织、灰质、白质，提取边界
2. 坐标转换：将4x边界和掩码转换到40x坐标系
3. 处理40x图像：使用灰质掩码提取ROI区域，调用StarDist分割细胞
4. 密度分析：计算细胞深度-密度直方图，分出6层
5. 可视化：绘制分层结果图像
"""

import os
import json
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

# 导入各模块功能
from src.processImage import (
    croptissue, cropwhite, extractgray, computeContours
)
from src.segmentCells import (
    segment_nuclei, save_segmentation_results, enhance_contrast
)
from src.boundaryTransfer import (
    read_json_center_and_scale, convert_points, save_points_to_csv,
    convert_mask_4x_to_40x
)
from src.analyseDensity import (
    analyze, computeAverage, segmentLayer_kmeans, segmentLayer_gmm, segmentLayer_peak_based
)
from src.layerVisualize import (
    assign_layers_to_mask
)


class LayerPipeline:
    """密度分层Pipeline类"""
    
    def __init__(self, input_dir: str, output_dir: str):
        """
        初始化Pipeline
        
        Args:
            input_dir: 输入文件夹路径，应包含4x.png, 4x.json, 40x.png, 40x.json
            output_dir: 输出文件夹路径
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        
        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 输入文件路径
        self.image_4x_path = self.input_dir / "4x.png"
        self.json_4x_path = self.input_dir / "4x.json"
        self.image_40x_path = self.input_dir / "40x.png"
        self.json_40x_path = self.input_dir / "40x.json"
        
        # 验证输入文件存在
        self._validate_inputs()
        
        # 中间结果存储
        self.center_4x = None
        self.scale_4x = None
        self.res_4x = None
        self.center_40x = None
        self.scale_40x = None
        self.res_40x = None
        
        # 掩码和边界
        self.tissue_mask = None
        self.gray_mask = None
        self.white_mask = None
        self.gm_boundary = None  # 灰质外边界（GM-pial边界）
        self.wm_boundary = None  # 白质边界（GM-WM边界）
        
        # 细胞分割结果
        self.cell_labels = None
        self.cell_details = None
        self.centroids_df = None
        
        # 分层结果
        self.layers = None
        
    def _check_files_exist(self, files: list) -> bool:
        """检查文件列表是否全部存在"""
        return all((self.output_dir / f).exists() for f in files)
    
    def _validate_inputs(self):
        """验证输入文件是否存在"""
        required_files = [
            self.image_4x_path, self.json_4x_path,
            self.image_40x_path, self.json_40x_path
        ]
        missing = [str(f) for f in required_files if not f.exists()]
        if missing:
            raise FileNotFoundError(f"缺少输入文件: {missing}")
        print(f"[✓] 输入文件验证通过")
        
    def _load_metadata(self):
        """加载JSON元数据，获取坐标转换参数"""
        print("\n[步骤0] 加载元数据...")
        
        self.center_4x, self.scale_4x, self.res_4x = read_json_center_and_scale(
            str(self.json_4x_path)
        )
        self.center_40x, self.scale_40x, self.res_40x = read_json_center_and_scale(
            str(self.json_40x_path)
        )
        
        print(f"  4x - 中心: {self.center_4x}, 像素比例: {self.scale_4x}, 分辨率: {self.res_4x}")
        print(f"  40x - 中心: {self.center_40x}, 像素比例: {self.scale_40x}, 分辨率: {self.res_40x}")
        
    def step1_process_4x_image(self, force=False):
        """
        步骤1: 处理4x图像
        - 分割完整组织区域
        - 分割灰质和白质
        - 提取灰质外边界(GM)和白质边界(WM)
        
        Args:
            force: 强制重新运行，忽略已存在的缓存文件
        """
        print("\n" + "="*60)
        print("[步骤1] 处理4x图像: 分割组织、灰质、白质")
        print("="*60)
        
        # 检查缓存文件是否存在
        required_files = ["tissueMask.png", "grayMask.png", "whiteMask.png", "GM.csv", "WM.csv"]
        if not force and self._check_files_exist(required_files):
            print("  [缓存] 检测到已有结果文件，跳过此步骤")
            print("  加载已有结果...")
            self.tissue_mask = cv2.imread(str(self.output_dir / "tissueMask.png"), cv2.IMREAD_GRAYSCALE)
            self.gray_mask = cv2.imread(str(self.output_dir / "grayMask.png"), cv2.IMREAD_GRAYSCALE)
            self.white_mask = cv2.imread(str(self.output_dir / "whiteMask.png"), cv2.IMREAD_GRAYSCALE)
            print("[✓] 步骤1完成: 从缓存加载")
            return
        
        # 读取4x图像
        image_4x = cv2.imread(str(self.image_4x_path), cv2.IMREAD_GRAYSCALE)
        if image_4x is None:
            raise ValueError(f"无法读取4x图像: {self.image_4x_path}")
        print(f"  读取4x图像: {image_4x.shape}")
        
        # 1.1 分割组织区域
        print("  [1.1] 分割组织区域...")
        self.tissue_mask, tissue_contours = croptissue(image_4x)
        tissue_image = cv2.bitwise_and(image_4x, image_4x, mask=self.tissue_mask)
        cv2.imwrite(str(self.output_dir / "tissueMask.png"), self.tissue_mask)
        cv2.imwrite(str(self.output_dir / "tissueImage.png"), tissue_image)
        print(f"    组织掩码已保存")
        
        # 1.2 分割白质区域
        print("  [1.2] 分割白质区域...")
        self.white_mask, white_contours = cropwhite(image_4x, self.tissue_mask, tissue_image)
        cv2.imwrite(str(self.output_dir / "whiteMask.png"), self.white_mask)
        white_image = cv2.bitwise_and(image_4x, image_4x, mask=self.white_mask)
        cv2.imwrite(str(self.output_dir / "whiteImage.png"), white_image)
        print(f"    白质掩码已保存")
        
        # 1.3 计算灰质区域
        print("  [1.3] 计算灰质区域...")
        self.gray_mask, gray_contours = extractgray(self.tissue_mask, self.white_mask)
        cv2.imwrite(str(self.output_dir / "grayMask.png"), self.gray_mask)
        gray_image = cv2.bitwise_and(image_4x, image_4x, mask=self.gray_mask)
        cv2.imwrite(str(self.output_dir / "grayImage.png"), gray_image)
        print(f"    灰质掩码已保存")
        
        # 1.4 计算边界线（GM外边界和WM边界）
        print("  [1.4] 计算边界线...")
        outer_contours, inner_contours = computeContours(
            white_contours, gray_contours, issave=False
        )
        
        # 保存边界到CSV（4x坐标系）
        self._save_boundary_csv(outer_contours, self.output_dir / "GM.csv")
        self._save_boundary_csv(inner_contours, self.output_dir / "WM.csv")
        print(f"    GM边界点: {sum(len(c) for c in outer_contours)}")
        print(f"    WM边界点: {sum(len(c) for c in inner_contours)}")
        
        # 可视化边界
        self._visualize_boundaries_4x(image_4x, outer_contours, inner_contours)
        
        print("[✓] 步骤1完成: 4x图像处理")
        
    def _save_boundary_csv(self, contours, path):
        """将轮廓点保存为CSV文件"""
        with open(path, "w") as f:
            f.write("x,y\n")
            for contour in contours:
                for point in contour:
                    x, y = point[0]
                    f.write(f"{x},{y}\n")
                    
    def _visualize_boundaries_4x(self, image, outer_contours, inner_contours):
        """在4x图像上可视化边界"""
        vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # 绘制GM边界（紫色）
        for cnt in outer_contours:
            for point in cnt:
                x, y = point[0]
                if 0 < x < image.shape[1]-1 and 0 < y < image.shape[0]-1:
                    cv2.circle(vis_image, (x, y), 3, (255, 0, 255), -1)
        
        # 绘制WM边界（黄色）
        for cnt in inner_contours:
            for point in cnt:
                x, y = point[0]
                if 0 < x < image.shape[1]-1 and 0 < y < image.shape[0]-1:
                    cv2.circle(vis_image, (x, y), 3, (255, 255, 0), -1)
                    
        cv2.imwrite(str(self.output_dir / "boundaries_4x.png"), vis_image)
        
    def step2_coordinate_transform(self, force=False):
        """
        步骤2: 坐标转换
        - 将4x边界点转换到40x坐标系
        - 将4x灰质掩码转换到40x坐标系
        
        Args:
            force: 强制重新运行，忽略已存在的缓存文件
        """
        print("\n" + "="*60)
        print("[步骤2] 坐标转换: 4x → 40x")
        print("="*60)
        
        # 检查缓存文件是否存在
        required_files = ["GM_40x.csv", "WM_40x.csv", "grayMask_40x.png"]
        if not force and self._check_files_exist(required_files):
            print("  [缓存] 检测到已有结果文件，跳过此步骤")
            print("[✓] 步骤2完成: 从缓存加载")
            return
        
        # 确保元数据已加载
        if self.center_4x is None:
            self._load_metadata()
        
        # 加载灰质掩码（如果未加载）
        if self.gray_mask is None:
            gray_mask_path = self.output_dir / "grayMask.png"
            if gray_mask_path.exists():
                self.gray_mask = cv2.imread(str(gray_mask_path), cv2.IMREAD_GRAYSCALE)
            else:
                raise ValueError("灰质掩码不存在，请先运行step1_process_4x_image()")
        
        # 读取4x边界CSV
        gm_csv_path = self.output_dir / "GM.csv"
        wm_csv_path = self.output_dir / "WM.csv"
        
        gm_pts_4x = self._read_points_csv(gm_csv_path)
        wm_pts_4x = self._read_points_csv(wm_csv_path)
        print(f"  4x坐标系 - GM点数: {len(gm_pts_4x)}, WM点数: {len(wm_pts_4x)}")
        
        # 坐标转换
        image_resolution = tuple(self.res_4x)
        gm_pts_40x = convert_points(
            gm_pts_4x, self.center_4x, self.scale_4x,
            self.center_40x, self.scale_40x, image_resolution
        )
        wm_pts_40x = convert_points(
            wm_pts_4x, self.center_4x, self.scale_4x,
            self.center_40x, self.scale_40x, image_resolution
        )
        
        # 保存转换后的边界
        save_points_to_csv(gm_pts_40x, str(self.output_dir / "GM_40x.csv"))
        save_points_to_csv(wm_pts_40x, str(self.output_dir / "WM_40x.csv"))
        print(f"  40x坐标系边界已保存")
        
        # 转换灰质掩码到40x坐标系
        print("  转换灰质掩码到40x坐标系...")
        image_40x = cv2.imread(str(self.image_40x_path))
        output_size = (image_40x.shape[1], image_40x.shape[0])
        
        gray_mask_40x = convert_mask_4x_to_40x(
            self.gray_mask, 
            self.center_4x, self.scale_4x,
            self.center_40x, self.scale_40x,
            self.res_4x, output_size
        )
        cv2.imwrite(str(self.output_dir / "grayMask_40x.png"), gray_mask_40x)
        print(f"  40x灰质掩码已保存")
        
        # 提取40x图像的灰质ROI区域
        roi_image = cv2.bitwise_and(image_40x, image_40x, mask=gray_mask_40x)
        cv2.imwrite(str(self.output_dir / "roiImage_40x.png"), roi_image)
        print(f"  40x灰质ROI图像已保存")
        
        # 在40x图像上可视化边界
        self._visualize_boundaries_40x(image_40x, wm_pts_40x, gm_pts_40x)
        
        print("[✓] 步骤2完成: 坐标转换")
        
    def _read_points_csv(self, path):
        """从CSV读取点坐标"""
        pts = []
        with open(path, 'r') as f:
            next(f)  # 跳过header
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    try:
                        pts.append((float(parts[0]), float(parts[1])))
                    except ValueError:
                        continue
        return pts
    
    def _visualize_boundaries_40x(self, image, wm_pts, gm_pts):
        """在40x图像上可视化边界"""
        vis_image = image.copy()
        
        # 绘制点
        for (x, y) in gm_pts:
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                cv2.circle(vis_image, (int(x), int(y)), 25, (255, 0, 255), -1)
        
        for (x, y) in wm_pts:
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                cv2.circle(vis_image, (int(x), int(y)), 25, (255, 255, 0), -1)
                
        cv2.imwrite(str(self.output_dir / "boundaries_40x.png"), vis_image)
        
    def step3_segment_cells(self, use_roi_mask=True, enhance=False, force=False):
        """
        步骤3: 40x图像细胞分割
        - 使用灰质掩码提取ROI区域
        - 调用StarDist模型分割细胞核
        
        Args:
            use_roi_mask: 是否使用灰质掩码限制分割区域
            enhance: 是否进行对比度增强
            force: 强制重新运行，忽略已存在的缓存文件
        """
        print("\n" + "="*60)
        print("[步骤3] 细胞分割: StarDist")
        print("="*60)
        
        # 检查缓存文件是否存在
        required_files = ["nuclei_centroids.csv", "segmented_mask.png"]
        if not force and self._check_files_exist(required_files):
            print("  [缓存] 检测到已有分割结果，跳过此步骤")
            print("  加载已有结果...")
            self.centroids_df = pd.read_csv(str(self.output_dir / "nuclei_centroids.csv"))
            print(f"  已加载细胞数: {len(self.centroids_df)}")
            print("[✓] 步骤3完成: 从缓存加载")
            return
        
        # 读取40x图像
        from skimage import io as skio
        image_40x = skio.imread(str(self.image_40x_path))
        print(f"  读取40x图像: {image_40x.shape}, dtype: {image_40x.dtype}")
        
        # 可选：对比度增强
        if enhance:
            print("  进行对比度增强...")
            if image_40x.ndim == 3:
                gray_40x = cv2.cvtColor(image_40x, cv2.COLOR_RGB2GRAY)
            else:
                gray_40x = image_40x
            image_40x = enhance_contrast(gray_40x)
            cv2.imwrite(str(self.output_dir / "enhanced_40x.png"), image_40x)
        
        # 可选：使用灰质掩码限制区域
        if use_roi_mask:
            gray_mask_40x_path = self.output_dir / "grayMask_40x.png"
            if gray_mask_40x_path.exists():
                print("  应用灰质掩码限制分割区域...")
                mask_40x = cv2.imread(str(gray_mask_40x_path), cv2.IMREAD_GRAYSCALE)
                
                if image_40x.ndim == 3:
                    mask_3ch = cv2.cvtColor(mask_40x, cv2.COLOR_GRAY2RGB)
                    image_for_seg = cv2.bitwise_and(image_40x, mask_3ch)
                else:
                    image_for_seg = cv2.bitwise_and(image_40x, image_40x, mask=mask_40x)
            else:
                print("  警告: 灰质掩码不存在，使用完整图像")
                image_for_seg = image_40x
        else:
            image_for_seg = image_40x
        
        # 调用StarDist分割
        print("  调用StarDist模型进行细胞分割...")
        self.cell_labels, self.cell_details = segment_nuclei(
            image_for_seg, isfilter=True
        )
        
        # 保存分割结果
        print("  保存分割结果...")
        self.centroids_df = save_segmentation_results(
            self.cell_labels, self.cell_details, str(self.output_dir)
        )
        
        print(f"  检测到细胞数: {len(self.centroids_df)}")
        print("[✓] 步骤3完成: 细胞分割")
        
    def step4_analyze_density_and_layer(self, n_layers=6, method='peak_based', merge_layer23=False, 
                                        keep_negative_coords=True, force=False):
        """
        步骤4: 密度分析与分层
        - 计算细胞深度-密度直方图
        - 调用分层算法分出皮层层次
        
        Args:
            n_layers: 目标层数（仅对kmeans/gmm方法有效，默认6层）
            method: 分层方法 ('peak_based', 'kmeans', 'gmm')
            merge_layer23: 是否合并第2层和第3层（仅对peak_based方法有效）
                           - False: 输出5层 (L1, L2, L3, L4, L5/6)
                           - True:  输出4层 (L1, L2/3, L4, L5/6)
            keep_negative_coords: 是否保留负坐标点，False时会过滤掉x<0或y<0的边界点
            force: 强制重新运行，忽略已存在的缓存文件
        """
        print("\n" + "="*60)
        print("[步骤4] 密度分析与分层")
        print("="*60)
        
        # 检查缓存文件是否存在
        required_files = ["segmented_layers.csv"]
        if not force and self._check_files_exist(required_files):
            print("  [缓存] 检测到已有分层结果，跳过此步骤")
            print("  加载已有结果...")
            layers_df = pd.read_csv(str(self.output_dir / "segmented_layers.csv"))
            self.layers = layers_df.to_dict('records')
            print(f"  已加载分层结果: {len(self.layers)} 层")
            for layer in self.layers:
                print(f"    Layer {layer['layer']}: "
                      f"depth=[{layer['start']:.3f}, {layer['end']:.3f}], "
                      f"mean_density={layer['mean_density']:.2f}")
            print("[✓] 步骤4完成: 从缓存加载")
            return
        
        # 确保细胞分割结果存在
        if self.centroids_df is None:
            centroids_path = self.output_dir / "nuclei_centroids.csv"
            if centroids_path.exists():
                self.centroids_df = pd.read_csv(centroids_path)
            else:
                raise ValueError("细胞分割结果不存在，请先运行step3_segment_cells()")
        
        # 读取40x坐标系的边界
        wm_path = self.output_dir / "WM_40x.csv"
        gm_path = self.output_dir / "GM_40x.csv"
        
        if not wm_path.exists() or not gm_path.exists():
            raise ValueError("40x边界文件不存在，请先运行step2_coordinate_transform()")
        
        wm_df = pd.read_csv(wm_path)
        gm_df = pd.read_csv(gm_path)
        
        # 过滤负坐标点（如果需要）
        if not keep_negative_coords:
            wm_before = len(wm_df)
            gm_before = len(gm_df)
            wm_df = wm_df[(wm_df['x'] >= 0) & (wm_df['y'] >= 0)]
            gm_df = gm_df[(gm_df['x'] >= 0) & (gm_df['y'] >= 0)]
            print(f"  过滤负坐标: WM {wm_before} -> {len(wm_df)}, GM {gm_before} -> {len(gm_df)}")
        
        # 准备细胞坐标数据
        # 注意: nuclei_centroids.csv的列顺序是 [Y, X]
        cells = pd.DataFrame({
            'X': self.centroids_df['X'].values,
            'Y': self.centroids_df['Y'].values
        })
        
        print(f"  细胞数: {len(cells)}")
        print(f"  WM边界点数: {len(wm_df)}")
        print(f"  GM边界点数: {len(gm_df)}")
        
        # 4.1 计算深度和密度
        print("  [4.1] 计算细胞深度和密度...")
        depth, density = analyze(wm_df, gm_df, cells)
        
        # 4.2 计算平均密度曲线
        print("  [4.2] 计算深度-平均密度曲线...")
        avg_density, bin_centers = computeAverage(
            depth, density, 
            isshow=True, issave=True, 
            mode='average', issmooth=True
        )
        
        # 4.3 分层
        print(f"  [4.3] 使用{method}方法分层...")
        if method == 'peak_based' or method == 'peak':
            # 基于峰值的分层算法（推荐）
            # merge_layer23=False: 输出5层 (L1, L2, L3, L4, L5/6)
            # merge_layer23=True:  输出4层 (L1, L2/3, L4, L5/6)
            self.layers = segmentLayer_peak_based(
                avg_density, bin_centers,
                sigma=2,
                merge_layer23=merge_layer23,
                isshow=True, issave=True
            )
        elif method == 'kmeans':
            self.layers = segmentLayer_kmeans(
                avg_density, bin_centers, 
                n_clusters=n_layers,
                offset=True,  # 使用先验初始化
                isshow=True, issave=True
            )
        elif method == 'gmm':
            self.layers = segmentLayer_gmm(
                avg_density, bin_centers,
                n_clusters=n_layers,
                isshow=True, issave=True
            )
        else:
            raise ValueError(f"未知的分层方法: {method}. 可选: 'peak_based', 'kmeans', 'gmm'")
        
        # 保存分层结果
        layers_df = pd.DataFrame(self.layers)
        layers_df.to_csv(str(self.output_dir / "segmented_layers.csv"), index=False)
        print(f"  分层结果已保存")
        
        # 打印分层信息
        print("\n  分层结果:")
        for layer in self.layers:
            print(f"    Layer {layer['layer']}: "
                  f"depth=[{layer['start']:.3f}, {layer['end']:.3f}], "
                  f"mean_density={layer['mean_density']:.2f}")
        
        print("[✓] 步骤4完成: 密度分析与分层")
        
    def step5_visualize(self, force=False):
        """
        步骤5: 分层结果可视化
        - 在40x图像上绘制分层线
        
        Args:
            force: 强制重新运行，忽略已存在的缓存文件
        """
        print("\n" + "="*60)
        print("[步骤5] 分层结果可视化")
        print("="*60)
        
        # 检查缓存文件是否存在
        required_files = ["layers_lines.png", "layers_color_mask.png"]
        if not force and self._check_files_exist(required_files):
            print("  [缓存] 检测到已有可视化结果，跳过此步骤")
            print(f"  可视化图像位于: {self.output_dir / 'layers_lines.png'}")
            print(f"  分层颜色Mask位于: {self.output_dir / 'layers_color_mask.png'}")
            print("[✓] 步骤5完成: 从缓存加载")
            return
        
        # 检查分层结果
        layers_path = self.output_dir / "segmented_layers.csv"
        if not layers_path.exists():
            raise ValueError("分层结果不存在，请先运行step4_analyze_density_and_layer()")
        
        wm_path = str(self.output_dir / "WM_40x.csv")
        gm_path = str(self.output_dir / "GM_40x.csv")
        image_path = str(self.image_40x_path)
        
        print("  绘制分层可视化图像...")
        assign_layers_to_mask(
            wm_path, gm_path, 
            str(layers_path), 
            image_path,
            issave=True,
            save_dir=str(self.output_dir)
        )
        
        print(f"  分层可视化图像已保存到 {self.output_dir / 'layers_lines.png'}")
        print(f"  分层颜色Mask已保存到 {self.output_dir / 'layers_color_mask.png'}")
        print("[✓] 步骤5完成: 可视化")
        
    def run_full_pipeline(self, n_layers=6, method='peak_based', merge_layer23=False,
                          use_roi_mask=True, enhance=False, keep_negative_coords=True, force=False):
        """
        运行完整的Pipeline
        
        Args:
            n_layers: 目标层数（仅对kmeans/gmm方法有效，默认6层）
            method: 分层方法 ('peak_based', 'kmeans', 'gmm')
            merge_layer23: 是否合并第2层和第3层（仅对peak_based方法有效）
                           - False: 输出5层 (L1, L2, L3, L4, L5/6)
                           - True:  输出4层 (L1, L2/3, L4, L5/6)
            use_roi_mask: 细胞分割时是否使用灰质掩码
            enhance: 是否对40x图像进行对比度增强
            keep_negative_coords: 是否保留负坐标点，False时会过滤掉x<0或y<0的边界点
            force: 强制重新运行所有步骤，忽略已存在的缓存文件
        """
        print("\n" + "#"*60)
        print("# 密度分层Pipeline - 开始执行")
        if force:
            print("# [强制模式] 将重新运行所有步骤")
        else:
            print("# [智能模式] 将跳过已有结果的步骤")
        if merge_layer23:
            print("# [2/3层融合] 启用，输出4层: L1, L2/3, L4, L5/6")
        print("#"*60)
        
        # 加载元数据
        self._load_metadata()
        
        # 执行各步骤（支持缓存跳过）
        self.step1_process_4x_image(force=force)
        self.step2_coordinate_transform(force=force)
        self.step3_segment_cells(use_roi_mask=use_roi_mask, enhance=enhance, force=force)
        self.step4_analyze_density_and_layer(n_layers=n_layers, method=method, merge_layer23=merge_layer23, 
                                            keep_negative_coords=keep_negative_coords, force=force)
        self.step5_visualize(force=force)
        
        print("\n" + "#"*60)
        print("# 密度分层Pipeline - 执行完成")
        print("#"*60)
        print(f"\n输出文件保存在: {self.output_dir}")
        print("主要输出文件:")
        print(f"  - 4x分割结果: tissueMask.png, grayMask.png, whiteMask.png")
        print(f"  - 边界文件: GM.csv, WM.csv, GM_40x.csv, WM_40x.csv")
        print(f"  - 细胞分割: nuclei_centroids.csv, segmented_mask.png")
        print(f"  - 分层结果: segmented_layers.csv")
        print(f"  - 可视化: layers_lines.png, depth_density_layers_peak_based.png")


def main():
    """主函数"""
    # 配置输入输出路径
    INPUT_DIR = "input"
    OUTPUT_DIR = "output"
    
    # 创建Pipeline实例
    pipeline = LayerPipeline(INPUT_DIR, OUTPUT_DIR)
    
    # 运行完整Pipeline
    # force=False: 智能模式，自动跳过已有结果的步骤（默认）
    # force=True: 强制模式，重新运行所有步骤
    # method: 'peak_based'(推荐), 'kmeans', 'gmm'
    # merge_layer23: 是否合并第2/3层
    #   - False: 输出5层 (L1, L2, L3, L4, L5/6)
    #   - True:  输出4层 (L1, L2/3, L4, L5/6)
    pipeline.run_full_pipeline(
        n_layers=5,             # 分层数（仅对kmeans/gmm有效）
        method='peak_based',    # 使用基于峰值的分层算法
        merge_layer23=True,    # 设为True时合并L2和L3
        use_roi_mask=True,      # 使用灰质掩码限制细胞分割区域
        enhance=False,          # 不进行对比度增强
        force=False             # 智能模式，跳过已有结果
    )
    
    # 或者分步骤运行（便于调试，每步都支持force参数）:
    # pipeline._load_metadata()
    # pipeline.step1_process_4x_image(force=False)
    # pipeline.step2_coordinate_transform(force=False)
    # pipeline.step3_segment_cells(force=False)
    # pipeline.step4_analyze_density_and_layer(method='peak_based', merge_layer23=True, force=True)
    # pipeline.step5_visualize(force=True)


if __name__ == "__main__":
    main()
