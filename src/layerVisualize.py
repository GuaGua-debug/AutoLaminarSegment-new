import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import cv2
import pandas as pd
import os
from scipy.interpolate import splprep, splev


# 默认输入输出路径（可通过参数覆盖）
input_dir = "input"
output_dir = "output"


def get_coords(df):
    """从DataFrame中获取坐标，兼容大小写列名"""
    if 'x' in df.columns and 'y' in df.columns:
        return df[['x', 'y']].values
    elif 'X' in df.columns and 'Y' in df.columns:
        return df[['X', 'Y']].values
    else:
        return df.iloc[:, :2].values


def smooth_boundary_points(pts, smoothing_factor=0.01):
    """
    使用样条曲线平滑边界点，使绘制的边界更加光滑
    """
    if len(pts) < 4:
        return pts
    
    # 去除重复点
    unique_pts = []
    for i, pt in enumerate(pts):
        if i == 0 or not np.allclose(pt, pts[i-1]):
            unique_pts.append(pt)
    pts = np.array(unique_pts)
    
    if len(pts) < 4:
        return pts
    
    try:
        # 使用样条曲线拟合
        x, y = pts[:, 0], pts[:, 1]
        tck, u = splprep([x, y], s=len(pts) * smoothing_factor, per=False)
        u_new = np.linspace(0, 1, len(pts) * 2)
        x_new, y_new = splev(u_new, tck)
        return np.column_stack([x_new, y_new]).astype(np.int32)
    except Exception:
        return pts


def draw_boundary_curve(img, pts, color, thickness=3, label=None, label_pos='center'):
    """
    绘制平滑的边界曲线
    
    Args:
        img: 要绘制的图像
        pts: 边界点坐标数组 (N, 2)
        color: BGR颜色
        thickness: 线条粗细
        label: 边界标签文字
        label_pos: 标签位置 ('center', 'start', 'end')
    """
    if len(pts) < 2:
        return
    
    # 平滑边界点
    smooth_pts = smooth_boundary_points(pts)
    
    # 转换为OpenCV格式
    pts_cv = smooth_pts.reshape((-1, 1, 2)).astype(np.int32)
    
    # 绘制曲线
    cv2.polylines(img, [pts_cv], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)
    
    # 添加标签
    if label:
        if label_pos == 'center':
            idx = len(smooth_pts) // 2
        elif label_pos == 'start':
            idx = min(10, len(smooth_pts) - 1)
        else:  # end
            idx = max(0, len(smooth_pts) - 10)
        
        label_pt = smooth_pts[idx]
        
        # 绘制标签背景
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
        cv2.rectangle(img, 
                     (int(label_pt[0]) - 5, int(label_pt[1]) - text_h - 10),
                     (int(label_pt[0]) + text_w + 5, int(label_pt[1]) + 5),
                     (0, 0, 0), -1)
        cv2.putText(img, label, (int(label_pt[0]), int(label_pt[1])),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3, cv2.LINE_AA)

def calculate_cell_depths(centroids_df, wm_path, gm_path):
    from scipy.spatial.distance import cdist
    
    # 读取边界数据
    try:
        wm_df = pd.read_csv(wm_path)
        gm_df = pd.read_csv(gm_path)
    except Exception as e:
        print(f"读取边界文件失败: {e}")
        return None

    wm_pts = get_coords(wm_df)
    gm_pts = get_coords(gm_df)
    cell_pts = centroids_df[['X', 'Y']].values

    print(f"正在计算深度... 细胞数:{len(cell_pts)}, WM点数:{len(wm_pts)}, GM点数:{len(gm_pts)}")

    # 计算距离 (注意：如果点很多，cdist可能会内存溢出，这里假设数据量适中)
    # 如果数据量大，应该用 KDTree
    dist_wm = cdist(cell_pts, wm_pts).min(axis=1)
    dist_gm = cdist(cell_pts, gm_pts).min(axis=1)

    depths = dist_gm / (dist_wm + dist_gm + 1e-8)
    return depths

def assign_layers_to_mask(wm_path, gm_path, layers_csv_path, image_path, issave=True, save_dir=None):
    """
    在图像上绘制分层线进行可视化
    
    Args:
        wm_path: 白质边界CSV文件路径
        gm_path: 灰质边界CSV文件路径
        layers_csv_path: 分层结果CSV文件路径
        image_path: 原始图像路径
        issave: 是否保存结果
        save_dir: 保存目录（如果为None则使用默认output_dir）
    """
    global output_dir
    if save_dir is not None:
        output_dir = save_dir
    
    # 读取分层结果
    layers_df = pd.read_csv(layers_csv_path)
    layers = layers_df.to_dict('records')
    print(f"已加载分层定义: {len(layers)} 层")

    # 读取原图用于可视化
    orig_img = cv2.imread(image_path)
    h, w = orig_img.shape[:2]
    vis_img = orig_img.copy()

    # 计算全图深度场 (Depth Map) 用于绘制分层线
    wm_df = pd.read_csv(wm_path)
    gm_df = pd.read_csv(gm_path)

    # 获取边界点坐标
    wm_pts = get_coords(wm_df)
    gm_pts = get_coords(gm_df)
    
    print(f"GM边界点数: {len(gm_pts)}, WM边界点数: {len(wm_pts)}")

    # def get_dist_map(pts_df, h, w):
    #     mask = np.zeros((h, w), dtype=np.uint8)
    #     cols = pts_df.columns
    #     x_col = 'x' if 'x' in cols else 'X'
    #     y_col = 'y' if 'y' in cols else 'Y'
        
    #     pts = pts_df[[x_col, y_col]].values
    #     for pt in pts:
    #         px, py = int(pt[0]), int(pt[1])
    #         if 0 <= px < w and 0 <= py < h:
    #             mask[py, px] = 255
        
    #     dist_mask = 255 - mask
    #     dist = cv2.distanceTransform(dist_mask, cv2.DIST_L2, 5)
    #     return dist

    def get_dist_map(pts_df, h, w):
        cols = pts_df.columns
        x_col = 'x' if 'x' in cols else 'X'
        y_col = 'y' if 'y' in cols else 'Y'
        pts = pts_df[[x_col, y_col]].values
        
        # 计算所有点的范围，确定需要扩展的边界
        min_x, max_x = int(pts[:, 0].min()), int(pts[:, 0].max())
        min_y, max_y = int(pts[:, 1].min()), int(pts[:, 1].max())
        
        # 计算偏移量（将坐标平移到正数范围）
        offset_x = max(0, -min_x)
        offset_y = max(0, -min_y)
        
        # 扩展后的画布尺寸
        ext_w = max(w, max_x + 1) + offset_x
        ext_h = max(h, max_y + 1) + offset_y
        
        # 在扩展画布上标记边界点
        mask = np.zeros((ext_h, ext_w), dtype=np.uint8)
        for pt in pts:
            px, py = int(pt[0]) + offset_x, int(pt[1]) + offset_y
            if 0 <= px < ext_w and 0 <= py < ext_h:
                mask[py, px] = 255
        
        # 计算距离变换
        dist_mask = 255 - mask
        dist = cv2.distanceTransform(dist_mask, cv2.DIST_L2, 5)
        
        # 裁剪回原始图像大小
        dist_cropped = dist[offset_y:offset_y + h, offset_x:offset_x + w]
        return dist_cropped


    print("正在计算全图深度场...")
    dist_wm = get_dist_map(wm_df, h, w)
    dist_gm = get_dist_map(gm_df, h, w)
    total_dist = dist_wm + dist_gm + 1e-8
    depth_map = dist_gm / total_dist

    # # 绘制深度场图
    # depth_map_normalized = (depth_map * 255).astype(np.uint8)
    # depth_colored = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_JET)
    # if issave:
    #     cv2.imwrite(f'{output_dir}\\depth_map.png', depth_colored)
    #     print(f"深度场图已保存为 {output_dir}\\depth_map.png")

    line_thickness = max(50, min(h, w) // 200)
    # ============ 绘制GM和WM边界曲线 ============
    gm_color = (255, 0, 255)
    gm_thickness = max(5, min(h, w) // 300)  # 根据图像大小自适应线条粗细
    draw_boundary_curve(vis_img, gm_pts, gm_color, thickness=line_thickness, label="GM", label_pos='center')
    
    wm_color = (255, 255, 0)
    draw_boundary_curve(vis_img, wm_pts, wm_color, thickness=line_thickness, label="WM", label_pos='center')
    

    # ============ 绘制分层线 ============
    cmap = plt.get_cmap('tab10')
    
    for i, layer in enumerate(layers):
        boundary_depth = layer['end']
        if boundary_depth > 0.99: continue

        rgba = cmap(i % 10)
        color_bgr = (int(rgba[2]*255), int(rgba[1]*255), int(rgba[0]*255))

        thresh_map = (depth_map <= boundary_depth).astype(np.uint8) * 255
        # 检查是否是Layer 2或Layer 5（支持字符串和整数类型）
        layer_id = str(layer['layer'])
        if layer_id == '2' or layer_id == '5' or layer_id == '5/6':
            line_type = cv2.LINE_AA
            line_style = (5, 10)  # 5像素实线，10像素空白
        else:
            line_type = cv2.LINE_AA
            line_style = None

        contours, _ = cv2.findContours(thresh_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # 过滤掉位于图像边界的点（x==0, x==w-1, y==0, y==h-1）以避免绘制贴边线
        filtered_contours = []
        for c in contours:
            pts = c.reshape(-1, 2)
            # 保留不在边界上的点
            mask = (pts[:, 0] > 0) & (pts[:, 0] < (w - 1)) & (pts[:, 1] > 0) & (pts[:, 1] < (h - 1))
            if mask.sum() >= 2:
                kept = pts[mask].astype(np.int32).reshape(-1, 1, 2)
                filtered_contours.append(kept)

        # 只有存在经过过滤后的轮廓才绘制
        if filtered_contours:
            # 忽略过远点：将轮廓按点间距拆分为连续段，仅绘制点间距不超过阈值的段
            segments = []
            # 距离阈值：取图像对角线的一个小比例作为最大允许间距（可根据需要调整）
            max_gap = np.hypot(w, h) * 0.05

            for c in filtered_contours:
                pts = c.reshape(-1, 2).astype(np.float32)
                if pts.shape[0] < 2:
                    continue

                # 计算相邻点距离
                d = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
                # 找到需要分割的位置（距离大于阈值）
                split_idx = np.where(d > max_gap)[0]

                start = 0
                for idx in split_idx:
                    seg = pts[start:idx+1]
                    if seg.shape[0] >= 2:
                        segments.append(seg.astype(np.int32).reshape(-1, 1, 2))
                    start = idx + 1

                # 添加最后一段
                seg = pts[start:]
                if seg.shape[0] >= 2:
                    segments.append(seg.astype(np.int32).reshape(-1, 1, 2))

            # 绘制所有连续段
            if segments:
                cv2.polylines(vis_img, segments, isClosed=False, color=color_bgr, thickness=line_thickness, lineType=cv2.LINE_AA)

                # 添加分层标签：基于最长的段
                best_seg = max(segments, key=lambda s: s.shape[0])
                if best_seg.shape[0] > 0:
                    pt = best_seg[best_seg.shape[0] // 2][0]
                    label_text = f"L{layer['layer']}"
                    font_scale = max(0.8, min(h, w) / 2000)
                    font_thickness = max(2, int(font_scale * 2))

                    # 绘制标签背景
                    (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                    cv2.rectangle(vis_img, 
                                 (pt[0] - 3, pt[1] - text_h - 5),
                                 (pt[0] + text_w + 3, pt[1] + 3),
                                 (0, 0, 0), -1)
                    cv2.putText(vis_img, label_text, (pt[0], pt[1]), 
                               cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_bgr, font_thickness, cv2.LINE_AA)
    
    # 尝试从output_dir或input_dir读取灰质掩码
    grayMask = cv2.imread(os.path.join(output_dir, 'grayMask_40x.png'), cv2.IMREAD_GRAYSCALE)
    if grayMask is None:
        grayMask = cv2.imread(os.path.join(input_dir, 'grayMask_40x.png'), cv2.IMREAD_GRAYSCALE)
    if grayMask is not None:
        # 创建3通道掩码
        mask_3ch = cv2.cvtColor(grayMask, cv2.COLOR_GRAY2BGR)
        # 将掩码区域设为黑色
        vis_img[mask_3ch == 0] = 0

    # # ============ 添加图例 ============
    # legend_x = 30
    # legend_y = 50
    # legend_spacing = 40
    # legend_font_scale = max(0.8, min(h, w) / 2500)
    # legend_thickness = max(2, int(legend_font_scale * 2))
    
    # # 图例背景
    # legend_items = [("GM Boundary", gm_color), ("WM Boundary", wm_color)]
    # for i, layer in enumerate(layers):
    #     rgba = cmap(i % 10)
    #     color_bgr = (int(rgba[2]*255), int(rgba[1]*255), int(rgba[0]*255))
    #     legend_items.append((f"Layer {layer['layer']}", color_bgr))
    
    # legend_height = len(legend_items) * legend_spacing + 20
    # legend_width = 250
    # cv2.rectangle(vis_img, (legend_x - 10, legend_y - 30), 
    #              (legend_x + legend_width, legend_y + legend_height - 20), 
    #              (40, 40, 40), -1)
    # cv2.rectangle(vis_img, (legend_x - 10, legend_y - 30), 
    #              (legend_x + legend_width, legend_y + legend_height - 20), 
    #              (200, 200, 200), 2)
    
    # for i, (label, color) in enumerate(legend_items):
    #     y_pos = legend_y + i * legend_spacing
    #     # 绘制颜色示例线
    #     cv2.line(vis_img, (legend_x, y_pos), (legend_x + 40, y_pos), color, 4, cv2.LINE_AA)
    #     # 绘制文字
    #     cv2.putText(vis_img, label, (legend_x + 55, y_pos + 5), 
    #                cv2.FONT_HERSHEY_SIMPLEX, legend_font_scale, (255, 255, 255), legend_thickness, cv2.LINE_AA)

    if issave:
        cv2.imwrite(f'{output_dir}\\layers_lines.png', vis_img)
        print(f"分层线图已保存为 {output_dir}\\layers_lines.png")

    # ============ 生成分层颜色填充Mask ============
    # 每层使用不同颜色填充，不包含任何文字标签
    layer_color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    # 定义层颜色（使用更鲜明的配色方案）
    layer_colors = [
        (255, 100, 100),   # Layer 1 - 浅红色
        (100, 255, 100),   # Layer 2 - 浅绿色
        (100, 100, 255),   # Layer 3 - 浅蓝色
        (255, 255, 100),   # Layer 4 - 黄色
        (255, 100, 255),   # Layer 5 - 粉色
        (100, 255, 255),   # Layer 6 - 青色
        (255, 180, 100),   # Layer 7 - 橙色（备用）
        (180, 100, 255),   # Layer 8 - 紫色（备用）
    ]
    
    # 先填充白质区域（最内层，depth >= 最后一层的end）
    # 白质用白色表示
    if len(layers) > 0:
        last_layer_end = layers[-1]['end']
        wm_region = (depth_map >= last_layer_end).astype(np.uint8) * 255
        layer_color_mask[wm_region == 255] = (255, 255, 255)  # 白色表示白质
    
    # 按层序填充区域（从外到内，即从GM到WM）
    for i, layer in enumerate(layers):
        layer_start = layer['start']
        layer_end = layer['end']
        
        # 获取当前层的颜色
        color_idx = i % len(layer_colors)
        fill_color = layer_colors[color_idx]
        
        # 创建当前层的mask：depth在[start, end]范围内
        layer_region = ((depth_map >= layer_start) & (depth_map < layer_end)).astype(np.uint8) * 255
        
        # 填充颜色
        layer_color_mask[layer_region == 255] = fill_color
    
    # 应用灰质掩码（只显示灰质区域内的分层）
    # 但保留白质区域的显示
    if grayMask is not None:
        # 创建包含灰质和白质的组合掩码
        if len(layers) > 0:
            last_layer_end = layers[-1]['end']
            wm_mask = (depth_map >= last_layer_end).astype(np.uint8) * 255
            combined_mask = cv2.bitwise_or(grayMask, wm_mask)
        else:
            combined_mask = grayMask
        mask_3ch = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)
        layer_color_mask[mask_3ch == 0] = 0
    
    if issave:
        cv2.imwrite(f'{output_dir}\\layers_color_mask.png', layer_color_mask)
        print(f"分层颜色Mask已保存为 {output_dir}\\layers_color_mask.png")

    # # 计算每个细胞的深度
    # depths = calculate_cell_depths(centroids_df, wm_path, gm_path)
    # if depths is None:
    #     return None, None

    # # 为每个 cell 分配层号
    # cell_layers = []
    # for d in depths:
    #     assigned = 0
    #     for L in layers:
    #         if d >= L['start'] and d <= L['end']:
    #             assigned = L['layer']
    #             break
    #     cell_layers.append(int(assigned))

    # # 生成 layer_mask 和 overlay
    # # 构建 label_id -> layer 映射
    # label_to_layer = {}
    # h, w = labels.shape[:2]
    # ys = centroids_df['Y'].to_numpy()
    # xs = centroids_df['X'].to_numpy()
    
    # for i, (y, x) in enumerate(zip(ys, xs)):
    #     iy = int(round(y))
    #     ix = int(round(x))
    #     if 0 <= iy < h and 0 <= ix < w:
    #         lid = int(labels[iy, ix])
    #     else:
    #         lid = 0
        
    #     if lid == 0: # 尝试邻域搜索
    #         rr = 3
    #         found = 0
    #         for dy in range(-rr, rr+1):
    #             for dx in range(-rr, rr+1):
    #                 ny, nx = iy+dy, ix+dx
    #                 if 0 <= ny < h and 0 <= nx < w:
    #                     v = int(labels[ny, nx])
    #                     if v > 0:
    #                         lid = v
    #                         found = 1
    #                         break
    #             if found: break
        
    #     if lid > 0 and lid not in label_to_layer:
    #         label_to_layer[lid] = cell_layers[i]

    # layer_mask = np.zeros_like(labels, dtype=np.uint8)
    # for lid, layerno in label_to_layer.items():
    #     layer_mask[labels == lid] = int(layerno)

    # # 可视化
    # if orig_img.ndim == 2:
    #     orig_bgr = cv2.cvtColor(orig_img, cv2.COLOR_GRAY2BGR)
    # elif orig_img.shape[2] == 4:
    #     orig_bgr = cv2.cvtColor(orig_img, cv2.COLOR_BGRA2BGR)
    # else:
    #     orig_bgr = orig_img.copy()

    # max_layer = int(layer_mask.max()) if layer_mask.max() > 0 else 1
    # colored_layers = np.zeros((h, w, 3), dtype=np.uint8)
    
    # for li in range(1, max_layer + 1):
    #     rgba = cmap((li - 1) % 10)
    #     rgb = (np.array(rgba[:3]) * 255).astype(np.uint8)
    #     bgr = rgb[::-1]
    #     colored_layers[layer_mask == li] = bgr

    # alpha = 0.5
    # blended_full = cv2.addWeighted(colored_layers, alpha, orig_bgr, 1 - alpha, 0)
    # mask_bool = layer_mask > 0
    # mask_3ch = np.repeat(mask_bool[:, :, np.newaxis], 3, axis=2)
    # blended = np.where(mask_3ch, blended_full, orig_bgr)

    # if issave:
    #     cv2.imwrite('labels_with_layers.png', layer_mask)
    #     cv2.imwrite('overlay_layers.png', blended)
    #     print("分层结果图已保存")

if __name__ == "__main__":
    # 加载分割结果
    centroids_df = pd.read_csv(f"{output_dir}\\nuclei_centroids.csv")
    layers_df = f'{output_dir}\\segmented_layers.csv'

    image_path = f'{input_dir}\\40x.png'
    wm_df = f"{input_dir}\\WM_40x.csv"
    gm_df = f"{input_dir}\\GM_40x.csv"

    # 调整列顺序为 ['X','Y']（如果需要）
    if list(centroids_df.columns[:2]) == ['Y','X']:
        df2 = pd.DataFrame({'X': centroids_df['X'].values, 'Y': centroids_df['Y'].values})
    else:
        df2 = centroids_df.rename(columns={centroids_df.columns[0]:'X', centroids_df.columns[1]:'Y'}) if 'X' not in centroids_df.columns else centroids_df

    assign_layers_to_mask(wm_df, gm_df, layers_df, image_path, issave=True)