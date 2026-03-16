import numpy as np
from stardist.models import StarDist2D
from csbdeep.utils import normalize
import matplotlib.pyplot as plt
from skimage import io
from skimage.color import rgb2gray
import cv2
import pandas as pd
import os
from PIL import Image
import tensorflow as tf

# 解除 PIL 图像大小限制（防止 DecompressionBombError）
Image.MAX_IMAGE_PIXELS = None
# Image.MAX_IMAGE_PIXELS = 300000000  # 3 亿像素

input_dir='input'
output_dir='output'

def enhance_contrast(image):
    # # 滚球算法进行背景校正
    # blurred = cv2.GaussianBlur(image, (111, 111), 0)
    # # 背景灰度乘以0.6
    # blurred = (blurred * 0.8).astype(np.uint8)
    # # 原图减去背景
    # corrected = cv2.subtract(image, blurred)
    # 形态学变换去除背景
    kernel_size = 51
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                        (kernel_size, kernel_size))
    background = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    corrected = cv2.subtract(image, background)
    # 增强亮度和对比度
    corrected = cv2.convertScaleAbs(corrected, alpha=1.5, beta=10)
    return corrected

def segment_nuclei(image, max_edge_length=5000, isfilter=True):

    # 加载图像
    img = image
    print(f"图片已加载，形状: {img.shape}, 数据类型: {img.dtype}")

    # 图像预处理：统一为单通道灰度图
    if img.ndim == 2:
        img = img[..., np.newaxis]
        print(f"灰度图，已增加通道维度，新形状: {img.shape}")
    # 如果是 RGBA，只用 RGB 部分
    if img.shape[-1] == 4:
        img = img[..., :3]
        print(f"RGBA图，已移除Alpha通道，新形状: {img.shape}")
    # 彩色图转灰度
    if img.ndim == 3 and img.shape[-1] == 3:
        print("彩色图，正在转换为灰度图...")
        img = rgb2gray(img)
    # 确保有通道维度
    if img.ndim == 2:
        img = img[..., np.newaxis]
        print(f"灰度图，已增加通道维度，新形状: {img.shape}")

    # # 图像缩放（如果过大）
    # if max(img.shape[0], img.shape[1]) > max_edge_length:
    #     scale_factor = max_edge_length / max(img.shape[0], img.shape[1])
    #     new_size = (int(img.shape[1] * scale_factor), int(img.shape[0] * scale_factor))
    #     print(f"图片过大，正在缩放到 {new_size}...")
    #     img_resized = cv2.resize(img[..., 0], new_size, interpolation=cv2.INTER_AREA)
    #     img = img_resized[..., np.newaxis]
    #     print(f"缩放完成，新形状: {img.shape}")

    # 图像归一化
    img_norm = normalize(img, 1, 99.8, axis=(0, 1))

    # 加载 StarDist 模型
    print("正在加载 StarDist 模型...")
    model = StarDist2D.from_pretrained('2D_versatile_fluo')

    print("模型加载完毕。")

    # 执行分割预测
    print("正在进行细胞核分割...")
    # 分块预测以节省内存
    # n_tiles 的维度必须与 img_norm 的维度匹配（高、宽、通道）
    h, w = img_norm.shape[0] // 1024, img_norm.shape[1] // 1376
    labels, details = model.predict_instances(img_norm, n_tiles=(h, w, 1))
    print(f"分割完成！检测到 {len(details['coord'])} 个细胞核。")

    if isfilter:
        # --- 过滤逻辑：去除过小和低置信度的细胞 ---
        n_instances = len(details['coord'])
        if n_instances > 0:
            # 计算每个标签的面积
            if labels.max() == n_instances:
                areas = np.bincount(labels.flat)[1:]  # 去掉背景0
                
                avg_area = np.mean(areas)
                area_thresh = 0.5 * avg_area
                prob_thresh = 0.5

                print(f"执行过滤: 平均面积={avg_area:.1f}, 面积阈值={area_thresh:.1f}, 置信度阈值={prob_thresh}")

                probs = details['prob']
                
                # 找出满足条件的索引
                keep_indices = []
                for i in range(n_instances):
                    if areas[i] >= area_thresh and probs[i] >= prob_thresh:
                        keep_indices.append(i)
                
                n_kept = len(keep_indices)
                print(f"过滤结果: 移除 {n_instances - n_kept} 个细胞, 保留 {n_kept} 个。")

                if n_kept < n_instances:
                    # 更新 details 字典
                    new_details = {}
                    new_details['coord'] = [details['coord'][i] for i in keep_indices]
                    new_details['points'] = details['points'][keep_indices]
                    new_details['prob'] = details['prob'][keep_indices]
                    if 'class_id' in details:
                        new_details['class_id'] = details['class_id'][keep_indices]
                    
                    details = new_details

                    # 更新 labels 图像
                    map_array = np.zeros(n_instances + 1, dtype=labels.dtype)
                    for new_id, old_idx in enumerate(keep_indices, start=1):
                        map_array[old_idx + 1] = new_id
                    
                    labels = map_array[labels]
            else:
                print("警告: 标签ID不连续或与实例数不匹配，跳过过滤步骤。")

    return labels, details


def save_segmentation_results(labels, details, output_dir):

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 提取细胞核中心点坐标
    centroids = details['points']
    polygons = details['coord']
    probs = details['prob']
    
    # 保存中心点坐标到 CSV
    centroids_df = pd.DataFrame(centroids, columns=['Y', 'X'])
    centroids_path = os.path.join(output_dir, 'nuclei_centroids.csv')
    centroids_df.to_csv(centroids_path, index=False)
    print(f"细胞核中心点坐标已保存到 {centroids_path}")
    
    # 保存细胞信息（包含概率）
    info_df = pd.DataFrame({
        'cell_id': range(1, len(centroids) + 1),
        'Y': centroids[:, 0],
        'X': centroids[:, 1],
        'probability': probs
    })
    info_path = os.path.join(output_dir, 'nuclei_info.csv')
    info_df.to_csv(info_path, index=False)
    print(f"细胞核信息已保存到 {info_path}")
    
    # 保存多边形轮廓坐标
    polygons_path = os.path.join(output_dir, 'nuclei_polygons.npy')
    np.save(polygons_path, polygons, allow_pickle=True)
    print(f"细胞核多边形轮廓已保存到 {polygons_path}")
    
    # 保存分割掩码图（标签图）
    mask_path = os.path.join(output_dir, 'segmented_mask.png')
    io.imsave(mask_path, labels.astype(np.uint16))
    print(f"分割掩码图已保存到 {mask_path}")
    
    return centroids_df


def visualize_overlay(labels, image, isave = False):
    original_image = image
    
    # 统一为 BGR 格式
    if original_image.ndim == 2:
        original_bgr = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    elif original_image.shape[2] == 4:
        original_bgr = cv2.cvtColor(original_image, cv2.COLOR_BGRA2BGR)
    else:
        original_bgr = original_image.copy()
    
    # 应用颜色映射
    colored_mask = cv2.applyColorMap(labels.astype(np.uint8), cv2.COLORMAP_JET)
    alpha = 0.5
    
    # 只在掩码>0的位置叠加颜色
    mask_bool = labels > 0
    mask_3ch = np.repeat(mask_bool[:, :, np.newaxis], 3, axis=2)
    blended = np.where(mask_3ch, 
                      cv2.addWeighted(colored_mask, alpha, original_bgr, 1 - alpha, 0), 
                      original_bgr)
    
    if isave:
        cv2.imwrite(f'{output_dir}\\overlay_result.png', blended)
        print(f"叠加图已保存为 {output_dir}\\overlay_result.png")


def visualize_heatmap(centroids_df, image_path, issave = False):
    # 读取原图
    orig_img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    h, w = orig_img.shape[:2]

    # 提取坐标
    ys = centroids_df['Y'].to_numpy()
    xs = centroids_df['X'].to_numpy()
    heat = np.zeros((h, w), dtype=np.float32)

    # 将每个中心点投影到像素网格
    for y, x in zip(ys, xs):
        iy = int(round(y))
        ix = int(round(x))
        if 0 <= iy < h and 0 <= ix < w:
            heat[iy, ix] += 1.0

    # 高斯平滑
    sigma = max(3, int(max(h, w) * 0.02))
    heat = cv2.GaussianBlur(heat, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma, 
                           borderType=cv2.BORDER_REPLICATE)

    # 归一化到 0-255
    if heat.max() > 0:
        heat_norm = (255.0 * (heat - heat.min()) / (heat.max() - heat.min())).astype(np.uint8)
    else:
        heat_norm = np.zeros_like(heat, dtype=np.uint8)

    # 生成彩色 heatmap
    colored_heat = cv2.applyColorMap(heat_norm, cv2.COLORMAP_JET)
    cv2.imwrite('nuclei_density_heatmap.png', colored_heat)
    print("密度热图已保存到 nuclei_density_heatmap.png")

    # 转换原图为 BGR
    if orig_img.ndim == 2:
        orig_bgr = cv2.cvtColor(orig_img, cv2.COLOR_GRAY2BGR)
    elif orig_img.shape[2] == 4:
        orig_bgr = cv2.cvtColor(orig_img, cv2.COLOR_BGRA2BGR)
    else:
        orig_bgr = orig_img.copy()

    # 叠加热图
    alpha = 0.5
    blended_all = cv2.addWeighted(colored_heat, alpha, orig_bgr, 1 - alpha, 0)

    mask_bool = heat_norm > 0
    mask_3ch = np.repeat(mask_bool[:, :, np.newaxis], 3, axis=2)
    overlay = np.where(mask_3ch, blended_all, orig_bgr)

    if issave:
        cv2.imwrite(f'{output_dir}\\overlay_heatmap.png', overlay)
        print(f"热图叠加已保存为 {output_dir}\\overlay_heatmap.png")


if __name__ == "__main__":
    image_path = f'{input_dir}\\40x.png'
    # image_path = f'{output_dir}\\enhanced_40x.png'
    image = io.imread(image_path)
    # image = enhance_contrast(image)
    # cv2.imwrite(f'{output_dir}\\enhanced_40x.png', image)
    
    labels, details = segment_nuclei(image, isfilter=True)
    centroids_df = save_segmentation_results(labels, details, output_dir)
    
    visualize_overlay(labels, image, isave=True)
    # visualize_heatmap(centroids_df, image_path, issave=True)