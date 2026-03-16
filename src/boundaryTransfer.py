import json
import csv
import os
import cv2
import numpy as np

# 默认输入输出路径
input_path = "input"
output_path = "output"

def read_json_center_and_scale(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    scan_info = data.get('scan_info', {})
    pixel_scale = float(scan_info.get('pixel_scale', 1.0))
    resolution = tuple(scan_info.get('camera_resolution', [1376, 1024]))
    positions = data.get('positions', [])
    if not positions:
        raise RuntimeError(f"{json_path} 中没有 positions")
    center = (float(positions[0]['x']), float(positions[0]['y']))
    return center, pixel_scale, resolution

def read_points_from_csv(csv_path):
    pts = []
    if not os.path.exists(csv_path):
        print(f"警告: CSV 文件不存在: {csv_path}")
        return pts
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        # 如果 header 看起来是数字则把它当作点
        try:
            float(header[0])
            f.seek(0)
            reader = csv.reader(f)
        except Exception:
            pass
        for row in reader:
            if not row:
                continue
            try:
                x = float(row[0])
                y = float(row[1])
            except Exception:
                continue
            pts.append((x, y))
    return pts

def convert_points(points, center_from, scale_from, center_to, scale_to, image_resolution):
    """
    将像素坐标从一张图 (scale_from, center_from) 映射到另一张图 (scale_to, center_to)。
    image_resolution: (width, height)
    坐标系：x 向右，y 向下。中心点对应像素中心 (width/2, height/2)。
    """
    w, h = image_resolution
    cx_px = w / 2.0
    cy_px = h / 2.0

    cx_from, cy_from = center_from
    cx_to, cy_to = center_to

    converted = []
    for (u, v) in points:
        # 物理坐标（以同一单位，与 json 中的物理坐标一致）
        phys_x = cx_from + (u - cx_px) * scale_from
        phys_y = cy_from + (v - cy_px) * scale_from

        # 转换到目标图像像素坐标
        u2 = (phys_x - cx_to) / scale_to + cx_px
        v2 = (phys_y - cy_to) / scale_to + cy_px

        converted.append((int(round(u2)), int(round(v2))))
    return converted

def save_points_to_csv(points, out_path):
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['x','y'])
        for x,y in points:
            writer.writerow([x,y])

def draw_points_on_image(image_path, points_wm, points_gm, out_path):
    # 尝试读取图片；如果不存在则创建空白画布
    if os.path.exists(image_path):
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    else:
        # 默认分辨率 1376x1024（如果 JSON 中分辨率不同可在调用处调整）
        img = 255 * np.ones((1024, 1376, 3), dtype=np.uint8)

    # # 绘制 WM（红）和 GM（绿）
    # def draw_list(pts, color):
    #     if not pts:
    #         return
    #     for i, p in enumerate(pts):
    #         cv2.circle(img, p, 2, color, -1)
    #         if i > 0:
    #             cv2.line(img, pts[i-1], pts[i], color, 25)

    # 绘制 WM和GM散点
    def draw_list(pts, color):
        if not pts:
            return
        for p in pts:
            cv2.circle(img, p, 25, color, -1)

    # # 绘制polylines
    # def draw_list(pts, color):
    #     if not pts:
    #         return
    #     pts_array = np.array(pts, np.int32)
    #     pts_array = pts_array.reshape((-1, 1, 2))
    #     cv2.polylines(img, [pts_array], isClosed=False, color=color, thickness=25)



    draw_list(points_wm, (255,255,0))
    draw_list(points_gm, (255,0,255))

    # 保存并返回路径
    cv2.imwrite(out_path, img)
    return out_path

def convert_mask_4x_to_40x(mask_4x, center_4x, scale_4x, center_40x, scale_40x, res_4x, output_size):
    """
    将 4x 掩码通过坐标转换映射到 40x 图像坐标系。
    使用仿射变换实现。
    """
    w, h = res_4x
    cx_px = w / 2.0
    cy_px = h / 2.0

    cx_from, cy_from = center_4x
    cx_to, cy_to = center_40x

    # 计算缩放比例
    scale_ratio = scale_4x / scale_40x

    # 计算4x图像中心在40x图像中的位置
    # 4x图像中心的物理坐标就是 center_4x
    # 转换到40x像素坐标
    center_4x_in_40x_u = (cx_from - cx_to) / scale_40x + cx_px
    center_4x_in_40x_v = (cy_from - cy_to) / scale_40x + cy_px

    # 构建仿射变换矩阵
    # 从4x像素 (u, v) 到 40x像素 (u', v'):
    # u' = (u - cx_px) * scale_ratio + center_4x_in_40x_u
    # v' = (v - cy_px) * scale_ratio + center_4x_in_40x_v
    # 即:
    # u' = scale_ratio * u + (center_4x_in_40x_u - cx_px * scale_ratio)
    # v' = scale_ratio * v + (center_4x_in_40x_v - cy_px * scale_ratio)

    tx = center_4x_in_40x_u - cx_px * scale_ratio
    ty = center_4x_in_40x_v - cy_px * scale_ratio

    # 仿射变换矩阵 (2x3)
    M = np.array([
        [scale_ratio, 0, tx],
        [0, scale_ratio, ty]
    ], dtype=np.float32)

    # 应用仿射变换
    mask_40x = cv2.warpAffine(mask_4x, M, output_size, flags=cv2.INTER_NEAREST)
    return mask_40x

def main():
    csv_wm = f"{output_path}\\WM.csv"
    csv_gm = f"{output_path}\\GM.csv"
    json_4x = f"{input_path}\\4x.json"
    json_40x = f"{input_path}\\40x.json"
    image_40x = f"{input_path}\\40x.png"
    out_csv_wm = f"{output_path}\\WM_40x.csv"
    out_csv_gm = f"{output_path}\\GM_40x.csv"
    out_image = f"{output_path}\\40x_with_boundaries.png"
    gray_mask_4x_path = f"{output_path}\\grayMask.png"
    # 读取 json 中的中心物理坐标与像素缩放
    center_4x, scale_4x, res_4x = read_json_center_and_scale(json_4x)
    center_40x, scale_40x, res_40x = read_json_center_and_scale(json_40x)
    # 打印全部信息
    print(f"4x - Center: {center_4x}, Scale: {scale_4x}, Resolution: {res_4x}")
    print(f"40x - Center: {center_40x}, Scale: {scale_40x}, Resolution: {res_40x}")

    # 确认分辨率一致
    if tuple(res_4x) != tuple(res_40x):
        print("警告: 两张图像分辨率不一致，结果可能不正确")
    image_resolution = tuple(res_4x)

    # 读取原始 CSV 点（4x 像素坐标）
    pts_wm_4x = read_points_from_csv(csv_wm)
    pts_gm_4x = read_points_from_csv(csv_gm)

    print(f"读取 WM 点: {len(pts_wm_4x)}，GM 点: {len(pts_gm_4x)}")
    # 坐标转换
    pts_wm_40x = convert_points(pts_wm_4x, center_4x, scale_4x, center_40x, scale_40x, image_resolution)
    pts_gm_40x = convert_points(pts_gm_4x, center_4x, scale_4x, center_40x, scale_40x, image_resolution)

    # 读取40x图像分辨率
    img_40x = cv2.imread(image_40x, cv2.IMREAD_COLOR)
    image_resolution = (img_40x.shape[1], img_40x.shape[0])
    # 遍历 pts_wm_40x，剔除超出图像范围的点
    # pts_wm_40x = [p for p in pts_wm_40x if 0 <= p[0] < image_resolution[0] and 0 <= p[1] < image_resolution[1]]
    # pts_gm_40x = [p for p in pts_gm_40x if 0 <= p[0] < image_resolution[0] and 0 <= p[1] < image_resolution[1]]

    # 保存转换后的 CSV
    save_points_to_csv(pts_wm_40x, out_csv_wm)
    save_points_to_csv(pts_gm_40x, out_csv_gm)
    print(f"已保存转换后 CSV: {out_csv_wm}, {out_csv_gm}")

    # empty_mask = np.zeros((image_resolution[1], image_resolution[0]), dtype=np.uint8)
    # # empty_mask中填充gm最低点和wm最高点之间的区域
    # # y轴向下增加！
    # min_gm_y = min(p[1] for p in pts_gm_40x)
    # max_wm_y = max(p[1] for p in pts_wm_40x)
    # if min_gm_y < max_wm_y:
    #     empty_mask[min_gm_y:max_wm_y, :] = 255
    #     cv2.imwrite(f"{output_path}\\boundaryMask.png", empty_mask)
    
    # roi = cv2.bitwise_and(img_40x, img_40x, mask=empty_mask)
    # cv2.imwrite(f"{output_path}\\roiImage.png", roi)

    # 在 40x 图像上绘制并保存
    out_img_path = draw_points_on_image(image_40x, pts_wm_40x, pts_gm_40x, out_image)
    print(f"已生成绘制图像: {out_img_path}")

    # ===== 新增：读取grayMask并转换到40x坐标系 =====
    if os.path.exists(gray_mask_4x_path):
        gray_mask_4x = cv2.imread(gray_mask_4x_path, cv2.IMREAD_GRAYSCALE)
        # 将4x掩码转换到40x坐标系
        gray_mask_40x = convert_mask_4x_to_40x(
            gray_mask_4x, center_4x, scale_4x, center_40x, scale_40x, 
            res_4x, image_resolution
        )
        # 保存转换后的掩码
        cv2.imwrite(f"{output_path}\\grayMask_40x.png", gray_mask_40x)
        print(f"已保存转换后掩码: {output_path}\\grayMask_40x.png")
        
        # 与40x图像相与得到新的roiImage
        roi_gray = cv2.bitwise_and(img_40x, img_40x, mask=gray_mask_40x)
        cv2.imwrite(f"{output_path}\\roiImage.png", roi_gray)
        print(f"已保存灰质区域ROI图像: {output_path}\\roiImage.png")
    else:
        print(f"警告: 掩码文件不存在: {gray_mask_4x_path}")

if __name__ == "__main__":
    main()