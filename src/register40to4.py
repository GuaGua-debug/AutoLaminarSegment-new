import cv2
import json



INPUT_PATH = "IO\\INPUT"
OUTPUT_PATH = "IO\\OUTPUT"

def read_json_center_and_scale(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    positions = data.get('positions', [])
    start = (float(positions[0]['x']), float(positions[0]['y']))

    pixel_scale = data.get('scan_info', {}).get('pixel_scale', 1.625)
    start = (start[0] - 1376 * pixel_scale / 2, start[1] - 1024 * pixel_scale / 2)

    return start


def register40to4(image4x, image40x, json4xStart, json40xStart):
    registered = image4x.copy()
    image40x = cv2.resize(image40x, (image40x.shape[1] // 10, image40x.shape[0] // 10))

    resolution4x = (image4x.shape[1], image4x.shape[0])
    resolution40x = (image40x.shape[1], image40x.shape[0])

    offset = (
        int((json40xStart[0] - json4xStart[0]) / 1.625),
        int((json40xStart[1] - json4xStart[1]) / 1.625)
    )

    for y in range(resolution40x[1]):
        for x in range(resolution40x[0]):
            x4x = x + offset[0]
            y4x = y + offset[1]
            if 0 <= x4x < resolution4x[0] and 0 <= y4x < resolution4x[1]:
                registered[y4x, x4x] = image40x[y, x]

    return registered



if __name__ == "__main__":
    image4x = cv2.imread(f"{INPUT_PATH}\\4x.png")
    # image40x = cv2.imread(f"{INPUT_PATH}\\40x.png")
    image40x = cv2.imread(f"{INPUT_PATH}\\layer_mask.png")
    # image4x = cv2.imread(f"{OUTPUT_PATH}\\OuterInnerPoints.png")
    # image40x = cv2.imread(f"{OUTPUT_PATH}\\40x_with_boundaries.png")
    
    json4xStart = read_json_center_and_scale(f"{INPUT_PATH}\\4x.json")
    json40xStart = read_json_center_and_scale(f"{INPUT_PATH}\\40x.json")


    #   → x
    # ↓ y

    registered = register40to4(image4x, image40x, json4xStart, json40xStart)
    cv2.imwrite(f"{OUTPUT_PATH}\\registered.png", registered)