import cv2
import json
import os

# 定义全局变量
annotations = {}
current_image = None
current_image_name = None
zoom_factor = 4  # 固定放大倍数（64x64 -> 512x512）
pan_offset = [0, 0]  # 平移偏移量
start_pan = None  # 用于记录鼠标拖动的起始点

def mouse_callback(event, x, y, flags, param):
    """
    鼠标回调函数，用于记录点击的坐标和类别。
    """
    global annotations, current_image_name, zoom_factor, pan_offset, start_pan

    # 将点击坐标映射回原始图像坐标
    orig_x = int((x + pan_offset[0]) / zoom_factor)
    orig_y = int((y + pan_offset[1]) / zoom_factor)

    if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击 -> 硒原子
        print(f"Se atom at ({orig_x}, {orig_y})")
        annotations[current_image_name].append({"x": orig_x, "y": orig_y, "class": "Se"})
    elif event == cv2.EVENT_RBUTTONDOWN:  # 右键点击 -> 碲原子
        print(f"Te atom at ({orig_x}, {orig_y})")
        annotations[current_image_name].append({"x": orig_x, "y": orig_y, "class": "Te"})
    elif event == cv2.EVENT_MBUTTONDOWN:  # 中键按下 -> 开始平移
        start_pan = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_MBUTTON:  # 中键拖动 -> 平移
        if start_pan is not None:
            dx = x - start_pan[0]
            dy = y - start_pan[1]
            pan_offset[0] -= dx  # 更新平移偏移量
            pan_offset[1] -= dy
            start_pan = (x, y)
    elif event == cv2.EVENT_MBUTTONUP:  # 中键释放 -> 停止平移
        start_pan = None

def annotate_images(image_dir, output_dir):
    """
    主函数，用于加载图像并进行标注。
    Args:
        image_dir (str): 图像文件夹路径。
        output_dir (str): 输出的 JSON 文件夹路径。
    """
    global annotations, current_image, current_image_name, zoom_factor, pan_offset

    # 获取所有图像文件
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print("No images found in the specified directory.")
        return

    # 创建输出文件夹（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    for image_file in image_files:
        current_image_name = image_file
        output_file = os.path.join(output_dir, f"{os.path.splitext(image_file)[0]}.json")

        # 如果之前有标注文件，加载已有标注
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                annotations[current_image_name] = json.load(f)
        else:
            annotations[current_image_name] = []

        # 加载图像
        image_path = os.path.join(image_dir, image_file)
        current_image = cv2.imread(image_path)
        if current_image is None:
            print(f"Failed to load image: {image_path}")
            continue

        # 创建窗口并设置鼠标回调
        cv2.namedWindow("Annotate", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Annotate", 512, 512)  # 固定窗口大小
        cv2.setMouseCallback("Annotate", mouse_callback)

        print(f"Annotating {image_file}... Press 's' to save and move to the next image, 'q' to quit, 'r' to undo.")
        while True:
            # 放大图像
            height, width = current_image.shape[:2]
            resized_image = cv2.resize(current_image, (width * zoom_factor, height * zoom_factor))

            # 平移图像
            canvas = resized_image[
                max(0, pan_offset[1]):min(height * zoom_factor, pan_offset[1] + 512),
                max(0, pan_offset[0]):min(width * zoom_factor, pan_offset[0] + 512)
            ]

            # 如果平移超出范围，用黑色填充
            canvas_padded = cv2.copyMakeBorder(
                canvas,
                top=max(0, 512 - canvas.shape[0]),
                bottom=0,
                left=max(0, 512 - canvas.shape[1]),
                right=0,
                borderType=cv2.BORDER_CONSTANT,
                value=[0, 0, 0]
            )

            # 在图像上绘制标注点
            for annotation in annotations[current_image_name]:
                color = (0, 255, 0) if annotation["class"] == "Se" else (0, 0, 255)  # 绿色: Se, 红色: Te
                cv2.circle(canvas_padded, 
                           (int(annotation["x"] * zoom_factor - pan_offset[0]), 
                            int(annotation["y"] * zoom_factor - pan_offset[1])), 
                           5, color, -1)

            # 显示图像
            cv2.imshow("Annotate", canvas_padded)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):  # 按 's' 保存并跳到下一张图像
                with open(output_file, 'w') as f:
                    json.dump(annotations[current_image_name], f, indent=4)
                print(f"Annotations saved to {output_file}.")
                break
            elif key == ord('q'):  # 按 'q' 退出程序
                print("Exiting...")
                cv2.destroyAllWindows()
                return
            elif key == ord('r'):  # 按 'r' 撤回最后一个标注
                if annotations[current_image_name]:
                    removed = annotations[current_image_name].pop()
                    print(f"Removed annotation: {removed}")
                else:
                    print("No annotations to undo.")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 设置图像文件夹路径和输出标注文件夹路径
    image_dir = "../data/images"  # 替换为你的图像文件夹路径
    output_dir = "../data/labels"  # 替换为你的标注文件夹路径

    # 开始标注
    annotate_images(image_dir, output_dir)