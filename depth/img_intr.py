import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS


# 读取EXIF信息中的焦距和图像分辨率
def get_exif_info(image_path):
    img = Image.open(image_path)
    exif_data = img._getexif()

    if not exif_data:
        raise ValueError("No EXIF data found")

    exif_info = {}

    for tag, value in exif_data.items():
        tag_name = TAGS.get(tag, tag)
        exif_info[tag_name] = value
    # print(exif_info)
    # 获取焦距（35mm）和图像分辨率
    focal_length = exif_info.get('FocalLengthIn35mmFilm', None)
    if focal_length is None:
        focal_length = exif_info.get('FocalLength', None)
        if focal_length is None:
            raise ValueError("No focal length found in EXIF data")

    # 图像宽度和高度
    width, height = img.size

    return focal_length, width, height


# 计算相机内参矩阵
def compute_camera_intrinsics(focal_length_mm, width_px, height_px, sensor_width_mm=36, sensor_height_mm=24):
    # # 计算每个像素的物理尺寸
    px_size_x = sensor_width_mm / width_px
    px_size_y = sensor_height_mm / height_px

    # # 焦距（以像素为单位）
    fx = focal_length_mm / px_size_x
    fy = focal_length_mm / px_size_y
    # 相机内参矩阵
    K = np.array([
        [fx, 0, width_px / 2],
        [0, fy, height_px / 2],
        [0, 0, 1]
    ])

    return K


def intr(image_path):
    # 读取EXIF信息并计算焦距、分辨率
    focal_length_mm, width_px, height_px = get_exif_info(image_path)

    # 计算相机内参矩阵
    K = compute_camera_intrinsics(focal_length_mm, width_px, height_px)
    return K
