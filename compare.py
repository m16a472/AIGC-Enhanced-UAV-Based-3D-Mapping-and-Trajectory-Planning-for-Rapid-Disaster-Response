import os
import argparse
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import torch
import lpips
from PIL import Image
import numpy as np
import csv

# 初始化LPIPS模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lpips_model = lpips.LPIPS(net='alex').to(device)


def load_image(image_path):
    """加载并转换图像为numpy格式"""
    img = Image.open(image_path).convert('RGB')
    img = np.array(img)
    return img


def calculate_ssim(img1, img2):
    """计算SSIM"""
    # 设置较小的win_size，确保不超过图像的尺寸，并且win_size为奇数
    win_size = min(img1.shape[:2]) if min(img1.shape[:2]) >= 7 else 3  # 小于7时使用3x3窗口
    if win_size % 2 == 0:  # 如果是偶数，减去1
        win_size -= 1
    return ssim(img1, img2, multichannel=True, win_size=win_size, channel_axis=2)


def calculate_psnr(img1, img2):
    """计算PSNR"""
    return psnr(img1, img2)


def calculate_lpips(img1, img2):
    """计算LPIPS"""
    img1 = torch.tensor(img1).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
    img2 = torch.tensor(img2).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
    return lpips_model(img1, img2).item()


def compare_images(gt_folder, render_folder):
    """比较两个文件夹下同名图片的SSIM、PSNR和LPIPS"""
    results = []
    ssim_values = []
    psnr_values = []
    lpips_values = []
    gt_files = os.listdir(gt_folder)
    render_files = os.listdir(render_folder)

    # 确保两文件夹中有同名图片
    common_files = set(gt_files).intersection(render_files)

    for file in common_files:
        gt_path = os.path.join(gt_folder, file)
        render_path = os.path.join(render_folder, file)

        # 加载图像
        gt_img = load_image(gt_path)
        render_img = load_image(render_path)
        print(file)

        # 计算SSIM、PSNR、LPIPS
        psnr_value = calculate_psnr(gt_img, render_img)
        ssim_value = calculate_ssim(gt_img, render_img)
        lpips_value = calculate_lpips(gt_img, render_img)

        ssim_values.append(ssim_value)
        psnr_values.append(psnr_value)
        lpips_values.append(lpips_value)
        print(ssim_value)
        print(psnr_value)
        print(lpips_value)
        results.append({
            "file": file,
            "SSIM": psnr_value,
            "PSNR": ssim_value,
            "LPIPS": lpips_value
        })

    # 计算每个指标的平均值
    avg_ssim = np.mean(ssim_values)
    avg_psnr = np.mean(psnr_values)
    avg_lpips = np.mean(lpips_values)

    print(f"psnr:{avg_psnr}")
    print(f"ssim:{avg_ssim}")
    print(f"lpips:{avg_lpips}")

    return results


def save_results(results, output_file):
    """保存结果到CSV文件"""
    with open(output_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["file", "SSIM", "PSNR", "LPIPS"])
        writer.writeheader()
        for result in results:
            writer.writerow(result)


def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='Compare images in GT and Render folders.')
    parser.add_argument('base_directory', type=str, help='The base directory containing the gt and render folders')
    parser.add_argument('--output', type=str, default='comparison_results.csv', help='Output CSV file name')

    args = parser.parse_args()

    # 定义 gt 和 render 文件夹路径
    gt_folder = os.path.join(args.base_directory, 'gt')
    render_folder = os.path.join(args.base_directory, 'renders')

    # 确保文件夹存在
    if not os.path.exists(gt_folder):
        print(f"Error: '{gt_folder}' does not exist!")
        return
    if not os.path.exists(render_folder):
        print(f"Error: '{render_folder}' does not exist!")
        return

    # 比较图片
    results = compare_images(gt_folder, render_folder)

    # 保存结果到 CSV 文件
    save_results(results, args.output)
    print(f"Comparison results saved to {args.output}")


if __name__ == '__main__':
    main()
