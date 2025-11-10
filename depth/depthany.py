import cv2
import torch
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
from skimage.restoration import denoise_tv_chambolle
from torchvision.transforms import Compose
from submodules.DepthAnythingV2.depth_anything_v2.dpt import DepthAnythingV2
from guided_filter_pytorch.guided_filter import GuidedFilter
from skimage.restoration import denoise_tv_chambolle
import warnings


class DepthEstimator:
    warnings.filterwarnings("ignore")

    def __init__(self, filename="", image=None):
        """
        初始化深度估计类

        Args:
            filename (str): 图片文件路径
            transform (callable): 图片预处理函数
            depth_anything (callable): 深度估计模型
            DEVICE (torch.device): 深度估计的计算设备
        """
        self.filename = filename
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.depth_anything = self.depth_setv2()
        self.image = image

    # def depth_set(self):
    #     model_configs = {
    #         'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    #         'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    #         'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
    #     }
    #
    #     encoder = 'vitl'  # or 'vitb', 'vits'
    #     depth_anything = DepthAnything(model_configs[encoder]).to(self.DEVICE).eval()
    #     depth_anything.load_state_dict(
    #         torch.load(f'./submodules/DepthAnything/checkpoints/depth_anything_{encoder}14.pth'))
    #
    #     self.depth_transforms = Compose([
    #         Resize(
    #             width=518,
    #             height=518,
    #             resize_target=False,
    #             keep_aspect_ratio=True,
    #             ensure_multiple_of=14,
    #             resize_method='lower_bound',
    #             image_interpolation_method=cv2.INTER_CUBIC,
    #         ),
    #         NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #         PrepareForNet(),
    #     ])
    #     return depth_anything

    def depth_setv2(self):
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }

        encoder = 'vitl'  # or 'vitb', 'vits'
        depth_anything = DepthAnythingV2(**model_configs[encoder]).to(self.DEVICE).eval()
        depth_anything.load_state_dict(
            torch.load(f'./submodules/DepthAnythingV2/checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
        depth_anything = depth_anything.to(self.DEVICE).eval()

        return depth_anything

    def get_depth_map(self):
        """
        计算图片的深度图

        Returns:
            np.ndarray: 归一化后的深度图
        """
        # 读取图片并转换为RGB
        raw_image = cv2.imread(self.filename)
        height, width = raw_image.shape[:2]

        # 设置目标宽度
        target_width = 1600

        # 计算缩放比例
        scale_factor = target_width / width

        # 计算等比例缩放后的新高度
        new_width = target_width
        new_height = int(height * scale_factor)

        # 使用插值方式进行缩放
        resized_image = cv2.resize(raw_image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        # image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
        #
        # h, w = raw_image.shape[:2]
        # h = int(h * 1600 / w)
        # w = 1600

        # # 应用预处理转换
        # image = self.depth_transforms({'image': image})['image']
        # image = torch.from_numpy(image).unsqueeze(0).to(self.DEVICE)

        # 估计深度
        with torch.no_grad():
            depth = self.depth_anything.infer_image(resized_image, 518)
        # 插值到原图大小
        # depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
        # depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        #
        depth = torch.tensor(depth).to(self.DEVICE)
        return depth

        # 如果是 torch.Tensor，转为 numpy
        # depth_np = depth.squeeze().cpu().numpy() if torch.is_tensor(depth) else depth
        #
        # # Step 1: TV 去噪，消除平坦区域抖动
        # depth_tv = denoise_tv_chambolle(depth_np, weight=0.05)
        #
        # # Step 2: Guided Filter 保边平滑
        # gray_np = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        # gray_np = gray_np.astype(np.float32) / 255.0  # [H, W] float32
        #
        # # 构建引导图像 I: [1, 1, H, W]
        # I = torch.from_numpy(gray_np).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        #
        # # 被滤波图像 P（深度图）: [1, 1, H, W]
        # P = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        # I = I.to(self.DEVICE)
        # P = P.to(self.DEVICE)
        #
        # # Guided Filtering
        # guided_filter = GuidedFilter(r=8, eps=1e-3).to(self.DEVICE)
        # depth_filtered = guided_filter(I.float(), P.float())  # [1, 1, H, W]
        #
        # return depth_filtered

    def get_depth(self):
        """
        获取深度图

        Returns:
            np.ndarray: 深度图
        """
        return self.depth_map

    def save_depth(self, path):
        np.save('data' + '/' + path + '.npy', self.depth_map)
        depth = cv2.applyColorMap(self.depth_map, cv2.COLORMAP_INFERNO)
        cv2.imwrite(self.filename + '_depth.png', self.depth_map)

    def esdepth(self, h, w):
        # 应用预处理转换
        image = self.depth_transforms({'image': self.image})['image']
        self.image = torch.from_numpy(image).unsqueeze(0).to(self.DEVICE)
        with torch.no_grad():
            depth = self.depth_anything(self.image)

            # 插值到原图大小
        depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0

        # 转换为numpy数组
        depth = depth.cpu().numpy().astype(np.uint8)
        # print(depth)
        return depth


def normalize_tensor(tensor, lower_percentile=2, upper_percentile=98):
    """
    根据分位数归一化 Tensor
    """
    tensor_np = tensor.detach().cpu().numpy()
    lower_bound = np.percentile(tensor_np, lower_percentile)
    upper_bound = np.percentile(tensor_np, upper_percentile)

    tensor_clipped = np.clip(tensor_np, lower_bound, upper_bound)
    normalized = (tensor_clipped - lower_bound) / (upper_bound - lower_bound + 1e-8)
    return normalized


def save_tensor_as_heatmap(tensor, filepath="heatmap.png", colormap=cv2.COLORMAP_JET):
    """
    使用 OpenCV 将 Tensor 渲染为热力图并保存为图片
    Args:
        tensor: 输入的 PyTorch Tensor, 形状为 [H, W]
        filepath: 保存图片的路径
        colormap: OpenCV 的颜色映射 (默认 COLORMAP_JET)
    """
    if tensor.dim() == 3:
        tensor = tensor.squeeze()

        # 根据分位数归一化
    normalized_tensor = normalize_tensor(tensor)

    # 转换为 uint8 数据类型
    tensor_uint8 = (normalized_tensor * 255).astype(np.uint8)

    # 使用 OpenCV 的 applyColorMap 函数生成热力图
    heatmap = cv2.applyColorMap(tensor_uint8, colormap)

    # 保存图片
    cv2.imwrite(filepath, heatmap)
