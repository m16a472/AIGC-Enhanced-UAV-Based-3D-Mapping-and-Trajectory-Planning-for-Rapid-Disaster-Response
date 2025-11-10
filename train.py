#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint

from sklearn.linear_model import LinearRegression

from utils.loss_utils import *
from gaussian_renderer import render, network_gui, render_large
import sys
from scene import Scene, GaussianModel, LargeScene
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, render_net_image
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, Fine_OptimizationParams
from utils.camera_utils import loadCam_woImage
import torch
from torchvision import transforms
from depth.depthany import *

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, load=-1,
             first_iter=0):
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = LargeScene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getsTrainCameras().copy()
        # print(len(scene.getTrainCameras()))
        # print(len(viewpoint_stack))
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii, depth = render_pkg["render"], render_pkg[
            "viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg['surf_depth']
        gt_image = viewpoint_cam.original_image.cuda()
        if not torch.all(torch.isnan(depth)):
            depth[torch.isnan(depth)] = 0.0
        # depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        # if iteration < 7000:
        dep_path = os.path.join(dataset.source_path, "depth")
        orimage = os.path.join(dep_path, viewpoint_cam.image_name + ".pt")
        esdepth = torch.load(orimage)
        esdepth = (esdepth.max() - esdepth) + esdepth.min()
        # save_tensor_as_heatmap(depth.squeeze(), f"{dep_path}/{viewpoint_cam.image_name}_dep.png")
        # save_tensor_as_heatmap(esdepth.squeeze(), f"{dep_path}/{viewpoint_cam.image_name}_esdep.png")
        # loss_depth = depth_loss(esdepth, depth) / 10_000
        loss_depth = torch.abs((depth - esdepth)).mean() / 1_000


        Ll1 = l1_loss(image, gt_image)
        l1loss = (1.0 - opt.lambda_dssim) * Ll1
        ssimloss = opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss = l1loss + ssimloss
        # print(f"ssim 是{ssim(image, gt_image)}")
        # regularization
        lambda_normal = opt.lambda_normal if iteration > 7000 or len(dataset.pretrain_path) > 0 else 0.0
        lambda_dist = opt.lambda_dist if iteration > 3000 or len(dataset.pretrain_path) > 0 else 0.0

        rend_dist = render_pkg["rend_dist"]
        rend_normal = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * (normal_error).mean()
        dist_loss = lambda_dist * (rend_dist).mean()
        total_loss = loss + dist_loss + normal_loss

        # # loss
        # if iteration > 2000:
        #     total_loss = loss + dist_loss + normal_loss + loss_depth
        # else:
        #     total_loss = loss + dist_loss + normal_loss
        if iteration < 5000:
            if iteration < 3000:
                total_loss += loss_depth
                # ssimloss += loss_depth
            total_loss += loss_depth / 1000
            # ssimloss += loss_depth / 1000

        w = total_loss / ssimloss

        if iteration < opt.densify_until_iter:
        #if iteration < 8000:
            ssimloss.backward(retain_graph=True)
            with torch.no_grad():
                # torch.cuda.empty_cache()
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter]
                )
                # if w > 2.0:
                #     w *= 0.7
                # if w > 1.2:
                #     w = 1.2
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, w)
                # 确保第二次反向传播不会占用过多显存
            gaussians.optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
        total_loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log
            loss_depth = 0.4 * loss_depth.item() + 0.6 * loss_depth

            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    # "distort": f"{ema_dist_for_log:.{5}f}",
                    "depth": f"{loss_depth:.{5}f}",
                    "normal": f"{ema_normal_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(loss_dict)

                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if tb_writer is not None:
                tb_writer.add_scalar('train_loss_patches/dist_loss', ema_dist_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)

            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                            testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                # if iteration % 1000 == 0:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration, dataset)
            # print(gaussians.max_radii2D.shape)
            # print(visibility_filter.shape)
            # print(radii.shape)

            # Densification
            if iteration < opt.densify_until_iter:
                # gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                #                                                      radii[visibility_filter])
                # gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, 1.0)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent,
                                                size_threshold)

                if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        with torch.no_grad():
            if network_gui.conn == None:
                network_gui.try_connect(dataset.render_items)
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()
                    if custom_cam != None:
                        render_pkg = render(custom_cam, gaussians, pipe, background, scaling_modifer)
                        net_image = render_net_image(render_pkg, dataset.render_items, render_mode, custom_cam)
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2,
                                                                                                                   0).contiguous().cpu().numpy())
                    metrics_dict = {
                        "#": gaussians.get_opacity.shape[0],
                        "loss": ema_loss_for_log
                        # Add more metrics as needed
                    }
                    # Send the data
                    network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    # raise e
                    network_gui.conn = None


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc,
                    renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/reg_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getsTestCameras()},
                              {'name': 'train',
                               'cameras': [scene.getsTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                                           range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        from utils.general_utils import colormap
                        depth = render_pkg["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name),
                                             depth[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                             image[None], global_step=iteration)

                        try:
                            rend_alpha = render_pkg['rend_alpha']
                            rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                            surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                            tb_writer.add_images(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name),
                                                 rend_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name),
                                                 surf_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name),
                                                 rend_alpha[None], global_step=iteration)

                            rend_dist = render_pkg["rend_dist"]
                            rend_dist = colormap(rend_dist.cpu().numpy()[0])
                            tb_writer.add_images(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name),
                                                 rend_dist[None], global_step=iteration)
                        except:
                            pass

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                                 gt_image[None], global_step=iteration)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        torch.cuda.empty_cache()


def es_pacth(path):
    dep_folder = os.path.join(path, 'depth')
    os.makedirs(dep_folder, exist_ok=True)
    images_folder = os.path.join(path, 'images')
    if not os.path.exists(images_folder):
        print(f"the path {images_folder} don't exist")
        return
    for image_name in os.listdir(images_folder):
        if image_name.endswith(('.png', '.jpg', '.jpeg', '.JPG')):
            image_path = os.path.join(images_folder, image_name)
            pt_file_name = f"{os.path.splitext(image_name)[0]}.pt"  # 获取去除扩展名后的文件名
            pt_file_path = os.path.join(dep_folder, pt_file_name)
            esdepth = DepthEstimator(filename=image_path).get_depth_map()
            #esdepth = retrenfer(esdepth)
            torch.save(esdepth, pt_file_path)
            save_tensor_as_heatmap(esdepth.squeeze(), f"{dep_folder}/{os.path.splitext(image_name)[0]}_esdep.png")


def retrenfer(depth_tensor):
    h, w = depth_tensor.shape

    # 计算每行的平均值
    row_means = depth_tensor.mean(dim=1).cpu().numpy()

    # 用行索引作为特征，拟合线性趋势
    row_indices = np.arange(h).reshape(-1, 1)
    reg = LinearRegression().fit(row_indices, row_means)

    # 获取线性拟合的趋势
    trend = reg.predict(row_indices)

    # 构造校正矩阵
    trend_tensor = torch.from_numpy(trend).view(-1, 1).repeat(1, w).to(depth_tensor.device)

    # 校正深度图
    depth_tensor_corrected = depth_tensor - trend_tensor

    return depth_tensor_corrected


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)

    # if lp.block_id < 0 == 0:
    #     print("预训练")
    #     op = OptimizationParams(parser)
    # else:
    #     op = Fine_OptimizationParams(parser)
    # op = Fine_OptimizationParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    # parser.add_argument('--block_id', type=int, default=-1)
    parser.add_argument('--port', type=int, default=6004)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--dep", action='store_true', default=False)
    parser.add_argument("--load", type=int, default=-1)
    parser.add_argument("--first", type=int, default=-0)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    print(len(args.pretrain_path))
    # os.path.join(dataset.source_path, "depth")
    # if args.block_id < 0 and len(args.pretrain_path) == 0 :
    if args.dep:
        es_pacth(args.source_path)
        print("完成深度计算")
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations,
             args.checkpoint_iterations, args.start_checkpoint, load=args.load, first_iter=args.first)

    # All done
    print("\nTraining complete.")
