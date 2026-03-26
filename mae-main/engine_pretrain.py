# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched
import logging

# 设置日志
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.StreamHandler(),
                        logging.FileHandler('train_log.txt')  # 将日志同时输出到控制台和文件
                    ])
logger = logging.getLogger()

#
# def train_one_epoch_1d(model: torch.nn.Module,
#                        data_loader: Iterable, optimizer: torch.optim.Optimizer,
#                        device: torch.device, epoch: int, loss_scaler,
#                        log_writer=None,
#                        args=None,
#                        model_without_ddp=None):
#     model.train()  # 确保模型是训练模式
#     data_iter_step = 0
#     total_loss = 0.0  # 用于累计损失
#     total_samples = 0  # 用于累计训练样本数
#     logger.info(f"Starting epoch {epoch} training...")
#     for ir_spectra in data_loader:
#         # 将数据移动到设备上
#         ir_spectra = ir_spectra.to(device, non_blocking=True)
#
#         # 前向传播和计算损失
#         with torch.cuda.amp.autocast():  # 自动混合精度
#             loss, _, _ = model(ir_spectra, mask_ratio=args.mask_ratio)
#
#         # 检查损失值是否为NaN或无穷大
#         if not torch.isfinite(loss):
#             print(f"Invalid loss value: {loss}. Stopping training.")
#             break  # 停止训练
#
#         # 反向传播和梯度更新
#         loss = loss / args.accum_iter  # 如果使用梯度累积，平均损失
#         loss_scaler(loss, optimizer, parameters=model.parameters(), update_grad=True)
#
#         # 梯度累积后，清除梯度
#         if (data_iter_step + 1) % args.accum_iter == 0:
#             optimizer.zero_grad()
#         data_iter_step += 1
#         total_loss += loss.item()
#         total_samples += ir_spectra.size(0)
#
#         # 记录和日志
#         if log_writer is not None and (data_iter_step + 1) % args.accum_iter == 0:
#             avg_loss = total_loss / total_samples  # 平均损失
#             log_writer.add_scalar('train_loss', avg_loss, epoch * len(data_loader) + data_iter_step)
#             logger.info(f"Epoch {epoch}, Step {data_iter_step}, Avg Loss: {avg_loss:.4f}")
#             total_loss = 0.0  # 重置损失累加器
#             total_samples = 0  # 重置样本数累加器
#
#         # 保存模型
#         if data_iter_step % (2e4) == 0:
#             misc.save_model(
#                 args=args, model= model_without_ddp, model_without_ddp=model, optimizer=optimizer,
#                 loss_scaler=loss_scaler, epoch=epoch)
#
#         torch.cuda.synchronize()  # 确保所有GPU操作同步
#
#     print(f"Epoch {epoch} training completed.")
def train_one_epoch_1d(model, data_loader, optimizer, device, epoch, loss_scaler,
                       log_writer=None, args=None, model_without_ddp=None):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    total_samples = 0
    logger.info(f"Starting epoch {epoch} training...")

    for data_iter_step, ir_spectra in enumerate(data_loader):
        ir_spectra = ir_spectra.to(device, non_blocking=True)

        if not torch.isfinite(ir_spectra).all():
            raise RuntimeError(f"Non-finite input detected at step {data_iter_step}")

        if data_iter_step % args.accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer,
                data_iter_step / len(data_loader) + epoch,
                args
            )

        with torch.cuda.amp.autocast(enabled=True):
            loss, _, _ = model(ir_spectra, mask_ratio=args.mask_ratio)

        if not torch.isfinite(loss):
            raise RuntimeError(f"Invalid loss at epoch {epoch}, step {data_iter_step}: {loss.item()}")

        loss_value = loss.item()
        loss = loss / args.accum_iter

        loss_scaler(
            loss,
            optimizer,
            clip_grad=1.0,
            parameters=model.parameters(),
            update_grad=((data_iter_step + 1) % args.accum_iter == 0)
        )

        if (data_iter_step + 1) % args.accum_iter == 0:
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss_value * ir_spectra.size(0)
        total_samples += ir_spectra.size(0)

    optimizer.zero_grad(set_to_none=True)
    avg_loss = total_loss / max(total_samples, 1)
    logger.info(f"Epoch {epoch} done, avg loss={avg_loss:.6f}")
    return {"loss": avg_loss}



def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            loss, _, _ = model(samples, mask_ratio=args.mask_ratio)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
