import argparse
import os
import importlib

import os.path as osp
import torch
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import fp16_compress_hook

from data_process.distreamify import VideoDataset
from model.i3d_arcface import I3D_ResNet
from model.losses import CombinedMarginLoss
from model.lr_scheduler import PolynomialLRWarmup
from model.partial_fc_v2 import PartialFC_V2
from torch.utils.data import DataLoader
from torch import distributed
from utils.loggings import AverageMeter, CallBackLogging, init_logging
from data_process.data_loader_ddp import get_dataloader

try:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    distributed.init_process_group("nccl")
except KeyError:
    rank = 0
    local_rank = 0
    world_size = 1
    distributed.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:12584",
        rank=rank,
        world_size=world_size,
    )

def get_config(config_file):
    assert config_file.startswith('configs/'), 'config file setting must start with configs/'
    temp_config_name = osp.basename(config_file)
    temp_module_name = osp.splitext(temp_config_name)[0]
    config = importlib.import_module("configs.base")
    cfg = config.config
    config = importlib.import_module("configs.%s" % temp_module_name)
    job_cfg = config.config
    cfg.update(job_cfg)
    if cfg.output is None:
        cfg.output = osp.join('work_dirs', temp_module_name)
    return cfg

def main(args):
    cfg = get_config(args.config)
    torch.cuda.set_device(local_rank)
    os.makedirs(cfg.output, exist_ok=True)
    init_logging(rank, cfg.output)

    video_dataset = VideoDataset(data_path=cfg.data, save_root=cfg.dataset_output, 
                                 max_frames=cfg.max_frame, recorded=cfg.dataset_recorded)
    num_image = len(video_dataset)
    num_classes = len(set(video_dataset.labels))
    train_loader = get_dataloader(video_dataset, local_rank,
                                  batch_size=cfg.batch_size,
                                  seed=cfg.seed,
                                  num_workers=cfg.num_workers)
    
    backbone = I3D_ResNet(layers=[3, 4, 14, 3], backbone_2d_path='./ckpt/backbone_2d.pth').cuda()
    backbone = torch.nn.parallel.DistributedDataParallel(
        module=backbone, broadcast_buffers=False, device_ids=[local_rank], bucket_cap_mb=16,
        find_unused_parameters=True)
    backbone.register_comm_hook(None, fp16_compress_hook)
    backbone.train()
    backbone._set_static_graph()

    margin_loss = CombinedMarginLoss(
        64,
        cfg.margin_list[0],
        cfg.margin_list[1],
        cfg.margin_list[2],
        cfg.interclass_filtering_threshold
    )

    if cfg.optimizer == "sgd":
        module_partial_fc = PartialFC_V2(
            margin_loss, cfg.embedding_size, num_classes,
            cfg.sample_rate, False)
        module_partial_fc.train().cuda()
        # TODO the params of partial fc must be last in the params list
        opt = torch.optim.SGD(
            params=[{"params": backbone.parameters()}, {"params": module_partial_fc.parameters()}],
            lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)

    elif cfg.optimizer == "adamw":
        module_partial_fc = PartialFC_V2(
            margin_loss, cfg.embedding_size, num_classes,
            cfg.sample_rate, False)
        module_partial_fc.train().cuda()
        opt = torch.optim.AdamW(
            params=[{"params": backbone.parameters()}, {"params": module_partial_fc.parameters()}],
            lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        raise

    cfg.total_batch_size = cfg.batch_size * world_size
    cfg.warmup_step = num_image // cfg.total_batch_size * cfg.warmup_epoch
    cfg.total_step = num_image // cfg.total_batch_size * cfg.num_epoch

    lr_scheduler = PolynomialLRWarmup(
        optimizer=opt,
        warmup_iters=cfg.warmup_step,
        total_iters=cfg.total_step)
    
    start_epoch = 0
    global_step = 0
    callback_logging = CallBackLogging(
        frequent=cfg.frequent,
        total_step=cfg.total_step,
        batch_size=cfg.batch_size,
        start_step = global_step
    )
    loss_am = AverageMeter()
    amp = torch.cuda.amp.grad_scaler.GradScaler(growth_interval=100)

    for epoch in range(start_epoch, cfg.num_epoch):
        if isinstance(train_loader, DataLoader):
            train_loader.sampler.set_epoch(epoch)
        for _, (img, local_labels, _) in enumerate(train_loader):
            global_step += 1
            local_embeddings = backbone(img)
            loss: torch.Tensor = module_partial_fc(local_embeddings, local_labels)
            if cfg.fp16:
                amp.scale(loss).backward()
                if global_step % cfg.gradient_acc == 0:
                    amp.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                    amp.step(opt)
                    amp.update()
                    opt.zero_grad()
            else:
                loss.backward()
                if global_step % cfg.gradient_acc == 0:
                    torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                    opt.step()
                    opt.zero_grad()
            lr_scheduler.step()
            with torch.no_grad():
                loss_am.update(loss.item(), 1)
                callback_logging(global_step, loss_am, epoch, cfg.fp16, lr_scheduler.get_last_lr()[0], amp)
        if cfg.save_all_states:
            checkpoint = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "state_dict_backbone": backbone.module.state_dict(),
                "state_dict_softmax_fc": module_partial_fc.state_dict(),
                "state_optimizer": opt.state_dict(),
                "state_lr_scheduler": lr_scheduler.state_dict()
            }
            torch.save(checkpoint, os.path.join(cfg.output, f"checkpoint_gpu_{rank}.pt"))
        if rank == 0:
            path_module = os.path.join(cfg.output, f"model_epoch-{epoch}.pt")
            torch.save(backbone.state_dict(), path_module)
    if rank == 0:
        path_module = os.path.join(cfg.output, "model_final.pt")
        torch.save(backbone.state_dict(), path_module)

if __name__ == "main":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    main(parser.parse_args())