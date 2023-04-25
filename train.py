import argparse, warnings
import importlib, torch, os
import os.path as osp
from datetime import datetime
from data_process.distreamify import VideoDataset
from model.i3d_arcface import I3D_ResNet
from model.losses import CombinedMarginLoss
from model.lr_scheduler import PolynomialLRWarmup
from model.partial_fc_v2 import PartialFC_V2
from torch.utils.data import DataLoader
from torch import distributed
from utils.loggings import AverageMeter, CallBackLogging, init_logging

warnings.filterwarnings('ignore')

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

    # get config
    cfg = get_config(args.config)
    device = torch.device('cuda')
    rank = 0
    local_rank = 0
    world_size = 1
    distributed.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:12584",
        rank=rank,
        world_size=world_size,
    )
    os.makedirs(cfg.output, exist_ok=True)
    init_logging(rank, cfg.output)

    video_dataset = VideoDataset(data_path=cfg.data, save_root=cfg.dataset_output, 
                                 max_frames=cfg.max_frame, recorded=cfg.dataset_recorded)
    train_loader = DataLoader(video_dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    num_image = len(video_dataset)
    num_classes = len(set(video_dataset.labels))

    wandb_logger = None
    if cfg.using_wandb:
        import wandb
        try:
            wandb.login(key=cfg.wandb_key)
        except Exception as e:
            print(f"Config WandB error: {e}")
        run_name = datetime.now().strftime('%y%m%d_%H%M') + f'_GPU{rank}'
        run_name = run_name if cfg.suffix_run_name is None else run_name + f"_{cfg.suffix_run_name}"
        try:
            wandb_logger = wandb.init(
                entity = cfg.wandb_entity, 
                project = cfg.wandb_project, 
                sync_tensorboard = True,
                resume=cfg.wandb_resume,
                name = run_name) if rank == 0 or cfg.wandb_log_all else None
            if wandb_logger:
                wandb_logger.config.update(cfg)
        except Exception as e:
            print(f"Config WandB error: {e}")

    backbone = I3D_ResNet(layers=[3, 4, 14, 3], backbone_2d_path='./ckpt/backbone_2d.pth').cuda()
    if cfg.pretrain:
        D = torch.load(cfg.pretrain)
        backbone.load_state_dict(D)
    backbone.train()

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
    if cfg.resume:
        pass
        # dict_checkpoint = torch.load(os.path.join(cfg.output, f"checkpoint_gpu_{rank}.pt"))
        # start_epoch = dict_checkpoint["epoch"]
        # global_step = dict_checkpoint["global_step"]
        # backbone.module.load_state_dict(dict_checkpoint["state_dict_backbone"])
        # module_partial_fc.load_state_dict(dict_checkpoint["state_dict_softmax_fc"])
        # opt.load_state_dict(dict_checkpoint["state_optimizer"])
        # lr_scheduler.load_state_dict(dict_checkpoint["state_lr_scheduler"])
        # del dict_checkpoint
    callback_logging = CallBackLogging(
        frequent=cfg.frequent,
        total_step=cfg.total_step,
        batch_size=cfg.batch_size,
        start_step = global_step
    )
    loss_am = AverageMeter()
    amp = torch.cuda.amp.grad_scaler.GradScaler(growth_interval=100)

    for epoch in range(start_epoch, cfg.num_epoch):
        for _, (img, local_labels) in enumerate(train_loader):
            img = img.to(device)
            local_labels = local_labels.to(device)
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
                if wandb_logger:
                    wandb_logger.log({
                        'Loss/Step Loss': loss.item(),
                        'Loss/Train Loss': loss_am.avg,
                        'Process/Step': global_step,
                        'Process/Epoch': epoch
                    })
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
            if wandb_logger and cfg.save_artifacts:
                artifact_name = f"{run_name}_E{epoch}"
                model = wandb.Artifact(artifact_name, type='model')
                model.add_file(path_module)
                wandb_logger.log_artifact(model)

    if rank == 0:
        path_module = os.path.join(cfg.output, "model_final.pt")
        torch.save(backbone.state_dict(), path_module)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(
        description="Distributed Arcface Training in Pytorch")
    parser.add_argument("config", type=str, help="py config file")
    main(parser.parse_args())
