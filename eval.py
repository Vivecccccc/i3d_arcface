import argparse, warnings
import importlib, torch, os
import os.path as osp
from data_process.distreamify import VideoDataset
from model.i3d_arcface import I3D_ResNet
from torch.utils.data import DataLoader
import json
from itertools import combinations
import torch.nn.functional as F
from typing import Dict, List
from copy import deepcopy

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

def calc_avg_sim(R: Dict[str, Dict[str, List[float]]]):
    c_t, c_f, c_a = 0, 0, 0
    homo, hetero = 0., 0.
    for o_key, o_val in R.items():
        for i_key, i_val in o_val.items():
            if i_key == o_key:
                c_t += len(i_val)
                homo += sum(i_val)
            else:
                c_f += len(i_val)
                hetero += sum(i_val)
            c_a += sum(i_val)
    return 0 if c_t == 0 else homo / c_t, 0 if c_f == 0 else hetero / c_f

def main(args):
    cfg = get_config(args.config)
    device = torch.device('cuda')
    video_dataset = VideoDataset(data_path=cfg.data, save_root=cfg.dataset_output, 
                                 max_frames=64, recorded=cfg.dataset_recorded)
    assert cfg.batch_size % 2 == 0
    test_loader = DataLoader(video_dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    num_image = len(video_dataset)
    backbone = I3D_ResNet(layers=[3, 4, 14, 4], backbone_2d_path='./ckpt/backbone_2d.pth')
    D = torch.load(cfg.pretrain)
    backbone.load_state_dict(D)
    backbone.eval()

    r = {id: [] for id in range(cfg.num_classes)}
    R = {id: deepcopy(r) for id in range(cfg.num_classes)}

    global_step = 0
    emb_lst, lbl_lst, nme_lst = [], [], []
    for _, (img, local_labels, filename) in enumerate(test_loader):
        global_step += 1
        # img = img.to(device)
        embs = backbone(img)
        if cfg.return_embedding:
            emb_lst.extend(embs.tolist())
            lbl_lst.extend(local_labels.tolist())
            nme_lst.extend(list(filename))
            continue
        assert local_labels.shape[0] == cfg.batch_size
        comb = combinations(list(range(cfg.batch_size)), 2)    
        for c in comb:
            lbl1 = int(local_labels[c[0]])
            lbl2 = int(local_labels[c[1]])
            emb1 = torch.unsqueeze(embs[c[0]], 0)
            emb2 = torch.unsqueeze(embs[c[1]], 0)
            sim = float(F.cosine_similarity(emb1, emb2))
            R.get(lbl1).get(lbl2).append(sim)
        homo_sim, hetero_sim = calc_avg_sim(R)
        print(f"by global step: {global_step}\thomo sim: {homo_sim: .3f}\thetero sim: {hetero_sim: .3f}")
        del img
        del embs
    if cfg.return_embedding:
        assert len(nme_lst) == len(lbl_lst) == len(emb_lst)
        J = {}
        for nm, lb, em in zip(nme_lst, lbl_lst, emb_lst):
            if J.get(nm) is None:
                J.update({nm: {"label": lb, "embeddings": []}})
            J[nm]['embeddings'].append(em)
        with open(f'{cfg.output}/emb_D.json', 'w') as f:
            json.dump(J, f)
            print(f"embeddings dumped to {cfg.output}/emb_D.json")
        return
    with open(f'{cfg.output}/test_D.json', 'w') as f:
        json.dump(R, f)
        print(f"result dumped to {cfg.output}/test_D.json")

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="py config file")
    main(parser.parse_args())

