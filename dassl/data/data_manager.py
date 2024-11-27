import numpy as np

import torch
import torchvision.transforms as T
from tabulate import tabulate
from torch.utils.data import Dataset as TorchDataset


from .datasets import build_dataset
from .samplers import build_sampler
from .transforms import (
    INTERPOLATION_MODES, build_transform,
    normal_pc, translate_pointcloud, 
    rotation_point_cloud, jitter_point_cloud,
    PointcloudToTensor,  PointcloudScale,
    PointcloudRotate, PointcloudRotatePerturbation,
    PointcloudTranslate, PointcloudJitter,
)


def build_data_loader(
    cfg,
    sampler_type="SequentialSampler",
    data_source=None,
    batch_size=64,
    n_domain=0,
    n_ins=2,
    tfm=None,
    is_train=True,
    dataset_wrapper=None
):
    # Build sampler
    sampler = build_sampler(
        sampler_type,
        cfg=cfg,
        data_source=data_source,
        batch_size=batch_size,
        n_domain=n_domain,
        n_ins=n_ins
    )

    if dataset_wrapper is None:
        dataset_wrapper = DatasetWrapper

    # Build data loader
    data_loader = torch.utils.data.DataLoader(
        dataset_wrapper(cfg, data_source, transform=tfm, is_train=is_train),
        batch_size=batch_size,
        sampler=sampler,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        drop_last=is_train and len(data_source) >= batch_size,
        pin_memory=(torch.cuda.is_available() and cfg.USE_CUDA)
    )
    assert len(data_loader) > 0

    return data_loader


class DataManager:

    def __init__(
        self,
        cfg,
        custom_tfm_train=None,
        custom_tfm_test=None,
        dataset_wrapper=None
    ):
        # Load dataset
        dataset = build_dataset(cfg)

        # Build transform
        if custom_tfm_train is None:
            tfm_train = build_transform(cfg, is_train=True)
        else:
            print("* Using custom transform for training")
            tfm_train = custom_tfm_train

        if custom_tfm_test is None:
            tfm_test = build_transform(cfg, is_train=False)
        else:
            print("* Using custom transform for testing")
            tfm_test = custom_tfm_test

        # Build train_loader_x
        train_loader_x = build_data_loader(
            cfg,
            sampler_type=cfg.DATALOADER.TRAIN_X.SAMPLER,
            data_source=dataset.train_x,
            batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
            n_domain=cfg.DATALOADER.TRAIN_X.N_DOMAIN,
            n_ins=cfg.DATALOADER.TRAIN_X.N_INS,
            tfm=tfm_train,
            is_train=True,
            dataset_wrapper=dataset_wrapper
        )

        # Build train_loader_u
        train_loader_u = None
        if dataset.train_u:
            sampler_type_ = cfg.DATALOADER.TRAIN_U.SAMPLER
            batch_size_ = cfg.DATALOADER.TRAIN_U.BATCH_SIZE
            n_domain_ = cfg.DATALOADER.TRAIN_U.N_DOMAIN
            n_ins_ = cfg.DATALOADER.TRAIN_U.N_INS

            if cfg.DATALOADER.TRAIN_U.SAME_AS_X:
                sampler_type_ = cfg.DATALOADER.TRAIN_X.SAMPLER
                batch_size_ = cfg.DATALOADER.TRAIN_X.BATCH_SIZE
                n_domain_ = cfg.DATALOADER.TRAIN_X.N_DOMAIN
                n_ins_ = cfg.DATALOADER.TRAIN_X.N_INS

            train_loader_u = build_data_loader(
                cfg,
                sampler_type=sampler_type_,
                data_source=dataset.train_u,
                batch_size=batch_size_,
                n_domain=n_domain_,
                n_ins=n_ins_,
                tfm=tfm_train,
                is_train=True,
                dataset_wrapper=dataset_wrapper
            )

        # Build val_loader
        val_loader = None
        if dataset.val:
            val_loader = build_data_loader(
                cfg,
                sampler_type=cfg.DATALOADER.TEST.SAMPLER,
                data_source=dataset.val,
                batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
                tfm=tfm_test,
                is_train=False,
                dataset_wrapper=dataset_wrapper
            )

        # Build test_loader
        test_loader = build_data_loader(
            cfg,
            sampler_type=cfg.DATALOADER.TEST.SAMPLER,
            data_source=dataset.test,
            batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
            tfm=tfm_test,
            is_train=False,
            dataset_wrapper=dataset_wrapper
        )

        # Attributes
        self._num_classes = dataset.num_classes
        self._num_source_domains = len(cfg.DATASET.SOURCE_DOMAINS)
        self._lab2cname = dataset.lab2cname

        # Dataset and data-loaders
        self.dataset = dataset
        self.train_loader_x = train_loader_x
        self.train_loader_u = train_loader_u
        self.val_loader = val_loader
        self.test_loader = test_loader

        if cfg.VERBOSE:
            self.show_dataset_summary(cfg)

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def num_source_domains(self):
        return self._num_source_domains

    @property
    def lab2cname(self):
        return self._lab2cname

    def show_dataset_summary(self, cfg):
        dataset_name = cfg.DATASET.NAME
        source_domains = cfg.DATASET.SOURCE_DOMAINS
        target_domains = cfg.DATASET.TARGET_DOMAINS

        table = []
        table.append(["Dataset", dataset_name])
        if source_domains:
            table.append(["Source", source_domains])
        if target_domains:
            table.append(["Target", target_domains])
        table.append(["# classes", f"{self.num_classes:,}"])
        table.append(["# train_x", f"{len(self.dataset.train_x):,}"])
        if self.dataset.train_u:
            table.append(["# train_u", f"{len(self.dataset.train_u):,}"])
        if self.dataset.val:
            table.append(["# val", f"{len(self.dataset.val):,}"])
        table.append(["# test", f"{len(self.dataset.test):,}"])

        print(tabulate(table))


class DatasetWrapper(TorchDataset):

    def __init__(self, cfg, data_source, transform=None, is_train=False):
        self.cfg = cfg
        self.data_source = data_source
        # self.transform = transform  # accept list (tuple) as input
        self.is_train = is_train
        self.is_pointda = cfg.DATASET.TYPE == "pointda"
        self.is_sim2real = cfg.DATASET.TYPE == "sim2real"
        self.is_sonn = cfg.DATASET.NAME == "ScanObjectNN"
        self.is_omni3d = cfg.DATASET.NAME == "OmniObject3D"
        self.is_o_lvis = cfg.DATASET.NAME == "Objaverse_LVIS"
        self.swapaxis = cfg.DATASET.NAME == "ModelNet_11" or cfg.DATASET.NAME == "Objaverse_LVIS"

        if self.is_sim2real:
            self.transforms = T.Compose([
                PointcloudToTensor(),
                PointcloudScale(),
                PointcloudRotate(),
                PointcloudRotatePerturbation(),
                PointcloudTranslate(),
                PointcloudJitter(),
            ])

        if self.is_sonn:
            print(f'>>>>>>> self.is_sonn: {self.is_sonn}')

        if self.is_omni3d:
            print(f'>>>>>>> self.is_omni3d: {self.is_omni3d}')

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        item = self.data_source[idx]

        output = {
            "label": item.label,
            "domain": item.domain,
            "impath": item.impath,  # assign values in specific `Dataset` definition 
            "index": idx
        }

        # NOTE not sure 1024 is enough for Objaverse_lvis, its default value is 10000
        num_points = self.cfg.PointEncoder.num_points
        pointcloud = item.impath[: num_points]

        # NOTE added by jerry
        if self.swapaxis:
            pointcloud[:, 1] = pointcloud[:, 2] + pointcloud[:, 1]
            pointcloud[:, 2] = pointcloud[:, 1] - pointcloud[:, 2]
            pointcloud[:, 1] = pointcloud[:, 1] - pointcloud[:, 2]

        if self.is_pointda or self.is_sim2real or self.is_sonn or self.is_omni3d or self.is_o_lvis:
            # normalize pointcloud both during training and test in this case
            pointcloud = normal_pc(pointcloud)

        if self.is_train:
            # NOTE data augmentations for pointda benchmark, added by jerry
            if self.is_pointda:
                pointcloud = rotation_point_cloud(pointcloud)
                pointcloud = jitter_point_cloud(pointcloud)

            # NOTE data augmentations for sim2real benchmark, added by jerry
            elif self.is_sim2real:
                pointcloud = self.transforms(pointcloud).numpy()

            # NOTE apply translation only by default
            else:
                pointcloud = translate_pointcloud(pointcloud)
                np.random.shuffle(pointcloud)

        output['pc'] = pointcloud.astype("float32")

        return output
