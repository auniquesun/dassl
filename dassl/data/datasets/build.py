from dassl.utils import Registry, check_availability

DATASET_REGISTRY = Registry("DATASET")


def build_dataset(cfg):
    avai_datasets = DATASET_REGISTRY.registered_names()
    check_availability(cfg.DATASET.NAME, avai_datasets)
    if cfg.VERBOSE:
        if cfg.DATASET.NAME == 'ScanObjectNN':
            print('Loading dataset: {} {}'.format(cfg.DATASET.NAME, cfg.DATASET.SONN_VARIANT))
        elif cfg.DATASET.NAME == 'ModelNet40_C':
            print('Loading dataset: {} {}'.format(cfg.DATASET.NAME, cfg.DATASET.CORRUPTION_TYPE))
        else:
            print('Loading dataset: {}'.format(cfg.DATASET.NAME))
    return DATASET_REGISTRY.get(cfg.DATASET.NAME)(cfg)
