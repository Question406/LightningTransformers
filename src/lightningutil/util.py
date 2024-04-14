import os
import json
import glob
import datetime
import importlib
from omegaconf import OmegaConf
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning import Trainer, Callback, seed_everything
from pytorch_lightning.utilities import rank_zero_only


def count_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def create_log_dir(configs):
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    if configs.name and configs.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    if configs.resume:
        if not os.path.exists(configs.resume):
            raise ValueError("Cannot find {}".format(configs.resume))
        if os.path.isfile(configs.resume):
            paths = configs.resume.split("/")
            idx = len(paths) - paths[::-1].index(configs.base_logdir) + 1
            logdir = "/".join(paths[:idx])
            ckpt = configs.resume
        else:
            assert os.path.isdir(configs.resume), configs.resume
            logdir = configs.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

        configs.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        configs.base = base_configs + configs.base
        _tmp = logdir.split("/")
        nowname = _tmp[-1]
    else:
        if configs.name:
            name = configs.name + "_"
        elif configs.base:
            cfg_name = os.path.split(configs.base[0])[-1]
            cfg_name = os.path.splitext(cfg_name)[0]
            name = cfg_name + "_"
        else:
            name = ""
        nowname = name + now + configs.postfix
        if configs.debug:
            logdir = os.path.join(
                configs.base_logdir, configs.project, "debug", nowname)
        else:
            logdir = os.path.join(configs.base_logdir, configs.project, nowname)

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    seed_everything(configs.seed)

    return now, nowname, logdir, ckptdir, cfgdir

def init_default_callbacks(resume, now, nowname, ckptdir, cfgdir, logdir, config, lightning_config, checkpoint_config):
    # add logger
    trainer_kwargs = {}
    default_logger_cfgs = {
        "wandb": {
           "target": "pytorch_lightning.loggers.WandbLogger",
           "params": {
               "project": config.project,
               "name": nowname,
               "save_dir": logdir,
               "offline": config.debug,
               "id": nowname,
           },
        },
    }
    logger_cfg = lightning_config.logger or OmegaConf.create()
    logger_cfg = OmegaConf.merge(default_logger_cfgs["wandb"], logger_cfg)
    os.makedirs(os.path.join(logdir, "wandb"), exist_ok=True)  # create wandb dir
    trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

    # add callback which sets up log directory
    default_callbacks_cfg = {
        "checkpoint_callback": {
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                **checkpoint_config
            },
        },
        "setup_callback": {
            "target": "src.lightningutil.SetupCallback",
            "params": {
                "resume": resume,
                "now": now,
                "logdir": logdir,
                "ckptdir": ckptdir,
                "cfgdir": cfgdir,
                "config": config,
                "lightning_config": lightning_config,
            },
        },
        "learning_rate_logger": {
            "target": "pytorch_lightning.callbacks.LearningRateMonitor",
            "params": {"logging_interval": "step", "log_momentum": True},
        },
    }

    callbacks_cfg = lightning_config.callbacks or OmegaConf.create()
    callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
    trainer_kwargs["callbacks"] = [
        instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg
    ]
    trainer_kwargs["callbacks"].append(
        RichProgressBar(leave=True)
    )

    return trainer_kwargs