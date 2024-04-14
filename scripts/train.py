import os
import hydra
import torch

from hydra.core.hydra_config import HydraConfig
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.cli import ArgsType, LightningCLI

from src.lightningutil.modelmodule import DistillModule
from src.lightningutil.datamodule import create_datamod
from src.lightningutil.util import create_log_dir, init_default_callbacks
from src.lightningutil.strategy import MyDeepSpeedStrategy

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.utilities import rank_zero_info
from omegaconf import OmegaConf

from src.utils import init_script, set_progress
from src.language_models import UnlearnLLM, SmallLLM
from src.gen_util import ContrastGenerationMixin
import src.dataset
import src.conv_util
from src.utils import init_script, set_progress

@hydra.main(version_base=None, config_path="../configs", config_name="train_config")
def main(configs):
    # Renew the logdir
    configs.base_logdir = os.path.join(HydraConfig.get().runtime.output_dir, "logs")
    LOGGER = init_script(configs)
    LOGGER.info("Configs", configs=configs)

    now, nowname, logdir, ckptdir, cfgdir = create_log_dir(configs)

    lightning_config = configs.get("lightning", OmegaConf.create())
    trainer_config = lightning_config.get("trainer", OmegaConf.create())    
    checkpoint_config = lightning_config['callbacks']['checkpoint_callback']['params']
    checkpoint_config['dirpath'] = ckptdir
    
    trainer_kwargs = init_default_callbacks(
        configs.resume, now, nowname, ckptdir, cfgdir, logdir, configs, lightning_config, checkpoint_config
    ) 

    if 'deepspeed' in trainer_config.get('strategy', ""):
        OmegaConf.set_struct(trainer_config, False)  # Disable struct mode temporarily
        trainer_config.pop('strategy')
        OmegaConf.set_struct(trainer_config, True)
        trainer = Trainer(**trainer_config, **trainer_kwargs, strategy=MyDeepSpeedStrategy(stage=2))
    else:
        trainer = Trainer(**trainer_config, **trainer_kwargs)

    with trainer.init_module():
        distill_model = DistillModule(**configs.model) #TODO: this does not support resume

    data_module = create_datamod(
        configs.data, 
        expand_data=configs.get('expand_data', False),
        expand_qanum=configs.get('expand_qanum', 5),
    )

    data_module.prepare_data()
    data_module.setup('fit')

    num_update_steps_per_epoch = len(data_module.train_dataloader()) // trainer_config.accumulate_grad_batches
    num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
    num_training_steps = num_update_steps_per_epoch * trainer_config.max_epochs 
    distill_model.on_before_configure_optimizers(
        num_training_steps=num_training_steps,
    )

    rank_zero_info("Start training!!!")
    trainer.fit(distill_model, datamodule=data_module)

if __name__ == "__main__":
    main()
