import logging
import warnings
from typing import List, Sequence

import pytorch_lightning as pl
import rich.syntax
import rich.tree
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only


def get_logger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    ):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


def extras(config: DictConfig) -> None:
    """A couple of optional utilities, controlled by main config file:
    - disabling warnings
    - forcing debug friendly configuration
    - verifying experiment name is set when running in experiment mode

    Modifies DictConfig in place.

    Args:
        config (DictConfig): Configuration composed by Hydra.
    """

    log = get_logger(__name__)

    # disable python warnings if <config.ignore_warnings=True>
    if config.get("ignore_warnings"):
        log.info("Disabling python warnings! <config.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # verify experiment name is set when running in experiment mode
    if config.get("experiment_mode") and not config.get("name"):
        log.info(
            "Running in experiment mode without the experiment name specified! "
            "Use `python run.py mode=exp name=experiment_name`"
        )
        log.info("Exiting...")
        exit()

    # force debugger friendly configuration if <config.trainer.fast_dev_run=True>
    # debuggers don't like GPUs and multiprocessing
    if config.trainer.get("fast_dev_run"):
        log.info("Forcing debugger friendly configuration! <config.trainer.fast_dev_run=True>")
        if config.trainer.get("gpus"):
            config.trainer.gpus = 0
        if config.datamodule.get("pin_memory"):
            config.datamodule.pin_memory = False
        if config.datamodule.get("num_workers"):
            config.datamodule.num_workers = 0


@rank_zero_only
def print_config(
    config: DictConfig,
    fields: Sequence[str] = (
        "trainer",
        "model",
        "datamodule",
        "callbacks",
        "logger",
        "test_after_training",
        "seed",
        "name",
    ),
    resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        config (DictConfig): Configuration composed by Hydra.
        fields (Sequence[str], optional): Determines which main fields from config will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)

    with open("config_tree.log", "w") as fp:
        rich.print(tree, file=fp)


@rank_zero_only
def log_hyperparameters(
    config: DictConfig,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: List[pl.Callback],
    logger: List[pl.loggers.LightningLoggerBase],
) -> None:
    """This method controls which parameters from Hydra config are saved by Lightning loggers.

    Additionaly saves:
        - number of model parameters
    """

    hparams = {}

    # choose which parts of hydra config will be saved to loggers
    hparams["trainer"] = config["trainer"]
    hparams["model"] = config["model"]
    hparams["datamodule"] = config["datamodule"]

    if "seed" in config:
        hparams["seed"] = config["seed"]
    if "callbacks" in config:
        hparams["callbacks"] = config["callbacks"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)


def finish(
    config: DictConfig,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: List[pl.Callback],
    logger: List[pl.loggers.LightningLoggerBase],
) -> None:
    """Makes sure everything closed properly."""

    # without this sweeps with wandb logger might crash!
    for lg in logger:
        if isinstance(lg, pl.loggers.wandb.WandbLogger):
            import wandb

            wandb.finish()


def params_from_name(name):
    """retrieves model parameters from experiment name
    
    Args:
        -name(string): The experiment name. 
            For SMILES2SMILES models, the required format is: S2S_RNNType_Attention_Enumeration_LatentSize_HiddenSize_NumLayers_Dataset_bidirectional.

    Additional Info:
        -RNNType: gru or lstm
        -Attention: a0 (no), a1 (yes)
        -Enumeration: 
            -no: e-0
            -yes, e.g., 6-fold: e-6-c2e (can2enum), e-6-e2c (enum2can), e-6-e2e (enum2enum)
        -LatentSize: e.g., ls32, ls64, ls128, ...
        -HiddenSize: e.g., h32, h64, h128
        -Numlayers: e.g., l1, l2, l3, ...
        -Dataset: either sub1 (50k/12.5k), sub2 (200k/12.5k), sub3 (500k/12.5k), sub4 (1M/12.5k), full (full MOSES set)
        -bidirectional:
            -encoder yes, decoder no: b1-0
            -encoder no, decoder yes: b0-1
            -encoder yes, decoder yes: b1-1
            -encoder no, decoder no: b0-0
        
    """
    name = name.split("_")
    model_param_dict = {}
    data_param_dict = {}
    # If it is a SMILES-to-SMILES model:
    if name[0] in ['S2S', 'G2S', 'G2SM']:
        model_param_dict['rnn_type']=name[1].upper()
        
        model_param_dict['hidden_size']= int(name[5].split('h')[1].split('-')[0])
        model_param_dict['num_layers']=int(name[6].split('l')[1].split('-')[0])
        model_param_dict['bidirectional_en']=bool(int(name[8].split('b')[1].split('-')[0]))
        model_param_dict['bidirectional_de']=bool(int(name[8].split('b')[1].split('-')[1]))
        data_param_dict['filename']=name[7]+'.csv'
        model_param_dict['latent_size']=int(name[4].split('ls')[1])
        if model_param_dict['latent_size'] != model_param_dict['hidden_size']:
            model_param_dict['extra_latent']=True
        else:
            model_param_dict['extra_latent']=False
        
        # attention:
        if name[2].split('a')[1]=='0':
            model_param_dict['attention']=False
        elif name[2].split('a')[1]=='1':
            model_param_dict['attention']=True
        else:
            raise Exception("attention must be given as 'a0' (no attention) or 'a1' (with attention).")
        # enumeration:
        if name[3] == 'e-0':
            model_param_dict['enumerated']=False
        else:
            model_param_dict['enumerated']=True
    else:
        raise NotImplementedError
    
    return model_param_dict, data_param_dict

