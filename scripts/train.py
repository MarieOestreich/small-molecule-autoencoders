import os 
wd = os.getcwd() + '/'


import argparse
import sys
sys.path.append(wd)


from pytorch_lightning import Trainer, callbacks
from pytorch_lightning import seed_everything

from src.datamodules.SMILES_datamodule import SMILES_datamodule as dm_smiles
from src.datamodules.SELFIES_datamodule import SELFIES_datamodule as dm_selfies
from src.models.Autoencoder import Autoencoder as ae_smiles
from src.models.Autoencoder_selfies import Autoencoder as ae_selfies
import time
from src.utils import utils
from omegaconf import DictConfig


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name",
        "-n",
        type=str,
        required=True,
        help="model name (see naming scheme in readme)",
    )
    parser.add_argument(
        "--batch_size",
        "-bs",
        type=int,
        required=True,
        help="batch size to use during training",
    )
    parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        required=False,
        help="number of epochs to train for",
        default=50
    )
    
    parser.add_argument(
        "--gpus",
        "-g",
        type=int,
        required=False,
        help="number of GPUs to use for training",
        default=0
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        required=False,
        help="seed to use for training",
        default=12345
    )
    args = parser.parse_args()
    return args

args = parse_arguments()

log = utils.get_logger(__name__)



# Set seed for random number generators in pytorch, numpy and python.random

seed_everything(args.seed, workers=True)
print(f"seed is {args.seed}")


# get model information from name:
model_params, data_params = utils.params_from_name(args.name)
# batch size:
batch_size = args.batch_size

# load datamodule
if 'selfies' in args.name:
    dm = dm_selfies
    ae = ae_selfies
else:
    dm = dm_smiles
    ae = ae_smiles

datamodule = dm(data_dir = wd + 'data',
                                num_workers=16, 
                                pin_memory=False, 
                                enumerated=model_params['enumerated'], 
                                batch_size = batch_size, 
                                **data_params)

if 'selfies' in args.name:
    tokenizer = DictConfig({'vocabulary': datamodule.alphabet})
else:
    tokenizer = datamodule.tokenizer


# load model
model = ae(dropout = 0,
                    lr = 0.005,
                    save_latent = False,
                    save_latent_dir = False,
                    log_token_position_acc = False,
                    tokenizer=tokenizer, 
                    max_seq_len=datamodule.molecular_dataset.max_seq_length, 
                    batch_size=batch_size, 
                    **model_params)

checkpoint_callback = callbacks.ModelCheckpoint(monitor='val_loss', save_top_k=1)


# Init trainer
trainer = Trainer(
        gpus = args.gpus,
        min_epochs = args.epochs,
        max_epochs = args.epochs,
        num_sanity_val_steps = 0,
        log_every_n_steps = 200,
        callbacks=checkpoint_callback
)


# train
trainer.fit(model=model, datamodule=datamodule)

# run test
trainer.test(model=model, datamodule=datamodule, ckpt_path='best')

print(f"Best model ckpt at {trainer.checkpoint_callback.best_model_path}")

