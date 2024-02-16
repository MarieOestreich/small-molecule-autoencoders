import os 
wd = os.getcwd() + '/'

import pickle
import argparse
import sys
sys.path.append(wd)


from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers.csv_logs import CSVLogger


from src.utils import utils
from src.datamodules.SMILES_datamodule import SMILES_datamodule as dm_smiles
from src.datamodules.SELFIES_datamodule import SELFIES_datamodule as dm_selfies
from src.models.Autoencoder import Autoencoder as ae_smiles
from src.models.Autoencoder_selfies import Autoencoder as ae_selfies
from omegaconf import DictConfig


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_file",
        "-ckpt",
        type=str,
        required=True,
        help="path to model checkpoint",
    )
    args = parser.parse_args()
    return args

args = parse_arguments()

name = args.checkpoint_file.split('_')
name = '_'.join(name[0:-2])

log = utils.get_logger(__name__)


# Set seed for random number generators in pytorch, numpy and python.random

seed_everything(args.checkpoint_file.split('_')[-2], workers=True)
print(f"seed is {args.checkpoint_file.split('_')[-2]}")


# get model information from name:
model_params, data_params = utils.params_from_name(name)
# batch size:
batch_size = 12500

# load datamodule
if 'selfies' in args.checkpoint_file:
    # dm = dm_selfies
    if 'sub' in args.checkpoint_file:
        dm = pickle.load(open('/data/selfies_subset_dm.pkl', 'rb'))
    else:
        dm = pickle.load(open('/data/selfies_fullMOSES_dm.pkl', 'rb'))
    ae = ae_selfies
else:
    # dm = dm_smiles
    if 'sub' in args.checkpoint_file:
        dm = pickle.load(open('/data/smiles_subset_dm.pkl', 'rb'))
    else:
        dm = pickle.load(open('/data/smiles_fullMOSES_dm.pkl', 'rb'))
    ae = ae_smiles

if 'selfies' in args.checkpoint_file:
    tokenizer = DictConfig({'vocabulary': dm.alphabet})
else:
    tokenizer = dm.tokenizer

# load model
model = ae(dropout = 0,
                    lr = 0.005,
                    save_latent = False,
                    save_latent_dir = False,
                    log_token_position_acc = False,
                    tokenizer=tokenizer, 
                    max_seq_len=dm.molecular_dataset.max_seq_length, 
                    batch_size=batch_size, 
                    **model_params)


logger = CSVLogger("logs", name=args.checkpoint_file.split('.')[0])
print('using csv logger')

# Init trainer
trainer = Trainer(
        gpus = 1,
        min_epochs = 1,
        max_epochs = 1,
        resume_from_checkpoint = 'Null',
        num_sanity_val_steps = 0,
        log_every_n_steps = 40,
        logger = logger
)


# run test
trainer.test(model=model, datamodule=dm, ckpt_path=(wd + 'model-checkpoints/'+args.checkpoint_file))

