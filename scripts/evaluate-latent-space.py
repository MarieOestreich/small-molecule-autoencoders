import os 
wd = os.getcwd() + '/'


import argparse
import sys
sys.path.append(wd)


from src.utils import utils
from src.utils.latent_space_analysis import latentEval
# datamodules
from src.datamodules.SMILES_datamodule import SMILES_datamodule as dm_smiles
from src.datamodules.SELFIES_datamodule import SELFIES_datamodule as dm_selfies
# autoencoder models
from src.models.Autoencoder import Autoencoder as ae_smiles
from src.models.Autoencoder_selfies import Autoencoder as ae_selfies
# variational autoencoder models
from src.models.VAE import Autoencoder as vae_smiles
from src.models.VAE_selfies import Autoencoder as vae_selfies

from omegaconf import DictConfig
import torch
import selfies as sf
import pandas as pd


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


class Encoder():
    def __init__(self):
        '''load encoder model'''
        pass

    def encode(self, smiles):
        '''given a list of SMILES strings, embed them with the encoder

        params:
            smiles: list of SMILES strings of the molecules to embed

        returns:
            the embedding z as a torch tensor with dimensions (len(smiles), |z|)'''
        pass


if 'selfies' in args.checkpoint_file:
    input = 'selfies'
    dm = dm_selfies
    if 'vae' in args.checkpoint_file:
        ae = vae_selfies
    else:
        ae = ae_selfies
else:
    input = 'smiles'
    dm = dm_smiles
    if 'vae' in args.checkpoint_file:
        ae = vae_smiles
    else:
        ae = ae_smiles

class Enc(Encoder):
    def __init__(self, checkpoint_file):
        self.name = checkpoint_file.split('_')
        self.name = '_'.join(self.name[0:-2])
        model_params, data_params = utils.params_from_name(self.name)

        self.ds = dm(data_dir = wd + 'data',
                                num_workers=16, 
                                pin_memory=False, 
                                enumerated=model_params['enumerated'], 
                                batch_size = 1, 
                                **data_params).molecular_dataset
        if 'selfies' in self.name:

            tokenizer = DictConfig({'vocabulary': self.ds.alphabet})
        else:
            tokenizer = self.ds.tokenizer

        
        self.ae = ae(dropout = 0,
                        lr = 0.005,
                        save_latent = False,
                        save_latent_dir = False,
                        log_token_position_acc = False,
                        tokenizer=tokenizer, 
                        max_seq_len=self.ds.max_seq_length, 
                        batch_size=1, 
                        **model_params)
        
        if torch.cuda.is_available():
            ckpt = torch.load((wd + 'model-checkpoints/' + checkpoint_file))
        else:
            ckpt = torch.load((wd + 'model-checkpoints/' + checkpoint_file),
                    map_location=torch.device('cpu'))

        self.ae.load_state_dict(ckpt['state_dict'])
        self.ae.eval()
        for name, param in self.ae.named_parameters():
            param.requires_grad = False

    def encode(self, smiles):
        if 'selfies' in self.name:
            selfies = [self.ds.encode_selfie(sf.encoder(s))[0] for s in smiles]
            batch = torch.stack(selfies)
        else:
            batch = self.ds.smiles2onehot(smiles)

        embs = self.ae.encoder(batch).squeeze(0)
        embs = embs.detach()
        return embs

enc = Enc(args.checkpoint_file)
if 'fullMOSES' in enc.name:
    dataSet = pd.read_csv(wd + 'data/fullMOSES.csv')
    dataSet.columns = [dataSet.columns[0], 'mode']
elif 'can2enum' in enc.name:
    dataSet = pd.read_csv(wd + 'data/sub1-5x-can2enum.csv')
elif 'enum2can' in enc.name:
    dataSet = pd.read_csv(wd + 'data/sub1-5x-enum2can.csv')
else:
    dataSet = pd.read_csv(wd + 'data/sub1.csv')
    dataSet.columns = [dataSet.columns[0], 'mode']


trainSet = list(dataSet[dataSet['mode']=='train'].iloc[:, 0].values)
testSet = list(dataSet[dataSet['mode']=='test'].iloc[:, 0].values)


le = latentEval(enc, input, trainSet, testSet)
eval_out = le.evaluate(3)

if 'vae' in args.checkpoint_file:
    eval_out[0].savefig(wd + enc.name + '_vae_dists.pdf', bbox_inches='tight', dpi =300)
    eval_out[1].savefig(wd + enc.name + '_vae_pca.pdf', bbox_inches='tight', dpi =300)
else:
    eval_out[0].savefig(wd + enc.name + '_dists.pdf', bbox_inches='tight', dpi =300)
    eval_out[1].savefig(wd + enc.name + '_pca.pdf', bbox_inches='tight', dpi =300)