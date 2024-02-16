import os 
from os import listdir
from os.path import isfile, join
wd = os.getcwd() + '/'

import pickle
import selfies as sf
import sys
sys.path.append(wd)
import torch
import pandas as pd


from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers.csv_logs import CSVLogger


from src.utils import utils
from src.datamodules.SMILES_datamodule import SMILES_datamodule as dm_smiles
from src.datamodules.SELFIES_datamodule import SELFIES_datamodule as dm_selfies
from src.models.Autoencoder import Autoencoder as ae_smiles
from src.models.Autoencoder_selfies import Autoencoder as ae_selfies
from omegaconf import DictConfig


files = [f for f in listdir('/model-checkpoints') if isfile(join('/model-checkpoints', f))]
files = [f for f in files if 'full' in f and not 'selfies' in f and not 'enum' in f and not 'vae' in f]

# load datamodule

# dm = pickle.load(open('/data/selfies_subset_dm.pkl', 'rb'))
# dm = pickle.load(open('/data/selfies_fullMOSES_dm.pkl', 'rb'))
# ae = ae_selfies


# dm = pickle.load(open('/data/smiles_subset_dm.pkl', 'rb'))
dm = pickle.load(open('/data/smiles_fullMOSES_dm.pkl', 'rb'))
ae = ae_smiles
dm.setup()



for checkpoint_file in files:

    if not isfile('/test-recs/' + checkpoint_file.split('.')[0] + '.csv'):
        print(f' ==== currently processing {checkpoint_file}')
        name = checkpoint_file.split('_')
        name = '_'.join(name[0:-2])

        log = utils.get_logger(__name__)

        # Set seed for random number generators in pytorch, numpy and python.random

        seed_everything(checkpoint_file.split('_')[-2], workers=True)
        print(f"seed is {checkpoint_file.split('_')[-2]}")


        # get model information from name:
        model_params, data_params = utils.params_from_name(name)
        # batch size:
        batch_size = dm.smiles_val.__len__()-1



        if 'selfies' in checkpoint_file:
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
        checkpoint = torch.load(wd + 'model-checkpoints/' + checkpoint_file)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()

        print('======= model loaded =======')

        dm.batch_size=dm.smiles_val.__len__()
        X, Y = next(iter(dm.test_dataloader()))
        X = X[1:,:,:]
        Y = Y[1:,:,:]
        print('======= test data loaded =======')
        loss, output_dim, output = model.model_run((X, Y), False)
        print('======= predictions made =======')
        output = torch.transpose(output, 1, 0)
        smiles_tensor = torch.zeros((output.shape[0], output.shape[1], output.shape[2]))
        smiles_tensor = smiles_tensor.type_as(X[0])

        # take token with highest probability as predicted token
        for s in range(output.shape[1]):
            for t in range((output.shape[0])):
                top1 = torch.argmax(output[t][s])
                smiles_tensor[t][s][top1] = 1
        if 'selfies' in checkpoint_file:
            smiles_list = []
            for s in smiles_tensor:
                selfie = sf.encoding_to_selfies(s.tolist(),
                    vocab_itos=model.vocab_itos,
                    enc_type = 'one_hot')
                selfie = selfie.replace('^', '').replace('[nop]', '')
                smiles_list.append(selfie)
            original_smiles_list = []
            for s in X:
                selfie = sf.encoding_to_selfies(s.tolist(),
                    vocab_itos=model.vocab_itos,
                    enc_type = 'one_hot')
                selfie = selfie.replace('^', '').replace('[nop]', '')
                original_smiles_list.append(selfie)
        else:
            smiles_list = model.tokenizer.decode(smiles_tensor)
            original_smiles_list = model.tokenizer.decode(X)

        print('======= molecules translated =======')
        df = pd.DataFrame([original_smiles_list, smiles_list], index = ['original', 'reconstruction']).T
        df.to_csv('/test-recs/' + checkpoint_file.split('.')[0] + '.csv')
    else:
        print(f"{checkpoint_file.split('.')[0]}.csv already exists")