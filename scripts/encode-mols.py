import argparse
from datetime import datetime
import os 
import sys
import pickle
import torch
import pandas as pd
import selfies as sf

wd = os.getcwd() + '/'
sys.path.append(wd)

from src.models.Autoencoder import Autoencoder as ae_smiles
from src.models.Autoencoder_selfies import Autoencoder as ae_selfies
from src.utils import utils
from omegaconf import DictConfig

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_file",
        "-ckptf",
        type=str,
        required=True,
        help="name of the checkpoint to be used",
    )
    parser.add_argument(
        "--mols",
        "-m",
        type=str,
        required=True,
        help="csv with SMILES/SELFIES to encode. Should contain one string per row, no headers, no indeces.",
    )
    args = parser.parse_args()
    return args

args = parse_arguments()


def smiles2onehot(smiles, tokenizer, max_seq):
    tokenized_smiles = tokenizer.encode(smiles)
    smiles_tensor = torch.zeros((len(tokenized_smiles), max_seq, len(tokenizer.vocabulary)))
    for i in range(len(tokenized_smiles)):
            for j in range(max_seq):
                if (j < tokenized_smiles[i].shape[0]):
                    smiles_tensor[i][j] = tokenized_smiles[i][j]
                else:
                    ts = torch.zeros(len(tokenizer.vocabulary))

                    ts[tokenizer.vocabulary['*']] = 1

                    smiles_tensor[i][j] = ts
    return smiles_tensor[:, 0:-1, :]

def selfies2onehot(selfies, tokenizer, max_seq):
    selfies_tensor = []
    for selfie in selfies:
        symbol_to_idx = {s: i for i, s in enumerate(tokenizer)}
        one_hot_start_token = torch.zeros((1, len(symbol_to_idx)))
        one_hot_start_token[0][symbol_to_idx['^']] = 1
        one_hot = sf.selfies_to_encoding(
            selfies=selfie,
            vocab_stoi=symbol_to_idx,
            pad_to_len=max_seq,
            enc_type="one_hot"
            )
        one_hot = torch.Tensor(one_hot)
        one_hot = torch.cat((one_hot_start_token, one_hot)).unsqueeze(0)
        selfies_tensor.append(one_hot[:, 0:-1, :])

    selfies_tensor = torch.vstack(selfies_tensor)
    return selfies_tensor



if 'selfies' in args.checkpoint_file:
    ae = ae_selfies
else:
    ae = ae_smiles

# load input molecules:
molecules = list(pd.read_csv(args.mols, header=None).iloc[:,0].values)


# get model information from name:
name = args.checkpoint_file.split('_')
name = '_'.join(name[0:-2])
model_params, data_params = utils.params_from_name(name)
if 'selfies' in args.checkpoint_file:
    if 'sub' in args.checkpoint_file:
        tokenizer = DictConfig({'vocabulary': list(pd.read_csv('data/vocab-selfies-subset.csv',
                                                    header=None).iloc[:,0].values)})
        # the longest string that was in the training set 
        max_seq = 49
    else:
        tokenizer = DictConfig({'vocabulary': list(pd.read_csv('data/vocab-selfies-fullset.csv',
                                                    header=None).iloc[:,0].values)})
        # the longest string that was in the training set 
        max_seq = 55
else:
    if 'sub' in args.checkpoint_file:
        tokenizer = pickle.load(open('data/tokenizer-smiles-subset.pkl', 'rb'))
        # the longest string that was in the training set 
        max_seq = 55
    else:
        tokenizer = pickle.load(open('data/tokenizer-smiles-fullset.pkl', 'rb'))
        # the longest string that was in the training set 
        max_seq = 59



# filter out all molecules that have unknown tokens:
molecules_filt = []
if 'selfies' in args.checkpoint_file:
    for m in molecules:
        tks = list(sf.split_selfies(m))
        if set(tks).issubset(set(tokenizer.vocabulary)) and (len(tks)<=(max_seq-2)):
            molecules_filt.append(m)

else:

    for m in molecules:
        ids = [int(t[1]) for t in tokenizer.convert_tokens_to_ids(tokenizer.tokenize(m))]
        if (not tokenizer._vocabulary[tokenizer._unknown_token] in ids) and (len(m)<=(max_seq-2)):
            molecules_filt.append(m)

print(f'{len(molecules)-len(molecules_filt)} molecules have been removed because they were either too long or contained unknown tokens.')

batch_size = len(molecules_filt)
if batch_size > 0:

    # load model
    model = ae(dropout = 0,
                        lr = 0.005,
                        save_latent = False,
                        save_latent_dir = False,
                        log_token_position_acc = False,
                        tokenizer=tokenizer, 
                        max_seq_len=max_seq, 
                        batch_size=batch_size, 
                        **model_params)
    checkpoint = torch.load(wd + 'model-checkpoints/' + args.checkpoint_file)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    print('======= model loaded =======')

    if 'selfies' in args.checkpoint_file:
        batch = selfies2onehot(molecules_filt, tokenizer.vocabulary, max_seq)
    else:
        batch = smiles2onehot(molecules_filt, tokenizer, max_seq)

    latents = model.encoder(batch)
    

    date_time = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    os.mkdir(wd+'/results_'+date_time)
    torch.save(latents, wd+'/results_'+date_time+'/latents.pt')
    pd.DataFrame(molecules_filt).to_csv(wd+'/results_'+date_time+'/molecules.csv', header=False, index=False)
    print('======= done =======')


else:
    print('After filtering, no molecules were left to encode.')