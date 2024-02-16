from torch.utils.data import DataLoader
import pandas as pd
import selfies as sf
from src.utils.tokenize import SMILESTokenizer
from torch.utils.data import Dataset
import pathlib
import torch
import numpy as np

class MolecularDataset(Dataset):
    def __init__(self, train=True, tokenizer=None, data_dir=None, filename = 'moses_subset.csv', enumerated = False):

        mode = 'test'
        if train == True:
            mode = 'train'

        self.data_dir = data_dir
        self.filename = filename
        self.enumerated = enumerated
        self.datafile = pathlib.Path(self.data_dir).joinpath(self.filename)
        self.smiles = pd.read_csv(self.datafile, header=None).values[:, 0]
        
        if (self.enumerated):
            self.sets = pd.read_csv(self.datafile, header=None).values[:, 2]
            self.can_smiles = pd.read_csv(self.datafile, header=None).values[:, 1]

        else:
            self.sets = pd.read_csv(self.datafile, header=0).values[:, 1]

        self.smiles = [self.smiles[i] for i in range(len(self.smiles) - 1) if self.sets[i] == mode]

        if(self.enumerated):
            self.can_smiles = [self.can_smiles[i] for i in range(len(self.can_smiles) - 1) if self.sets[i] == mode]

        if (tokenizer == None):
            self.tokenizer = SMILESTokenizer(smiles=self.smiles, encoding_type='one hot', padding_token='*')
        else:
            self.tokenizer = tokenizer

        # convert list of smiles into list of one hot encoded tensors list: [seq_length, vocab_length]
        self.tokenized_smiles = self.tokenizer.encode(self.smiles)

        # get max seq-length for padding
        self.max_seq_length = max([self.tokenized_smiles[i].shape[0] for i in range(len(self.tokenized_smiles))])

        if(self.enumerated):
            self.tokenized_can_smiles = self.tokenizer.encode(self.can_smiles)
            smiles_can_tensor = torch.zeros(
                (len(self.tokenized_can_smiles), self.max_seq_length, len(self.tokenizer.vocabulary)))


        # create tensor with shape: (smiles_num, max_seq_length, vocab_len)
        smiles_tensor = torch.zeros((len(self.tokenized_smiles), self.max_seq_length, len(self.tokenizer.vocabulary)))

        if (self.enumerated):
            min_index = min(len(self.tokenized_smiles), len(self.tokenized_can_smiles))
        else:
            min_index = len(self.tokenized_smiles)

            # copy tensors from self.tokenized_smiles and pad rest with one hot encoded padding tokens
        for i in range(min_index):
            for j in range(self.max_seq_length):
                if (j < self.tokenized_smiles[i].shape[0]):
                    smiles_tensor[i][j] = self.tokenized_smiles[i][j]
                else:
                    ts = torch.zeros(len(self.tokenizer.vocabulary))

                    ts[self.tokenizer.vocabulary['*']] = 1

                    smiles_tensor[i][j] = ts
                if(self.enumerated):
                    if (j < self.tokenized_can_smiles[i].shape[0]):
                        smiles_can_tensor[i][j] = self.tokenized_can_smiles[i][j]
                    else:
                        ts = torch.zeros(len(self.tokenizer.vocabulary))

                        ts[self.tokenizer.vocabulary['*']] = 1

                        smiles_can_tensor[i][j] = ts


        # tensors in self.tokenized_smiles have now equal shape: [max_seq_length, vocab_length]
        # self.groundtruth is offsetted by start token

        self.smiles = smiles_tensor[:, 0:-1, :]
        self.groundtruth = smiles_tensor[:, 1:, :]

        if (self.enumerated):
            self.groundtruth = smiles_can_tensor[:, 1:, :]
            self.y_can = smiles_can_tensor[:, 0:-1, :]
        else:
            self.groundtruth = smiles_tensor[:, 1:, :]


    def __len__(self):
        return len(self.tokenized_smiles)

    def __getitem__(self, idx):
        if (self.enumerated):
            return self.smiles[idx], self.groundtruth[idx], self.y_can[idx]
        else:
            return self.smiles[idx], self.groundtruth[idx]

    def smiles2onehot(self, smiles):
        ''' smiles: a list '''
        tokenized_smiles = self.tokenizer.encode(smiles)
        smiles_tensor = torch.zeros((len(tokenized_smiles), self.max_seq_length, len(self.tokenizer.vocabulary)))
        min_index = len(tokenized_smiles)
        for i in range(min_index):
            for j in range(self.max_seq_length):
                if (j < tokenized_smiles[i].shape[0]):
                    smiles_tensor[i][j] = tokenized_smiles[i][j]
                else:
                    ts = torch.zeros(len(self.tokenizer.vocabulary))

                    ts[self.tokenizer.vocabulary['*']] = 1

                    smiles_tensor[i][j] = ts
        smiles = smiles_tensor[:, 0:-1, :]
        return smiles





class SELFIESDataset(Dataset):
    def __init__(self, train=True, alphabet = None, data_dir=None, filename = 'moses_subset.csv', enumerated = False):

        mode = 'test'
        if train == True:
            mode = 'train'

        self.data_dir = data_dir
        self.filename = filename
        self.enumerated = enumerated
        self.datafile = pathlib.Path(self.data_dir).joinpath(self.filename)
        self.data = pd.read_csv(self.datafile, header=0)
        print(f"--- Data loaded. shape is: {self.data.shape}") 
        self.selfies = self.data.values[:, 0]
        print(self.datafile)

        print(f' ===== ENUMERATED IN DS: {self.enumerated}')
        
        if alphabet == None:
            print("--- Creating alphabet")
            if(self.enumerated):
                self.alphabet = sf.get_alphabet_from_selfies(self.data.values[:, 0]+self.data.values[:, 1])
            else:
                self.alphabet = sf.get_alphabet_from_selfies(self.selfies)
            self.alphabet.add("[nop]") # padding token
            self.alphabet.add("^") # start token
            self.alphabet = list(sorted(self.alphabet))  
        else:
            self.alphabet = alphabet
        
        if (self.enumerated):
            self.sets = self.data.values[:, 2]
            self.can_selfies = self.data.values[:, 1]

        else:
            self.sets = self.data.values[:, 1]

        self.selfies = [self.selfies[i] for i in range(len(self.selfies) - 1) if self.sets[i] == mode]

        if(self.enumerated):
            self.can_selfies = [self.can_selfies[i] for i in range(len(self.can_selfies) - 1) if self.sets[i] == mode]


        
        if self.enumerated:
            self.max_seq_length = max(sf.len_selfies(s) for s in self.selfies+self.can_selfies) 
        else:
            self.max_seq_length = max(sf.len_selfies(s) for s in self.selfies) 

        print("--- Datset fully instantieted!")


    def __len__(self):
        return len(self.selfies)

    def __getitem__(self, idx):
        if (self.enumerated):
            x, y = self.encode_selfie(self.selfies[idx])
            y_can, _ = self.encode_selfie(self.can_selfies[idx])
            return x, y, y_can
        else:
            return self.encode_selfie(self.selfies[idx])
#             return self.selfies[idx], self.groundtruth[idx]
        
    def encode_selfie(self, selfie):
        symbol_to_idx = {s: i for i, s in enumerate(self.alphabet)}
        one_hot_start_token = torch.zeros((1, len(symbol_to_idx)))
        one_hot_start_token[0][symbol_to_idx['^']] = 1
        one_hot = sf.selfies_to_encoding(
            selfies=selfie,
            vocab_stoi=symbol_to_idx,
            pad_to_len=self.max_seq_length,
            enc_type="one_hot"
            )
        one_hot = torch.Tensor(one_hot)
        one_hot = torch.cat((one_hot_start_token, one_hot))#.unsqueeze(0)
        return one_hot[0:-1, :], one_hot[1:, :]

    def selfies2onehot(self, selfies):
        ''' selfies: a list '''
        symbol_to_idx = {s: i for i, s in enumerate(self.alphabet)}
        tokenized_selfies = []
        one_hot_start_token = torch.zeros((1, len(symbol_to_idx)))
        one_hot_start_token[0][symbol_to_idx['^']] = 1
        for s in selfies:
            one_hot = sf.selfies_to_encoding(
            selfies=s,
            vocab_stoi=symbol_to_idx,
            pad_to_len=self.max_seq_length,
            enc_type="one_hot"
            )
            one_hot = torch.Tensor(one_hot)
            tokenized_selfies.append(torch.cat((one_hot_start_token, one_hot)))
        # one_hot_encodings will be a tensor of shape [number of selfies, max selfie length, vocab length]:
        tokenized_selfies = torch.stack(tokenized_selfies)
        selfies = tokenized_selfies[:, 0:-1, :]
        return selfies
    
    
