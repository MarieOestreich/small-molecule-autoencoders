from src.models.VAE import Autoencoder as VAE
from src.models.VAE_selfies import Autoencoder as sVAE
from src.models.Autoencoder import Autoencoder as AE
from src.utils import utils
from src.datamodules.MolecularDataset import MolecularDataset
import torch
from src.utils.latent_space_analysis import encoder
from src.models.Autoencoder_selfies import Autoencoder as sAE
import selfies as sf
from src.datamodules.MolecularDataset import SELFIESDataset
from omegaconf import DictConfig


class smilesVaeEnc(encoder):
    
    def __init__(self, model_ckpt):
        super().__init__()
        model_name = [x for x in model_ckpt.split('/') if x.startswith('S2S')][0]
#         model_ckpt.split('/ckpts/')[1].split('/')[0]
        model_params, data_params = utils.params_from_name(model_name)
        # load dataset of model to access tokenizer
        ### Note: think about pickling this for faster loading !!!!!
        self.ds = MolecularDataset(data_dir='/home/oestreichm/molly/MolAE-publication-revision/data/',
                            filename=data_params['filename'],
                            enumerated=model_params['enumerated'])
        

        self.ae = VAE(rnn_type = model_params['rnn_type'], 
                 hidden_size = model_params['hidden_size'], 
                 num_layers = model_params['num_layers'], 
                 dropout = 0, 
                 lr = 0.005, 
                 batch_size = 1,
                 bidirectional_en = model_params['bidirectional_en'], 
                 bidirectional_de = model_params['bidirectional_de'], 
                 extra_latent = model_params['extra_latent'], 
                 latent_size = model_params['latent_size'], 
                 attention = model_params['attention'], 
                 save_latent = False, 
                 save_latent_dir = 'NULL',
                 log_token_position_acc = False, 
                 tokenizer = self.ds.tokenizer, 
                 max_seq_len = self.ds.max_seq_length, 
                 enumerated = model_params['enumerated'],
                 p = 0.3)
        if torch.cuda.is_available():
            ckpt = torch.load(model_ckpt)
        else:
            ckpt = torch.load(model_ckpt,
                    map_location=torch.device('cpu'))

        self.ae.load_state_dict(ckpt['state_dict'])
        self.ae.eval()
        for name, param in self.ae.named_parameters():
            param.requires_grad = False
            
        
    def encode(self, smiles):
        batch = self.ds.smiles2onehot(smiles)
        embs = self.ae.encoder(batch).squeeze(0)
        embs = embs.detach()
        return embs
    
class selfiesVaeEnc(encoder):
    
    def __init__(self, model_ckpt):
        super().__init__()
        model_name = [x for x in model_ckpt.split('/') if x.startswith('S2S')][0]
#         model_ckpt.split('/ckpts/')[1].split('/')[0]
        model_params, data_params = utils.params_from_name(model_name)
        # load dataset of model to access tokenizer
        ### Note: think about pickling this for faster loading !!!!!
        self.ds = SELFIESDataset(data_dir='/home/oestreichm/molly/MolAE-publication-revision/data/',
                            filename=data_params['filename'],
                            enumerated=model_params['enumerated'])
        tokenizer = DictConfig({'vocabulary': self.ds.alphabet})
        

        self.ae = sVAE(rnn_type = model_params['rnn_type'], 
                 hidden_size = model_params['hidden_size'], 
                 num_layers = model_params['num_layers'], 
                 dropout = 0, 
                 lr = 0.005, 
                 batch_size = 1,
                 bidirectional_en = model_params['bidirectional_en'], 
                 bidirectional_de = model_params['bidirectional_de'], 
                 extra_latent = model_params['extra_latent'], 
                 latent_size = model_params['latent_size'], 
                 attention = model_params['attention'], 
                 save_latent = False, 
                 save_latent_dir = 'NULL',
                 log_token_position_acc = False, 
                 tokenizer = tokenizer, 
                 max_seq_len = self.ds.max_seq_length, 
                 enumerated = model_params['enumerated'],
                 p = 0.3)
        if torch.cuda.is_available():
            ckpt = torch.load(model_ckpt)
        else:
            ckpt = torch.load(model_ckpt,
                    map_location=torch.device('cpu'))

        self.ae.load_state_dict(ckpt['state_dict'])
        self.ae.eval()
        for name, param in self.ae.named_parameters():
            param.requires_grad = False
    
    def encode(self, selfies):
        
        selfies = [self.ds.encode_selfie(s)[0] for s in selfies]
        batch = torch.stack(selfies)
        embs = self.ae.encoder(batch).squeeze(0)
        embs = embs.detach()
        return embs
        
    
    
class AeEnc(encoder):
    
    def __init__(self, model_ckpt, tokenizer = None, max_len = None):
        super().__init__()
        model_name = [x for x in model_ckpt.split('/') if x.startswith('S2S')][0]
        model_params, data_params = utils.params_from_name(model_name)
        # load dataset of model to access tokenizer
        ### Note: think about pickling this for faster loading !!!!!
        if tokenizer is None:
            self.ds = MolecularDataset(data_dir='/home/oestreichm/molly/MolAE-publication-revision/data/',
                                filename=data_params['filename'],
                                enumerated=model_params['enumerated'])
            self.tokenizer = self.ds.tokenizer
            self.max_seq_length = self.ds.max_seq_length
            self.smiles2onehot = self.ds.smiles2onehot
        else:
            self.tokenizer = tokenizer
            self.max_seq_length = max_len
            

#         print(self.ds.tokenizer)
        print(model_params)
        self.ae = AE(rnn_type = model_params['rnn_type'],
                        hidden_size = model_params['hidden_size'],
                        num_layers = model_params['num_layers'],
                        dropout = 0, 
                        lr = 0.005,
                        batch_size = 1,
                        bidirectional_en = model_params['bidirectional_en'], 
                        bidirectional_de = model_params['bidirectional_de'], 
                        extra_latent = model_params['extra_latent'], 
                        latent_size = model_params['latent_size'], 
                        attention = model_params['attention'], 
                        save_latent = False, 
                        save_latent_dir = "Null",
                        log_token_position_acc = False, 
                        tokenizer = self.tokenizer, #
                        max_seq_len = self.max_seq_length, #
                        enumerated = model_params['enumerated'])
        print(self.ae)
        if torch.cuda.is_available():
            ckpt = torch.load(model_ckpt)
        else:
            ckpt = torch.load(model_ckpt,
                    map_location=torch.device('cpu'))

        self.ae.load_state_dict(ckpt['state_dict'])
        self.ae.eval()
        for name, param in self.ae.named_parameters():
            param.requires_grad = False
        
    def smiles2onehot(self, smiles):
        ''' smiles: a list '''
        tokenized_smiles = self.tokenizer.encode(smiles)
        # max_seq_length = max([tokenized_smiles[i].shape[0] for i in range(len(tokenized_smiles))])
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
        
    def encode(self, smiles):
        batch = self.smiles2onehot(smiles)
        embs = self.ae.encoder(batch).squeeze(0)
        embs = embs.detach()
        return embs
    
    def decode(self, encs, smiles):
        batch = self.ds.smiles2onehot(smiles)
        if (self.ae.rnn_type == 'LSTM'):
            output, hidden, cell = self.ae.decoder(batch, encs)
        else:
            output, hidden = self.ae.decoder(batch, encs)
        return output
    



class SelfiesAeEnc(encoder):
    
    def __init__(self, model_ckpt):
        super().__init__()
        model_name = [x for x in model_ckpt.split('/') if x.startswith('S2S')][0]
        model_params, data_params = utils.params_from_name(model_name)
        print(data_params['filename'])
        # load dataset of model to access tokenizer
        ### Note: think about pickling this for faster loading !!!!!
        self.ds = SELFIESDataset(data_dir='/home/oestreichm/molly/MolAE-publication-revision/data/',
                            filename=data_params['filename'],
                            enumerated=model_params['enumerated'])
        tokenizer = DictConfig({'vocabulary': self.ds.alphabet})

        self.ae = sAE(rnn_type = model_params['rnn_type'],
                        hidden_size = model_params['hidden_size'],
                        num_layers = model_params['num_layers'],
                        dropout = 0, 
                        lr = 0.005,
                        batch_size = 1,
                        bidirectional_en = model_params['bidirectional_en'], 
                        bidirectional_de = model_params['bidirectional_de'], 
                        extra_latent = model_params['extra_latent'], 
                        latent_size = model_params['latent_size'], 
                        attention = model_params['attention'], 
                        save_latent = False, 
                        save_latent_dir = "Null",
                        log_token_position_acc = False, 
                        tokenizer = tokenizer, #
                        max_seq_len = self.ds.max_seq_length, #
                        enumerated = model_params['enumerated'])
        if torch.cuda.is_available():
            ckpt = torch.load(model_ckpt)
        else:
            ckpt = torch.load(model_ckpt, map_location=torch.device('cpu'))

        self.ae.load_state_dict(ckpt['state_dict'])
        self.ae.eval()
        for name, param in self.ae.named_parameters():
            param.requires_grad = False
            
        
    def encode(self, selfies):
        
        selfies = [self.ds.encode_selfie(s)[0] for s in selfies]
        batch = torch.stack(selfies)
        embs = self.ae.encoder(batch).squeeze(0)
        embs = embs.detach()
        return embs
    
    def decode(self, encs, selfies):
        selfies = [self.ds.encode_selfie(s)[0] for s in selfies]
        batch = torch.stack(selfies)
        if (self.ae.rnn_type == 'LSTM'):
            output, hidden, cell = self.ae.decoder(batch, encs)
        else:
            output, hidden = self.ae.decoder(batch, encs)
        return output