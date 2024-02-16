import pytorch_lightning as pl
import torch
import selfies as sf
from torch import nn
from difflib import SequenceMatcher
import torch.nn.functional as F
import pandas as pd
import numpy as np


class RNNEncoder(nn.Module):
    def __init__(self, rnn_type, input_size, num_layers, hidden_size, bidirectional, extra_latent, latent_size,
                 attention, device):
        super(RNNEncoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type
        self.extra_latent = extra_latent
        self.latent_size = latent_size
        self.attention = attention
        self.device = device

        if (self.rnn_type == 'LSTM'):
            self.rnn = nn.LSTM(input_size=self.input_size,
                               hidden_size=self.hidden_size,
                               bidirectional=self.bidirectional,
                               num_layers=self.num_layers)

        else:
            self.rnn = nn.GRU(input_size=self.input_size,
                              hidden_size=self.hidden_size,
                              bidirectional=self.bidirectional,
                              num_layers=self.num_layers)

        # linear layer for combination of hidden and cell state to one latent vector
        self.fc1 = nn.Linear((hidden_size * 2), hidden_size)
        # linear layer to reduce dimension of latent vector to 1D if num_layers > 1
        self.fc2 = nn.Linear(num_layers, 1)

        if (self.extra_latent):
            self.latent = nn.Linear(self.hidden_size, self.latent_size)

        if (self.attention):
            self.attention_layer = nn.Linear(self.hidden_size, 1)

    def forward(self, input):
        # input: batch_size x sequence_length x vocab_len
        # input.transpose(0,1) -> input:[sequence_length, batch_size, input_size (=vocab_len)]
        # hidden: num_layers x batch_size x hidden_size, outputs: sequence_length x batch_size x hidden_size
        # cell: num_layers x batch_size x hidden_size
        if (self.rnn_type == 'LSTM'):
            outputs, (hidden, cell) = self.rnn(input.transpose(0, 1))
            if (self.attention):
                # outputs: seq_len, batch_size, hidden_size
                a = F.softmax(self.attention_layer(outputs))
                # a: seq_len, batch_size, 1
                a = a.permute(1, 2, 0)
                # a: batch_size, 1, seq_len
                outputs = outputs.permute(1, 0, 2)
                # outputs: batch_size, seq_len, hidden_size
                weighted = torch.bmm(a, outputs)
                # weighted: batch_size, 1, hidden_size
                weighted = weighted.permute(1, 0, 2)
                # concatenate hidden and cell state for one latent vector
                states = torch.cat((weighted, cell), -1)
                # reduce dimension of latent vec from hidden_size * 2 to hidden_size
                latent_vec = self.fc1(states)
            else:
                # concatenate hidden and cell state for one latent vector
                states = torch.cat((hidden, cell), -1)
                # reduce dimension of latent vec from hidden_size * 2 to hidden_size
                latent_vec = self.fc1(states)

        else:
            outputs, hidden = self.rnn(input.transpose(0, 1))
            if (self.attention):
                # outputs: seq_len, batch_size, hidden_size
                a = F.softmax(self.attention_layer(outputs))
                # a: seq_len, batch_size, 1
                a = a.permute(1, 2, 0)
                # a: batch_size, 1, seq_len
                outputs = outputs.permute(1, 0, 2)
                # outputs: batch_size, seq_len, hidden_size
                weighted = torch.bmm(a, outputs)
                # weighted: batch_size, 1, hidden_size
                weighted = weighted.permute(1, 0, 2)
                # weighted: 1, batch_size, hidden_size
                latent_vec = weighted
            else:
                latent_vec = hidden

        if (self.num_layers > 1):
            if (self.attention == False):
                # (num_layers, batch_size, hidden_size) -> (hidden_size, batch_size, 1)
                latent_vec = self.fc2(latent_vec.transpose(0, 2))
                # -> (1, batch_size, hidden_size)
                latent_vec = latent_vec.transpose(2, 0)

        if (self.extra_latent):
            latent_vec = self.latent(latent_vec)

        return latent_vec


class RNNDecoder(nn.Module):
    def __init__(self, rnn_type, output_size, hidden_size, num_layers, dropout, bidirectional, extra_latent,
                 latent_size, attention, device):
        super(RNNDecoder, self).__init__()

        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type
        self.extra_latent = extra_latent
        self.latent_size = latent_size
        self.attention = attention
        self.device = device

        if (self.rnn_type == 'LSTM'):
            self.rnn = nn.LSTM(input_size=self.output_size,
                               hidden_size=self.hidden_size,
                               bidirectional=self.bidirectional,
                               num_layers=self.num_layers)

        else:
            self.rnn = nn.GRU(input_size=self.output_size,
                              hidden_size=self.hidden_size,
                              bidirectional=self.bidirectional,
                              num_layers=self.num_layers)

        # dense output layer to get distribution over vocabulary
        self.fc = nn.Linear(self.hidden_size, self.output_size)

        # linear layer to reconstruct hidden_state from latent vector
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        # linear layer to reconstruct cell_state from latent vector
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # linear layer to reconstruct dimension from latent vector if num_layers > 1
        if (self.num_layers > 1):
            self.fc3 = nn.Linear(1, 1)
            self.fc4 = nn.Linear(1, 1)
        if (self.num_layers == 3):
            self.fc5 = nn.Linear(1, 1)
        if (self.extra_latent):
            self.latent = nn.Linear(self.latent_size, self.hidden_size)

    def forward(self, x_input, encoder_states):

        # x_input: [batch_size, seq_len, vocab_len] -> [seq_len, batch_size, vocab_len]
        x_input = x_input.transpose(1, 0)

        if (self.extra_latent):
            encoder_states = self.latent(encoder_states)

        if (self.num_layers > 1):
            # (1, batch_size, hidden_size) -> (hidden_size, batch_size, num_layers)
            encoder_states_1 = self.fc3(encoder_states.transpose(0, 2))
            encoder_states_2 = self.fc4(encoder_states.transpose(0, 2))

            if (self.num_layers == 3):
                encoder_states_3 = self.fc5(encoder_states.transpose(0, 2))
                encoder_states = torch.cat((encoder_states_1, encoder_states_2, encoder_states_3), 2)
            else:
                encoder_states = torch.cat((encoder_states_1, encoder_states_2), 2)
            # -> (num_layers, batch_size, hidden_size)
            encoder_states = encoder_states.transpose(2, 0)

        if (self.rnn_type == 'LSTM'):
            # reconstruct hidden_state
            hidden_state = self.fc1(encoder_states)
            # reconstruct cell_state
            cell_state = self.fc2(encoder_states)
            # encoder_hidden_states: num_layers x batch_size x hidden_size
            # output: (seq_len, batch_size, hidden_size)
            # hidden: (num_layers, batch_size, hid_dim)
            output, (hidden, cell) = self.rnn(x_input, (hidden_state, cell_state))
            #  -> prediction: (seq_len, batch_size, output_dim)
            prediction = self.fc(output)
            return prediction, hidden, cell

        else:
            encoder_states = encoder_states.contiguous()
            output, hidden = self.rnn(x_input, encoder_states)
            #  -> prediction: (seq_len, batch_size, output_dim)
            prediction = self.fc(output)
            return prediction, hidden


class Autoencoder(pl.LightningModule):
    def __init__(self, rnn_type, hidden_size, num_layers, dropout, lr, batch_size,
                 bidirectional_en, bidirectional_de, extra_latent, latent_size, attention, save_latent, save_latent_dir,
                 log_token_position_acc, tokenizer, max_seq_len, enumerated):
        super(Autoencoder, self).__init__()

        self.dropout = dropout
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional_en = bidirectional_en
        self.bidirectional_de = bidirectional_de
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr
        self.batch_size = batch_size
        self.rnn_type = rnn_type
        self.extra_latent = extra_latent
        self.latent_size = latent_size
        self.tokenizer = tokenizer
        self.input_size = len(self.tokenizer.vocabulary)
        self.output_size = self.input_size
        self.attention = attention
        self.vocab = []
        self.char_counter = {}
        self.mismatch_counter = {}
        self.latent_vecs = []
        self.smiles = []
        self.save_latent = save_latent
        self.save_latent_dir = save_latent_dir
        self.latent = []
        self.log_token_position_acc = log_token_position_acc
        self.max_seq_len = max_seq_len
        self.token_loc_counter = np.zeros(max_seq_len)
        self.token_loc_correct_counter = np.zeros(max_seq_len)
        self.enumerated = enumerated
        self.vocab_itos = {i:t for i,t in enumerate(self.tokenizer.vocabulary)}


        for char in self.tokenizer.vocabulary:
            self.vocab.append(char)

        for char in self.vocab:
            self.char_counter[char] = 0
            self.mismatch_counter[char] = 0

        # instantiate Encoder and Decoder
#         if len(self.hidden_size) == 2:
#             self.encoder = RNNEncoder(self.rnn_type, self.input_size, self.num_layers[0], self.hidden_size[0],
#                                   self.bidirectional_en, self.extra_latent, self.latent_size, self.attention,
#                                   self.device)
#             self.decoder = RNNDecoder(self.rnn_type, self.output_size, self.hidden_size[1], self.num_layers[1], self.dropout,
#                                       self.bidirectional_de, self.extra_latent, self.latent_size, self.attention,
#                                       self.device)
#         else:
#             self.encoder = RNNEncoder(self.rnn_type, self.input_size, self.num_layers[0], self.hidden_size[0],
#                                       self.bidirectional_en, self.extra_latent, self.latent_size, self.attention,
#                                       self.device)
#             self.decoder = RNNDecoder(self.rnn_type, self.output_size, self.hidden_size[0], self.num_layers[0], self.dropout,
#                                       self.bidirectional_de, self.extra_latent, self.latent_size, self.attention,
#                                       self.device)
        self.encoder = RNNEncoder(self.rnn_type, self.input_size, self.num_layers, self.hidden_size,
                                              self.bidirectional_en, self.extra_latent, self.latent_size, self.attention,
                                              self.device)
        self.decoder = RNNDecoder(self.rnn_type, self.output_size, self.hidden_size, self.num_layers, self.dropout,
                                  self.bidirectional_de, self.extra_latent, self.latent_size, self.attention,
                                  self.device)

    def training_step(self, batch, batch_idx):

        loss, _, _ = self.model_run(batch, False)

        # logging
        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        # x: (batch_size, sequence_length, vocab_len), y: (batch_size, sequence_length, vocab_len) -> offsetted by one character in comparison to x\n",
        if (self.enumerated):
            x, y, y_can = batch
        else:
            x, y = batch
        batch_size = x.shape[0]
        x_len = x.shape[1]

        loss, output_dim, output = self.model_run(batch, False)

        mean_sim, eq_proc = self.compute_metrics(x_len, batch_size, output_dim, output, x, y, False, False)

        # logging
        self.log('val_loss', loss, on_epoch=True, sync_dist=True)
        self.log('eq_proc', eq_proc, on_epoch=True, sync_dist=True)
        self.log('mean_sim', mean_sim, on_epoch=True, sync_dist=True)

        return {'val_loss': loss}

    def test_step(self, batch, batch_idx):
        # x: (batch_size, sequence_length, vocab_len), y: (batch_size, sequence_length, vocab_len) -> offsetted by one character in comparison to x\n",
        if (self.enumerated):
            x, y, y_can = batch
        else:
            x, y = batch
        batch_size = x.shape[0]
        x_len = x.shape[1]

        loss, output_dim, output = self.model_run(batch, self.save_latent)

        mean_sim, eq_proc = self.compute_metrics(x_len, batch_size, output_dim, output, x, y, True, self.save_latent)

        return{'test_full_rec': eq_proc, 'test_mean_sim': mean_sim}

    def test_epoch_end(self, outputs):
        
        avg_full_rec = [x['test_full_rec'] for x in outputs]
        avg_full_rec = sum(avg_full_rec)/len(avg_full_rec)
        avg_mean_sim = [x['test_mean_sim'] for x in outputs]
        avg_mean_sim = sum(avg_mean_sim)/len(avg_mean_sim)

        
        self.log('full_rec', avg_full_rec, on_epoch=True, sync_dist=True)
        self.log('mean_sim', avg_mean_sim, on_epoch=True, sync_dist=True)

        return {'full_rec': avg_full_rec, 'mean_sim': avg_mean_sim}

    def log_token_pos_acc(self):
        self.token_loc_sim = np.zeros(self.max_seq_len)

        for c, (x, y) in enumerate(zip(self.token_loc_correct_counter, self.token_loc_counter)):
            if (y != 0):
                self.token_loc_sim[c] = x / y
            else:
                self.token_loc_sim[c] = 0

        for c, token_loc in enumerate(self.token_loc_sim):
            self.log(str(c + 1) + '_pos', token_loc, sync_dist=True)

    def save_latent_as_csv(self):
        self.latent_df = pd.DataFrame(self.smiles, columns=['SMILES'])
        self.latent_df['latent_vector'] = self.latent_vecs
        self.latent_df.to_csv(self.save_latent_dir, index=False)

    def model_run(self, batch, save_latent):
        # x: (batch_size, sequence_length, vocab_len), y: (batch_size, sequence_length, vocab_len) -> offsetted by one character in comparison to x\n",

        if (self.enumerated):
            x, y, y_can = batch
        else:
            x, y = batch

        batch_size = x.shape[0]
        x_len = x.shape[1]

        # use last hidden_state of the encoder as initial hidden state of decoder
        encoder_states = self.encoder(x)

        if (save_latent):
            self.latent = encoder_states.squeeze(0)

        if (self.rnn_type == 'LSTM'):
            if (self.enumerated):
                output, hidden, cell = self.decoder(y_can, encoder_states)
            else:
                output, hidden, cell = self.decoder(x, encoder_states)
        else:
            if (self.enumerated):
                output, hidden = self.decoder(y_can, encoder_states)
            else:
                output, hidden = self.decoder(x, encoder_states)
        output_dim = output.shape[-1]
        # -> [(seq_len*batch_size), vocab_len]
        output_reshaped = output.view(-1, output_dim).type_as(x)
        # -> y: (sequence_length, batch_size, vocab_len)
        y = y.transpose(1, 0)
        # save index of input token in tensor with size [seq_len, batch_size]
        y_reshaped = torch.zeros((x_len, batch_size), device=self.device)
        y_reshaped = y_reshaped.long()

        for i in range(0, x_len):
            for j in range(batch_size):
                y_reshaped[i][j] = torch.argmax(y[i][j])
        # -> [(seq_len) * batch_size]
        y_reshaped = y_reshaped.view(-1)

        # calculate cross-entropy loss
        loss = self.criterion(output_reshaped, y_reshaped)

        return loss, output_dim, output

    def compute_metrics(self, x_len, batch_size, output_dim, output, x, y, test, save_latent):
        # transpose to (batch_size, sequence_length)
        smiles_tensor = torch.zeros((x_len, batch_size, output_dim), device=self.device)
        smiles_tensor = smiles_tensor.type_as(x)

        # take token with highest probability as predicted token
        for s in range(batch_size):
            for t in range((x_len)):
                top1 = torch.argmax(output[t][s])
                smiles_tensor[t][s][top1] = 1

        # -> (batch_size, sequence_length, output_dim)
        smiles_tensor = torch.transpose(smiles_tensor, 1, 0)
        # -> (batch_size, sequence_length, output_dim)
        #y = torch.transpose(y, 1, 0)

        # decode one-hot encoded predicted SMILES to strings
        smiles_list = []
        for s in smiles_tensor:
            selfie = sf.encoding_to_selfies(s.tolist(),
                vocab_itos=self.vocab_itos,
                enc_type = 'one_hot')
            selfie = selfie.replace('^', '').replace('[nop]', '')
            smiles_list.append(selfie)
    
        # smiles_list = self.tokenizer.decode(smiles_tensor)
        # # decode one-hot encoded original SMILES to strings
        # original_smiles_list = self.tokenizer.decode(x)

        original_smiles_list = []
        for s in x:
            selfie = sf.encoding_to_selfies(s.tolist(),
                vocab_itos=self.vocab_itos,
                enc_type = 'one_hot')
            selfie = selfie.replace('^', '').replace('[nop]', '')
            original_smiles_list.append(selfie)




        if (save_latent):
            # save latent vec + original smiles string into file
            for i in range(len(self.latent)):
                self.latent_vecs.append(self.latent[i].tolist())
                self.smiles.append(original_smiles_list[i])

        if(self.enumerated):
            # original_smiles_list = self.tokenizer.decode(y)
            original_smiles_list = []
            for s in y:
                selfie = sf.encoding_to_selfies(s.tolist(),
                    vocab_itos=self.vocab_itos,
                    enc_type = 'one_hot')
                selfie = selfie.replace('^', '').replace('[nop]', '')
                original_smiles_list.append(selfie)

        # compute percentage of reconstructed characters in generated smiles
        if (test):
            sim_list = [self.similar_mismatch_counter(s_or, s) for (s_or, s) in zip(original_smiles_list, smiles_list)]
        else:
            sim_list = [self.similar(s_or, s) for (s_or, s) in zip(original_smiles_list, smiles_list)]
        mean_sim = sum(sim_list) / len(sim_list)
        eq_proc = len([s for (s_or, s) in zip(original_smiles_list, smiles_list) if s == s_or]) / len(smiles_list)

        return mean_sim, eq_proc

    def similar(self, a, b):
        # function to compute similarity of reconstructed smiles to original smiles
        return SequenceMatcher(None, a, b).ratio()

    def similar_mismatch_counter(self, a, b):
       
        a = list(sf.split_selfies(a))
        if len(a) == 0:
                print(f"a with length zero: {a}")
        b = list(sf.split_selfies(b))
        match_counter = 0
        for t, (c, d) in enumerate(zip(a, b)):
            self.char_counter[c] += 1
            self.token_loc_counter[t] += 1
            if (c == d):
                match_counter += 1
                self.token_loc_correct_counter[t] += 1
            else:
                self.mismatch_counter[str(c)] += 1
        if len(a) == 0:
            return 0
        else:
            sim = match_counter / len(a)
            return sim

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {"optimizer": optimizer}
