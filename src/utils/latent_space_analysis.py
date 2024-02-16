import torch
from src.utils.enumerate_smiles import enumerate_smiles
import pandas as pd
import random
import copy
from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import selfies as sf 
from rdkit import DataStructs, Chem
import pytorch_lightning as pl

pl.seed_everything(126)


# adapted from https://stackoverflow.com/questions/470690/how-to-automatically-generate-n-distinct-colors

kelly_colors = [
    '#FFB300', # Vivid Yellow
    '#803E75', # Strong Purple
    '#FF6800', # Vivid Orange
    '#A6BDD7', # Very Light Blue
    '#C10020', # Vivid Red
    '#CEA262', # Grayish Yellow
    '#817066', # Medium Gray

    # The following don't work well for people with defective color vision
    '#007D34', # Vivid Green
    '#F6768E', # Strong Purplish Pink
    '#00538A', # Strong Blue
    '#FF7A5C', # Strong Yellowish Pink
    '#53377A', # Strong Violet
    '#FF8E00', # Vivid Orange Yellow
    '#B32851', # Strong Purplish Red
    '#F4C800', # Vivid Greenish Yellow
    '#7F180D', # Strong Reddish Brown
    '#93AA00', # Vivid Yellowish Green
    '#593315', # Deep Yellowish Brown
    '#F13A13', # Vivid Reddish Orange
    '#232C16', # Dark Olive Green
    ]

class encoder():
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
    
class latentEval():
    def __init__(self, encoder, inputType, trainSet, testSet):
        '''
        params:
            encoder: an object of the encoder class
            inputType: either "smiles", "selfies" or "graph", depending on the model input
            trainSet: a list of SMILES the the model was trained on
            testSet: a list of SMILES from which random ones will be used for evaluation
        '''
        self.encoder = encoder
        if inputType not in ['smiles', 'selfies', 'graph']:
            raise ValueError(f'provided input type is "{inputType}", but must be one of: "smiles", "selfies", "graph"')
        else:
            print(f'input type set to "{inputType}"')
            self.inputType = inputType
        self.trainSet = trainSet
        self.testSet = testSet
    
    def evaluate(self, n):
        '''runs the embedding evaluation depending on the input type
        
        params:
            n: how many SMILES should be randomly picked from the testSet for evaluaiton
        '''
        if n > len(self.testSet):
            print(f" ... testSet is not large enough to pick {n} different SMILES. Using {len(self.testSet)} SMILES instead.")
            mols = self.testSet
        else:
            random.seed(126)
            mols = random.sample(self.testSet, n)
        
        print(" ... Embedding train set ... ")
        trainEmbs = self.encoder.encode(self.trainSet) # (len(smiles), |z|)
        
        if self.inputType == 'smiles':
            # enumerate 5x
            enums = []
            print(" ... Enumerating SMILES ... ")
            for i in range(len(mols)):
                enums.append(enumerate_smiles(mols[i], 4)) # nested list len(SMILES) * 5
                
            # embed
            enumsEmbs = []
            print(" ... Embedding evaluation SMILES ... ")
            random.seed(126)
            for i in enums:
                enumsEmbs.append(self.encoder.encode(i))
            enumsEmbs = torch.stack(enumsEmbs)
            
            
            print(" ... Distance comparison ... ")
            f1 = self._distVsRand(enumsEmbs, trainEmbs, mols)
            print(" ... PCA ... ")
            f2 = self._pca(enumsEmbs, trainEmbs, mols)
            
            return f1, f2

        if self.inputType == 'selfies':
            # enumerate 5x
            enums = []
            print(" ... Enumerating SMILES ... ")
            for i in range(len(mols)):
                enumSmiles = enumerate_smiles(sf.decoder(mols[i]), 4)
                enumSelfies = [sf.encoder(s) for s in enumSmiles]
                enums.append(enumSelfies) # nested list len(SMILES) * 5
               
            # embed
            enumsEmbs = []
            print(" ... Embedding evaluation SMILES ... ")
            random.seed(126)
            for i in enums:
                enumsEmbs.append(self.encoder.encode(i))
            enumsEmbs = torch.stack(enumsEmbs)
            
            
            print(" ... Distance comparison ... ")
            f1 = self._distVsRand(enumsEmbs, trainEmbs, [sf.decoder(x) for x in mols])
            print(" ... PCA ... ")
            f2 = self._pca(enumsEmbs, trainEmbs, [sf.decoder(x) for x in mols])
            
            return f1, f2
            
        if self.inputType == 'graph':
            pass
        
        
    def mutationEffect(self, n):
        SMILES = random.sample(self.testSet, 1000)
        SELFIES = [sf.encoder(s) for s in SMILES]
        print('... Translating trainset so selfies...')
        self.trainSetSelfies = [sf.encoder(s) for s in self.trainSet]
        print('... Translating testset so selfies...')
        self.testSetSelfies = [sf.encoder(s) for s in self.testSet]

        sims = []
        print('... Mutating selfies...')
        for s in SELFIES:
            sfs, sms = self._randPermute(s, n)
            sims += sms.tolist()
            
        print('... Computing sims of test mols...')
        sims_can = []
        for i in range(len(SELFIES)-1):
            for j in range(i+1, len(SELFIES)-1):
                sims_can.append(self._selfiesTanimoto(SELFIES[i], SELFIES[j]))
            
        return sims, sims_can
    
    def _distVsRand(self, enumsEmbs, trainEmbs, SMILES):
        d_sim = []
        for i in enumsEmbs:
            d_sim.append(self._pwL2(i).mean())
        # randomly pick 1000 mols from train set:
        if trainEmbs.shape[0] > 1000:
            random.seed(126)
            trainEmbs = trainEmbs[random.sample(range(trainEmbs.shape[0]), 1000)]
        d_rand = self._pwL2(trainEmbs)
        counts, bins = np.histogram(d_rand, bins = 50)
        f = plt.figure(figsize=(5,5))
        plt.stairs(counts, bins, label = 'random')
        for i in range(len(SMILES)):
            plt.vlines(d_sim[i], ymin = 0, ymax = counts.max(), label=SMILES[i], color = kelly_colors[i])
        plt.legend(loc='upper right', bbox_to_anchor=(2.5,1))
        plt.title('Embedding distances')
        plt.xlabel('L2-distance')
        plt.ylabel('Frequency')
        plt.show()
        
        return f 
    
    def _pwL2(self, embs):
        d = torch.cdist(embs, embs, p=2)
        d = d + 1
        d = torch.tril(d, -1)
        d = d.reshape(-1,1)
        d = d[d>0]
        d -= 1
        return d
    
    def _pca(self, enumsEmbs, trainEmbs, SMILES):
        if trainEmbs.shape[0] > 10000:
            random.seed(126)
            trainEmbs = trainEmbs[random.sample(range(trainEmbs.shape[0]), 10000)]
        embs = torch.cat((enumsEmbs.reshape(enumsEmbs.shape[0]*enumsEmbs.shape[1], -1), trainEmbs))
        
        pca = PCA(n_components=2)
        res = pca.fit_transform(embs.numpy())
        X = [x[0] for x in res]
        Y = [x[1] for x in res]
        
        f = plt.figure(figsize=(5,5))
        plt.tight_layout()
        plt.plot(X, Y, 'o', alpha = 1, markersize = 1.5, color = 'lightgrey')
        for i in range(enumsEmbs.shape[0]):
            plt.plot(X[i*enumsEmbs.shape[1]], Y[i*enumsEmbs.shape[1]], 'o', alpha = 1, markersize = 7, color = kelly_colors[i], label = SMILES[i])
            plt.plot(X[i*enumsEmbs.shape[1]+1:(i+1)*enumsEmbs.shape[1]], Y[i*enumsEmbs.shape[1]+1:(i+1)*enumsEmbs.shape[1]], '1', alpha = 1, markersize = 11, color = kelly_colors[i])
            
        plt.xlabel(f'PCA-1 {round(pca.explained_variance_ratio_[0]*100, 2)} %')
        plt.ylabel(f'PCA-2 {round(pca.explained_variance_ratio_[1]*100, 2)} %')
        plt.title('5x enumeration of mols')
        plt.legend(loc='upper right', bbox_to_anchor=(2.5,1))
        plt.show()
        
        return f
    
    def _randPermute(self, selfie, n):
        vocab = list(sf.get_alphabet_from_selfies(self.trainSetSelfies + self.testSetSelfies))
        vocab.sort(reverse=False)
        new_selfies = []
        seed = 126
        count = 0
        while count < 20: 
            random.seed(seed)
            _selfie = list(sf.split_selfies(selfie))
            idx = random.choices(range(len(_selfie)), k=1)[0]
            _vocab = copy.deepcopy(vocab)
            _vocab.remove(_selfie[idx])
            random.seed(seed)
            tkn = random.choices(_vocab, k=1)[0]
            _selfie[idx] = tkn
            _selfie = ''.join(_selfie)
            try:
                _selfie = sf.encoder(sf.decoder(_selfie))
                if _selfie not in new_selfies and _selfie != selfie:
                    new_selfies.append(_selfie)
                    count += 1
                seed += 1
            except:
                count = count
                seed += 1
         
        sims = [self._selfiesTanimoto(s, selfie) for s in new_selfies]
        new_smiles = [sf.decoder(s) for s in new_selfies]
        df = pd.DataFrame([new_smiles, sims], index = ['smiles', 'sims']).T.sort_values('sims', ascending=False)
        return df['smiles'].values[:n].tolist(), df['sims'].values[:n]
            
    def _selfiesTanimoto(self, s1, s2):
        s1 = sf.decoder(s1)
        s2 = sf.decoder(s2)
        ms = [Chem.MolFromSmiles(s) for s in [s1, s2]]
        fps = [Chem.RDKFingerprint(x) for x in ms]
        sim = DataStructs.FingerprintSimilarity(fps[0],fps[1])
        return sim
            