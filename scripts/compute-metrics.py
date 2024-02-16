
string_type = 'smiles'

import selfies as sf
from difflib import SequenceMatcher
from Levenshtein import distance as lev_dist
import pandas as pd
from os import listdir

dir = '/test-recs/'
files = [f for f in listdir(dir)]
if string_type == 'selfies':
    files = [f for f in files if 'selfies' in f]
else:
     files = [f for f in files if not 'selfies' in f]

def mean_sim(ref, rec):
        ref = list(ref)
        rec = list(rec)
        match_counter = 0
        for t, (c, d) in enumerate(zip(ref, rec)):
            if (c == d):
                match_counter += 1

        sim = match_counter / max(len(ref), len(rec)) # in case rec has more tokens in the end
        return sim

def mean_sim_selfies(ref, rec):
       
        ref = list(sf.split_selfies(ref))
        if len(ref) == 0:
                print(f"a with length zero: {ref}")
        rec = list(sf.split_selfies(rec))
        match_counter = 0
        for t, (c, d) in enumerate(zip(ref, rec)):

            if (c == d):
                match_counter += 1
        if len(ref) == 0:
            return 0
        else:
            sim = match_counter / max(len(ref), len(rec))
            return sim


def seq_matcher(ref, rec):
    return SequenceMatcher(None, ref, rec).ratio()
    
def levenshtein(ref, rec):
    d = lev_dist(ref, rec)
    return 1-(d/len(ref)) #to make it a sim

def levenshtein_selfies(ref, rec):
    ref = list(sf.split_selfies(ref))
    rec = list(sf.split_selfies(rec))
    d = lev_dist(ref, rec)
    return 1-(d/len(ref)) #to make it a sim

res = pd.DataFrame()
for f in files:
    print(f'currectly processing: {f}')
    row = [f]
    data = pd.read_csv(dir + f)
    sims = []

    # compute mean sims
    if string_type == 'selfies':
        for ref, rec in zip(data.original.values, data.reconstruction.values):
            sims.append(mean_sim_selfies(ref, rec))
        row.append(sum(sims)/len(sims))
    else:
        for ref, rec in zip(data.original.values, data.reconstruction.values):
            sims.append(mean_sim(ref, rec))
        row.append(sum(sims)/len(sims))

    # compute seq_matcher
    sims = []
    for ref, rec in zip(data.original.values, data.reconstruction.values):
        sims.append(seq_matcher(ref, rec))
    row.append(sum(sims)/len(sims))

    # compute levenshtein
    sims = []
    if string_type == 'selfies':
        for ref, rec in zip(data.original.values, data.reconstruction.values):
            sims.append(levenshtein_selfies(ref, rec))
        row.append(sum(sims)/len(sims))
    else:
        for ref, rec in zip(data.original.values, data.reconstruction.values):
            sims.append(levenshtein(ref, rec))
        row.append(sum(sims)/len(sims))

    res = pd.concat([res, pd.DataFrame(row, index = ['file', 'mean_sim', 'seq_matcher', 'levenshtein']).T])

if string_type == 'selfies':
    res.to_csv('/metrics_selfies.csv')
else:
    res.to_csv('/metrics_smiles.csv')