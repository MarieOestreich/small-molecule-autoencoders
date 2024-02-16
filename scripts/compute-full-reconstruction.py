
string_type = 'smiles'

import pandas as pd
from os import listdir

dir = '/test-recs/'
files = [f for f in listdir(dir)]
if string_type == 'selfies':
    files = [f for f in files if 'selfies' in f]
else:
     files = [f for f in files if not 'selfies' in f]



res = pd.DataFrame()
for f in files:
    print(f'currectly processing: {f}')
    row = [f]
    data = pd.read_csv(dir + f)
    sims = []
    for ref, rec in zip(data.original.values, data.reconstruction.values):
        if ref == rec:
            sims.append(1)
        else:
            sims.append(0)
    row.append(sum(sims)/len(sims))
    res = pd.concat([res, pd.DataFrame(row, index = ['file', 'full_reconstruction']).T])

if string_type == 'selfies':
    res.to_csv('/full_rec_selfies.csv')
else:
    res.to_csv('/full_rec_smiles.csv')