import sys
from src.utils.enumerator_bjerrum import SmilesEnumerator
import pandas as pd
from rdkit import Chem
import numpy as np
import random


def enumerate_smiles(smiles, n):
    sme = SmilesEnumerator()
    enum = []
    can_s = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    enum.append(can_s)
    random.seed(126)
    for i in range(n):
        smi_enum = sme.randomize_smiles(smiles)
        enum.append(smi_enum)
    return enum