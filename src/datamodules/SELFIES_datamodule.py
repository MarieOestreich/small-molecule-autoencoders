import pytorch_lightning as pl
from src.datamodules.MolecularDataset import SELFIESDataset
from torch.utils.data import DataLoader
from typing import Optional

class SELFIES_datamodule(pl.LightningDataModule):
    def __init__(self, data_dir, filename, batch_size: int = 32, num_workers=4, pin_memory=False, enumerated = False, train=True):
        super().__init__()
        self.data_dir = data_dir
        self.filename = filename
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.enumerated = enumerated
        self.molecular_dataset = SELFIESDataset(
            train=train, data_dir=self.data_dir, filename= self.filename, enumerated = self.enumerated)
        self.alphabet = self.molecular_dataset.alphabet

    def setup(self, stage: Optional[str] = None):
        self.smiles_val = SELFIESDataset(
            train=False, alphabet = self.alphabet, data_dir=self.data_dir, filename= self.filename, enumerated = self.enumerated)
        self.smiles_train = SELFIESDataset(
            train=True, alphabet = self.alphabet, data_dir=self.data_dir, filename= self.filename, enumerated = self.enumerated)

    def train_dataloader(self):
        return DataLoader(self.smiles_train,
            batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, drop_last=True, pin_memory=self.pin_memory)
            
    def val_dataloader(self):
        return DataLoader(self.smiles_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, drop_last=True, pin_memory= self.pin_memory)
        
    def test_dataloader(self):
        return DataLoader(self.smiles_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, drop_last=True, pin_memory=self.pin_memory)
        
