# Small Molecule Autoencoders: Architecture Engineering to Optimize Latent Space Utility and Sustainability

## Embedding you own SMILES or SELFIES with pre-trained models
We have added checkpoints of the the three best performing models ready for use:

- ```model-checkpoints/S2S_gru_a0_e-0_ls128_h128_l1_fullMOSES_b0-0_670487_50.ckpt```: The Autoencoder with the best full reconstruction rate for **SMILES**.
- ```model-checkpoints/S2S_gru_a1_e-1_ls128_h128_l3_sub1-5x-can2enum_b0-0_26225_200.ckpt```: The **SMILES** autoencoder trained with canonical SMILES as input and enumerated SMILES as targets. This model stood out due to its good latent space organisation.
- ```model-checkpoints/S2S_gru_a0_e-0_ls128_h128_l1_fullMOSES-selfies_b0-0_116739_50.ckpt```: The Autoencoder with the best full reconstruction rate for **SELFIES**.

**NOTE**: *All* checkpoints of the models investigated in our paper including source code and scripts for testing are depsited on Zenodo.

## evaluate reconstruction performance on the test set

For convenience reasons, test molecules and their reconstruction attempts for all models have been generated with ```/scripts/predict-all.py``` and are deposited in the 'test-recs' folder.
These files can be used to compute the **Full Reconstruction** rates with ```/scripts/compute-full-reconstruction.py``` and the **Mean Similarity, Levenshtein Distance and SeqMatcher Similarity** with ```/scripts/compute-metrics.py```.

## evaluate the latent space

To evaluate the latent space of an existing checkpoint, run:

```
scripts/evaluate-latent-space.py -ckpt='S2S_gru_a1_e-0_ls128_h128_l3_sub1_b0-0_26225_200.ckpt'
```
PCA and histomgram will be returned as a PDF.

Note, that the checkpoints follow a naming convention that provides all necessary parameters to build the model that fits the checkpoint. For more information, see checkpoint naming convention below.

## train new model

To train a new model, run:

```
scripts/train.py -n='S2S_gru_a0_e-0_ls8_h8_l1_sub1-selfies_b0-0' -bs=1024 -e=1 -g=4
```
With `-n` being the model name according to the naming convention (see below), `-bs` being the batch size, `e` the number of training epochs and `-g` the number of GPUs to use.

## checkpoint naming convention

Explained on the example `S2S_gru_a0_e-0_ls8_h8_l1_sub1-selfies_b0-0`:

-  `S2S`: string to string, do not change this part
-  `gru`: alternatively `lstm`, indicated which RNN type to use
-  `a0`: whether or not to use attention. `a0` = no attention, `a1` = with attention
-  `e-0`: whether or not the input or output is enumerated (`e-0` = no, `e-1` = yes)
-  `ls8`: the latent size (here 8)
-  `h8`: the hidden size (here 8)
-  `l1`: number of layers (here 1)
-  `sub1-selfies`: the dataset. One of `sub1-selfies` (50k subset selfies), `sub1`(50k subset smiles), `fullMOSES-selfies`(full set selfies), `fullMOSES` (full set smiles)
-  `b0-0`: Do not change.

Provided checkpoints further include the seed and the number of epochs.
