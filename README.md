> This projetc has been moved to [ProtAI](https://github.com/sarisabban/ProtAI) (where I use a GAN instead of an LSTM to generate protein backbones instead of sequences)

# ProtLSTM
An LSTM neural network to generate novel protein sequences

## Requirements:
Use the following command (in GNU/Linux) to install all necessary programs and python modules for this script to run successfully:

`sudo apt update && sudo apt full-upgrade && sudo apt install python3-pip python3-pandas python3-numpy python3-tensorflow tensorboard && pip3 install keras`

## Description:
This is a script that uses Deep Learning Neural Networks (Keras on TensorFlow), specifically an LSTM neural network, to generate protein sequences.

There are two types of sequences that this script generates, a Secondary Structure (SS) sequence, and a FASTA sequence.

Example of SS sequence:

`LLLLLLLSSSLLLLLHHHHHHHHHHHHHLLLLLSSSSSSSHHHHHHHHHHHHHHHHHHLLLLSSSSSSSSSSSSSLLLLLLLLLSSSSSSSSSSSSLLL`

Where L=Loop, S=Strand, and H=Helix

Example of FASTA sequence:

`MACEGAPEVRIGRKPVMNYVLAILTTLMEQGTNQVVVKARGRNINRAVDAVEIVRKRFAKNIEIKDIKIDSQEIEVQTPEGQTRTRRVSSIEICLEKAG`

Some algorithms, such as the Rosetta BluePrintMover that is part of the [Rosetta modeling software](https://www.rosettacommons.org), need the structure's secondary structure topology as input, thus the SS sequence would be benificial to those who want to *De Novo* design a protein given its secondary structures.

On the other hand, the FASTA sequence can be used to generate totally new protein sequences. Whether these sequences fold into a good structure, or fold at all, is a difficult question to answer. But providing this script might be used as a good starting point to build up on as the technology and our understanding of structural biology develops.

## How To Use:
The database SS.cvs and FASTA.csv are already provided, but they can be generated using the Database.py script from the [AIDenovo project](https://github.com/sarisabban/AIDeNovo).

The neural networks are already trained (at accuracy of 85%), the SS neural network weights are in SS.h5 and the FASTA neural network weights are in FASTA.h5 so you will not have to train the neural network yourself, you can just execute the script and get an output immidiatly. You can view the training detailes using the following command:

`tensorboard --logdir=./`

If you want to replicate the work and train the networks yourself, you can just simply run the following command:

`python3 protlstm.py train SS`

`python3 protlstm.py train FASTA`

This script will trian the neural network on the specified dataset (SS or FASTA) and save a new weights file (SS.h5 or FASTA.h5) and TensorBoard log file to view the training results. Currently the accuracy is low (85% for both datasets) even though the network is generating, what seems to be, good enough protein sequences. You are welcome to modify the neural network model and attempt to achieve higher accuracy, I will add you here as a collaborator.

### SS
To generate a secondary structure (SS) sequence simply use the following command:

`python3 protlstm.py generate SS`

### FASTA
To generate a FASTA sequence simply use the following command:

`python3 protlstm.py generate FASTA`

The FASTA sequence preforms additional analysis to give some sort of indication whether the sequence is useful or not. The first analysis is a BLAST run, to see how similar the generated protein sequence is to all other identified protein sequences in the NCBI nr database. The second analysis is a PSIPRED run, where the sequence is analysed for any secondary structures.
