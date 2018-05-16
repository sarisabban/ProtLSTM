# ProtLSTM
An LSTM neural network to generate novel protein sequences

## Requirements:
1. Use the following command (in GNU/Linux) to install all necessary programs and python modules for this script to run successfully:

`sudo apt update && sudo apt full-upgrade && sudo apt install python3-pip python3-pandas python3-numpy python3-tensorflow && pip3 install keras`

## Description:
This is a script that uses Deep Learning Neural Networks, specifically an LSTM neural network, to 

## How To Use:
1. The database SS.cvs and SEQ.csv are already provided, but they can be generated using the Database.py script from the [AIDenovo project](https://github.com/sarisabban/AIDeNovo).
2. A trained model is also provided as model.h5 so you do not have to run the training everytime

3. To generate a protein's secondary structure use the following command:

`python3 lstm.py `

The result would be a string that looks like this:

``

4. To generate a protein sequence use the following command:

`python3 lstm.py `

The result should look like this:

``

But also useful information would be the BLAST and the PSIPRED results, where the BLAST shows how similar the generated sequence is to all knows protein sequences, while the PSIPRED results show its predicted secodary structure.

BLAST result:

``

PSIPRED result:

``
