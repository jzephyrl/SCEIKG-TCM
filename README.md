# SCEIKG
1. Setup the environment

   - Download the repository

     `cd  SCEIKG-TCM`

   - Create a conda environment: 

     - This is PyTorch implementation for the paper

     - The code has been tested running under Python 3.7.13. 

       `conda env create -n sceikg_tcm_env python=3.7.13`

     - The required packages are as follows:

       - `pandas==1.3.4`
       - `python == 3.7.13`
       - `scikit-learn==1.0.1`
       - `scipy == 1.7.3`
       - `tqdm == 4.64.1`
       - `torch  == 1.8.1`
       - `numpy == 1.21.6`
       -`pickle == 0.7.5`

2. Data Example 

   Due to the privacy of the ZzzTCM data, we have provided samples and descriptions of the data here

   - train_kg_data.txt contains the ternary of the TCM knowledge graph, with the first column being the head entity, the     second column being the relationship, and the third column being the tail entity

   - IKG.txt is the interaction knowledge graph constructed by the TCM knowledge graph as well as ZzzTCM, in which the content is a ternary, the first column is the head entity, the second column is the relationship, and the third column is the tail entity

   - train_kg_data.txt on the other hand is a mapping of the triples in IKG.txt, where the first column is the head entity, the second column is the relationship, and the third column is the tail entity.

   - records.txt contains the patient number and the patient's serial visit records, which are self-constructed and do not involve disclosure of the patient's privacy.

   - symptoms.txt is based on the patient's serial visit records and use the API of ChatGPT to get the patient's symptoms, in addition to the use of some rules to process the data, the content of the patient's number and the patient's serial visit symptoms

     `python API.py`

   - herbs.txt is the prescription corresponding to the serial visit records, here the prescription is self-constructed and do not involve disclosure of the patient's privacy.

3. Pre-trained model

   We have provided the best model `model_epoch_best.pth` for SCEIKG, along with `Train_Test_log`, which  can be downloading from the [link](链接：https://pan.baidu.com/s/1cun5DB1vlErQeY9yWr_3bA?pwd=1234), and put it in the trained_model folder. 

4. Run the Code

   `python main.py`

