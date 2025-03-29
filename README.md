# Multiple Instance Verification
Official code repository for the paper "Multiple Instance Verification" (http://www.arxiv.org/abs/2407.06544)

We provide the code, and instructions of executing the code, needed to reproduce the main experimental results in our paper

This implementation of "Cross Attention Pooling" (CAP) is based on Tensorflow. A PyTorch implementation can be found here (coming soon)

## Instructions common to all the three tasks 

Go to the "code/" subfolder

### Requirements

To install requirements (python 3.9):

```
pip install -r requirements.txt
```

### Four steps of executing the code for each task

To reproduce the main experimental results of each task, 4 steps are generally required:
- Collecting raw data
- Constructing datasets in MI verification format from the raw data
- Training
- Evaluation

We provide code and/or instructions for each of the 4 steps in each task here.

### Additional code explanations

For each command described below, to get its full list of arguments and the explanations, use '--help', e.g.

```
python XXXXX.py --help
```

For all the tasks, one of the following 'model_id' needs to be supplied when executing the training code:

| Model_id | Description                                                                          |
| -------- | ------------------------------------------------------------------------------------ |
|  0       | Baseline model                                                                       |
|  1       | Gated-attention based MIL (GABMIL) [9]                                               |
|  2       | Cross attention pooling, variance-excited multiplicative attention (CAP-VEMA) [ours] |
|  2.1     | Cross attention pooling, distance-based attention (CAP-DBA) [ours]                   |
|  3       | Pooling by multi-head attention (PMA, i.e. transformer decoder) [13]                 |
|  4       | Multi-head self-attention (MSA, i.e. transformer encoder) [5, 18]                    |



## Task of QMNIST handwriting verification

Let $ROOTQMNIST be the root directory of this task.

### Collecting raw data

There is no need to manually collect QMNIST raw data if using the default value of argument '--download_qmnist=True' when constructing MI verification datasets (see the next step below). Otherwise, obtain the .gz files from https://github.com/facebookresearch/qmnist and put them in the folder:$ROOTQMNIST/QMNIST/raw/.

### Constructing datasets in MI verification format from the raw data

To construct MI verification datasets for 3 rounds of experiments shown in Section 6.1.1, run this command (which outputs train/dev/test datasets as .tsv files for each round of experiment, under $ROOTQMNIST/. Number of rounds is specified by '--number_datasets' in the command):

```
python miv_data_qmnist.py --root_folder=$ROOTQMNIST/ --number_datasets=3
```

The default values of other arguments should suffice to produce datasets similar to those reported in Section E.1.1 of the supplementary material.

Moreover, to reproduce the experimental results in Section 6.1.2, additional training datasets are required, which need to contain the user-specified number of exemplars and the specified average number of instances per target bag (i.e. mean bag-size). For example, to construct one training dataset, output as a .tsv file under $ROOTQMNIST/, with 16000 exemplars and the mean bag-size of 20 instances, run the following command:

```
python miv_data_qmnist_size-experiment.py --root_folder=$ROOTQMNIST/ --number_datasets=1 --number_exemplars=16000 --mean_bagsize=20
```

### Training

To train all the models in the paper for this task, run this command (which requires train and dev datasets, model_id, a directory under which trained model files (.hdf5) are stored)

```
python train_qmnist.py --root_folder=$ROOTQMNIST/ --train=train_qmnist2.tsv --dev=dev_qmnist2.tsv --model_id=2.1 --ckpt_folder=round2ckpt/
```

In the above command, the saved model files can be found under $ROOTQMNIST/round2ckpt/. It is strongly recommended to have a separate '--ckpt_folder' for each round of experiment, because names of the model-files for the same model_id can be duplicate, resulting model-file overwrite. (As an illustration, the model-file name may look like: milmcadse2heads-1se_resnet18tuning-excludeBN_020.hdf5)

### Evaluation

To evaluate a trained model on a test dataset, including both classification and explainability, run this command ($MODELFILE is the model-file name):

```
python eval_qmnist.py --root_folder=$ROOTQMNIST/ --test=test_qmnist2.tsv --model_file=round2ckpt/$MODELFILE
```

The results of evaluation metrics are printed to stdout. It should suffice to reproduce the experimental results of Section 6.1 by running the above training/evaluation steps for all the models across 3 rounds of experiments, and collecting/summarizing results. 



## Task of signature verification against multiple anchors

Let $ROOTSIGVER be the root directory of this task.

### Collecting raw data

The raw data needs to be manually downloaded, upon accepting the data disclaimer, from http://www.iapr-tc11.org/mediawiki/index.php/ICDAR_2011_Signature_Verification_Competition_(SigComp2011).

More precisely, the following two .zip files should be extracted after downloads, with all the extracted folders under the directory of $ROOTSIGVER/$RAW/, where $RAW is the folder under $ROOTSIGVER/ to store images from the raw data:
- http://www.iapr-tc11.org/dataset/ICDAR_SignatureVerification/SigComp2011/sigComp2011-trainingSet.zip
- http://www.iapr-tc11.org/dataset/ICDAR_SignatureVerification/SigComp2011/sigComp2011-test.zip

Consequently, the folder structure to be used for the next step should look like: $ROOTSIGVER/$RAW/Offline Genuine Chinese/, $ROOTSIGVER/$RAW/Offline Genuine Dutch/, etc.

### Constructing datasets in MI verification format from the raw data

To construct MI verification datasets for the 3 rounds of experiments shown in Section 6.2, run this command:

```
python miv_data_sigver.py --root_folder=$ROOTSIGVER/ --image_raw=$RAW --number_datasets=3
```

The above command outputs train/dev/test datasets as .tsv files, for each of the 3 rounds of experiments (specified by '--number_datasets'), under $ROOTSIGVER/. The default values of other arguments should suffice to produce datasets similar to those reported in Section E.1.2 of the supplementary material.

### Training

To train all the models in the paper for this task, run this command (which requires train and dev datasets, model_id, a directory under which trained model files (.hdf5) are stored)

```
python train_sigver.py --root_folder=$ROOTSIGVER/ --train=train_sigver2.tsv --dev=dev_sigver2.tsv --model_id=2 --ckpt_folder=round2ckpt/
```

After running the above command, the saved model files can be found under $ROOTSIGVER/round2ckpt/. (As an illustration, the model-file name may look like: milmcase6heads-1se_EfficientNetV2B3tuning-excludeBN_010.hdf5)

### Evaluation

To evaluate a trained model on a test dataset, including both classification and explainability, run this command ($MODELFILE is the model-file name):

```
python eval_sigver.py --root_folder=$ROOTSIGVER/ --test=test_sigver2.tsv --model_file=round2ckpt/$MODELFILE
```

The results of evaluation metrics are printed to stdout. It should suffice to reproduce the experimental results of Section 6.2 by running the above training/evaluation steps for all the models across 3 rounds of experiments, and collecting/summarizing results. 



## Task of fact extraction and verification (FEVER)

Let $ROOTFEVER be the root directory of this task.

### Collecting raw data

The raw data needs to be manually downloaded from:
- https://fever.ai/download/fever/train.jsonl
- https://fever.ai/download/fever/paper_dev.jsonl
- https://fever.ai/download/fever/paper_test.jsonl
- https://fever.ai/download/fever/wiki-pages.zip

After downloads and extraction, put the resultant files/folders under the directory of $ROOTFEVER/$RAW/, where $RAW is the folder under $ROOTFEVER/ to store the raw data.

### Constructing datasets in MI verification format from the raw data

To construct MI verification datasets for the 3 rounds of experiments shown in Section 6.3, run this command:

```
python miv_data_fever.py --root_folder=$ROOTFEVER/ --image_raw=$RAW --number_datasets=3
```

The above command outputs 3 train datasets as .npy files, for 3 rounds of experiments (specified by '--number_datasets'), under $ROOTFEVER/. Dev/test datasets (also .npy files under $ROOTFEVER/) are the same across different rounds of experiments. The default values of other arguments should suffice to produce datasets similar to those reported in Section E.1.3 of the supplementary material.
Note: 	The first-time running of this command needs to use the default value of argument "--preprocess=True", which is one-off but requires long time (to preprocess the data from the Wikipedia dump).
	After the first-time running, if more train datasets are needed, run the command using "--preprocess=False". 

### Training

To train all the models in the paper for this task, run this command (which requires train and dev datasets, model_id, a directory under which trained model files (.hdf5) are stored)

```
python train_fever.py --root_folder=$ROOTFEVER/ --train=train_fever1.npy --dev=dev_fever_all.npy --model_id=2 --ckpt_folder=round1ckpt/
```

After running the above command, the saved model files can be found under $ROOTFEVER/round1ckpt/. (As an illustration, the model-file name may look like: milmcase4heads-1se_SBERTmulti-qa-MiniLM-L6-cos-v1clspool_trainingNone_015.hdf5)

### Evaluation

To evaluate a trained model on the test dataset, including both classification and explainability, run this command ($MODELFILE is the model-file name):

```
python eval_fever.py --root_folder=$ROOTFEVER/ --test=test_fever_all.npy --model_file=round1ckpt/$MODELFILE
```

The results of evaluation metrics are printed to stdout. It should suffice to reproduce the experimental results of Section 6.3 by running the above training/evaluation steps for all the models across 3 rounds of experiments, and collecting/summarizing results.

