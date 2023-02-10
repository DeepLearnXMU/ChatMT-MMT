# Flat-NCT+MMT
Our code is basically based on the publicly available toolkit: THUMT-Tensorflow[1] (python version 3.6). Please refer to it in Github for the required dependency. (Just seach it on Github.)

The following steps are training our model and then test its performance in terms of BLEU, TER, and Sentence Similarity.

# Data Preprocessing
Please refer to the "data_preprocess_code" file.

Take En->De as an example
# Training

Our work involves three-stage training
## The first stage
1) bash train_ende_base_step1.sh # set the training_step=200,000; Suppose the generated checkpoint file is located in path1

## The second stage (i.e., fine-tuning on the general translation data and monolingual chat translation data)
2) bash train_ende_base_step2.sh # Here, set the training_step=1; Suppose the generated checkpoint file is located in path2
3) python thumt-step1/thumt/scripts/combine_add.py --model path2 --part path1 --output path3  # copy the weight of the first stage to the second stage.
4) bash train_ende_base_step2.sh # Here, set the --output=path3 and the training_step=205,000; Suppose the generated checkpoint file is path4


## The third stage (i.e., fine-tuning on the monolingual chat translation data and target chat translation data)
5) bash train_ende_base_step3_bi.sh # Here, set the training_step=1; Suppose the generated checkpoint file is located in path5
6) python thumt-step1/thumt/scripts/combine_add.py --model path4 --part path5 --output path6  # copy the weight of the first stage to the second stage.
7) bash train_ende_base_step3_bi.sh # Here, set the --output=path6 and the training_step=210,000; Suppose the generated checkpoint file is path7


# Test by multi-blue.perl
8) bash test_ende_base.sh # set the checkpoint file path to path7 in this script. # Suppose the predicted file is located in path8 at checkpoint step xxxxx

# Test by SacreBLEU and TER
Required TER: v0.7.25; Sacre-BLEU: version.1.4.13 (BLEU+case.mixed+numrefs.1+smooth.exp+tok.13a+version.1.4.13)

9) python SacreBLEU_TER_Coherence_Evaluation_code/cal_bleu_ter4ende.py # Please correctly set the golden file and predicted file in this file and in sacrebleu_ende.py, respectively.


# Coherence Evaluation by Sentence Similarity
Required: gensim; MosesTokenizer

10) python SacreBLEU_TER_Coherence_Evaluation_code/train_word2vec.py # firstly downloading the corpus in [2] and then training the word2vec.
11) python SacreBLEU_TER_Coherence_Evaluation_code/eval_coherence.py # putting the file containing three precoding utterances and the predicted file in corresponding location and then running it.


# Citation
@article{DBLP:journals/corr/abs-2301-11749,
  author    = {Chulun Zhou and
               Yunlong Liang and
               Fandong Meng and
               Jie Zhou and
               Jinan Xu and
               Hongji Wang and
               Min Zhang and
               Jinsong Su},
  title     = {A Multi-task Multi-stage Transitional Training Framework for Neural
               Chat Translation},
  journal   = {CoRR},
  volume    = {abs/2301.11749},
  year      = {2023},
  url       = {https://doi.org/10.48550/arXiv.2301.11749},
  doi       = {10.48550/arXiv.2301.11749},
  eprinttype = {arXiv},
  eprint    = {2301.11749},
  timestamp = {Tue, 31 Jan 2023 16:32:09 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2301-11749.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}


# Reference
[1] Zhixing Tan, Jiacheng Zhang, Xuancheng Huang, Gang Chen, Shuo Wang, Maosong Sun, Huanbo Luan, and Yang Liu. 2020. THUMT: An open-source toolkit for neural machine translation. In Proceedings of AMTA, pages 116–122.
[2] Bill Byrne, Karthik Krishnamoorthi, ChinnadhuraiSankar, Arvind Neelakantan, Ben Goodrich, DanielDuckworth, Semih Yavuz, Amit Dubey, KyuYoungKim, and Andy Cedilnik. 2019. Taskmaster-1: Toward a realistic and diverse dialog dataset. In Proceedings of EMNLP-IJCNLP, pages 4516–4525.
