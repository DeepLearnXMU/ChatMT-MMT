# BConTrast comes from [1].

This file contains the train, dev, and test sets of the BConTrasT[1] corpus used in the chat translation task for WMT20.

It is based on the Taskmaster-1 corpus[2] which includes monolingual (i.e. English) task-based dialogs in six domains, 
  i.e. (i) ordering pizza, (ii) creating auto repair appointments, (iii) setting up ride service, (iv) ordering movie tickets, (v) ordering coffee drinks, and (vi) making restaurant reservations.
  A subset of Taskmaster-1 corpus[2] was selected and translated into German at Unbabel.
  
Each conversation in the data file has the following structure:
* ConversationID: A unique identifier for each conversation.
* Utterances: An array of utterances that make up the conversation. Each utterance has the following fields:
  - UtteranceID: A 0-based index indicating the order of the utterances in the conversation.
  - Speaker: Either customer or agent, indicating which role generated this utterance.
  - Source: The utterance in the original source language.
  - Target: The utterance in the translated target language.
    

**Note:** Since here we assume customer and agent speak in their own language, the source and target text might be in English or German depending on the role.


# Reference
[1] M. Amin Farajian, Ant ́onio V. Lopes, Andr ́e F. T. Martins, Sameen Maruf, and Gholamreza Haffari. 2020. Findings of the WMT 2020 shared task on chat translation. In Proceedings of WMT, pages 65–75.
[2] Bill Byrne, Karthik Krishnamoorthi, ChinnadhuraiSankar, Arvind Neelakantan, Ben Goodrich, DanielDuckworth, Semih Yavuz, Amit Dubey, KyuYoungKim, and Andy Cedilnik. 2019. Taskmaster-1: Toward a realistic and diverse dialog dataset. In Proceedings of EMNLP-IJCNLP, pages 4516–4525.
