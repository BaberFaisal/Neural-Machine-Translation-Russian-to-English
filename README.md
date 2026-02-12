# Neural Machine Translation: Russian to English

This repository features an implementation of an Encoder-Decoder (Seq2Seq) architecture for translating Russian text into English. The project focuses on the fundamentals of Neural Machine Translation (NMT), including data preprocessing with subword tokenization and training recurrent neural networks.

##  Project Overview

The core of this project is the `LLM.ipynb` notebook, which guides you through:
- **Data Pipeline**: Loading and cleaning parallel Russian-English corpora.
- **Tokenization**: Implementing subword segmentation (BPE) to handle complex morphology and rare words.
- **Architecture**: Building a Sequence-to-Sequence (Seq2Seq) model using PyTorch.
- **Translation Inference**: Generating translations through an auto-regressive decoding process.

##  Tech Stack
- **Framework**: PyTorch
- **Environment**: Jupyter / Google Colab (GPU accelerated)
- **Techniques**: RNN/GRU/LSTM, Encoder-Decoder, BPE Tokenization.

##  How It Works

1. **Encoder**: A recurrent network reads the Russian source sentence and produces a fixed-size representation of the sentence's meaning.
2. **Decoder**: A second recurrent network takes the representation from the encoder and generates English words sequentially.
3. **Training**: The model is trained to minimize cross-entropy loss between the predicted English words and the actual ground truth.

##  Examples from Training
- **Input (RU)**: этот отель расположен в 10 минутах езды от границы...
- **Target (EN)**: this hotel is located 10 minutes drive from the border...

##  Future Improvements
- Implement **Attention Mechanisms** (Bahdanau or Luong) to improve long-sentence performance.
- Transition from Recurrent cells to a **Transformer-based** architecture.
- Experiment with pre-trained embeddings like FastText or GloVe.

---
*Developed for research and educational purposes in the field of Natural Language Processing.*
