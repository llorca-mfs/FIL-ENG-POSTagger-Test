# Filipino POS
This repository implements a simple LSTM-based part-of-speech tagger for Filipino, written for demo purposes. Pretrained weights and a prediction script are included, which you may use for tagging. A mode for training is also included to train a simple LSTM tagger using your own tagset and data.

# Requirements
* PyTorch v1.x
* NVIDIA GPU (For training. Not needed for inference/prediction.)

# Tagging a Sentence
To tag a sentence, you need a trained tagger. 

We provide a pretrained tagger checkpoint that you can use in the folder `checkpoint`, which includes two files: `model.bin` (the saved PyTorch weights), and `settings.bin` (contains vocabularies and model settings). Use the `main.py` script to tag a sentence as follows, making sure your example sentence is space-splittable:

```
python filipino-pos/main.py \
    --do_predict \
    --checkpoint checkpoint \
    --sentence 'ginagamit ang matematika sa agham .'
```
This should give you the following output:

```
['VBTR', 'DTC', 'NNC', 'CCT', 'NNC', 'PMP']
```

# Training a Tagger
To train a tagger, you will need a tagset. The script expects you to have four files:
* `train_tags.txt` - the POS tags for the training data, space-splittable, one sentence per line.
* `train_text.txt` - the sentences for the training data, space-splittable, one sentence per line.
* `eval_tags.txt` - POS tags for validation/evaluation.
* `eval_text.txt` - sentences for validation/evaluation.

Since the code is implemented for demo purposes only, we do not include tokenzation/preprocessing beyond normalization of quotation marks and space-split tokenization. The training script assumes that your data can be split via spaces already. Make sure to preprocess your data before using the training script.

You can train a tagger using the following command:

```
python filipino-pos/main.py \
    --do_train \
    --checkpoint checkpoint \
    --train_data data/train_text.txt \
    --train_tags data/train_tags.txt \
    --evaluation_data data/eval_text.txt \
    --evaluation_tags data/eval_tags.txt \
    --embedding_dim 100 \
    --bidirectional \
    --num_layers 2 \
    --recur_dropout 0.1 \
    --hidden_dim 128 \
    --dropout 0.5 \
    --min_freq 2 \
    --msl 128 \
    --bs 128 \
    --learning_rate 3e-4 \
    --weight_decay 0.0 \
    --epochs 5 \
    --overwrite_save_directory
```

This will save `model.bin` and `settings.bin` in a folder named `checkpoint/` that you can use to tag sample sentences.

# Tagset Information
The pretrained weights included in this repository are trained on a dataset with tags that follow the MGNN Tagset [(Nocon and Borra, 2016)](https://www.aclweb.org/anthology/Y16-3010.pdf). A full description of their tagset can be found in [this document](https://drive.google.com/file/d/0B7lapk7DR3X4cHF5M3gxM1pCdGs/view). Refer to their paper for more information on the tags outputted by the pretrained model.

For training your own POS Tagger, you can adapt any tagset that you want.

# Changelogs and To-Do
- [x] Implement a standard BiLSTM Tagger
- [x] Provide an open-source trained tagger
- [ ] Implement an LSTM-CRF Tagger (Viterbi Decoding)

# Contributing
If you spot any bugs or would like to add things to this demo repository, feel free to file an issue on the Issues tab!

*This repository is managed by the DLSU Machine Learning Group*


