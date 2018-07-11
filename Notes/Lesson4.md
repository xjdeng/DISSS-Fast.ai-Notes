# Lesson 4: Natural Language Processing

[[Lecture 4 Video](http://course.fast.ai/lessons/lesson4.html)] [[IPython Notebook](https://github.com/fastai/fastai/blob/master/courses/dl1/lesson4-imdb.ipynb)]

## Initial Setup

First, import these:
```
from fastai.learner import *
import torchtext
from torchtext import vocab, data
from torchtext.datasets import language_modeling
from fastai.rnn_reg import *
from fastai.rnn_train import *
from fastai.nlp import *
from fastai.lm_rnn import *
import dill as pickle
```
Note: torchtext is PyTorch's NLP library

Download the [IMDB Large Movie Review Dataset](http://ai.stanford.edu/~amaas/data/sentiment/)

And set up the following:

```
PATH = 'data/aclImdb/'
TRN_PATH = 'train/all/'
VAL_PATH = 'test/all/'
TRN = f'{PATH}{TRN_PATH}'
VAL = f'{PATH}{VAL_PATH}'
```
(Note: no separate test and validation sets here.)

Check # of words in dataset:

```
!find {TRN} -name '*.txt' | xargs cat | wc -w
17486581
!find {VAL} -name '*.txt' | xargs cat | wc -w
5686719
```

Tokenizers recognize pieces in your sentence.  We'll be using spacy_tok.

## Creating a Field

```
TEXT = data.Field(lower=True, tokenize=spacy_tok)
bs=64; bptt=70
FILES = dict(train=TRN_PATH, validation=VAL_PATH, test=VAL_PATH)
md = LanguageModelData.from_text_files(PATH, TEXT, **FILES, bs=bs, 
                                       bptt=bptt, min_freq=10)
```
**PATH**: location to store data, save models, etc.

**\*\*FILES**: list of all files.

**bs**: batch size

**bptt**: Back Prop Through Time. Max length of a sentence on GPU.

**min_freq=10**: denote words that occur less than 10 times as "unknown".

A note on the ``TEXT.vocab`` field: it stores unique words (aka tokens) and maps them to a unique integer id.
```
# 'itos': 'int-to-string' 
TEXT.vocab.itos[:12]
['<unk>', '<pad>', 'the', ',', '.', 'and', 'a', 'of', 'to', 'is', 'it', 'in']
# 'stoi': 'string to int'
TEXT.vocab.stoi['the']
2
```
