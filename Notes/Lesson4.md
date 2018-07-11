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

