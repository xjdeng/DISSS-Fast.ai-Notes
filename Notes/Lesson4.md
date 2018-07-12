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

## Batch Size and Back Prop Thru Time (BPTT)

- All movie reviews are concatenated into one block of text
- We then split it into 64 sections, each about 1 million long.
- We then end up with a 1 million x 64 matrix
- We then grab a chunk that's about ~70 x 64 and feed it to the GPU
- Our bptt is 70 but we'll get chunks of approximately 70 on the side by 64. The # changes slightly every time to get slightly different bits of text, kinda like shuffling images in computer vision.
- Can't randomly shuffle words because they need to be in the right order, but we can randomly move their breakpoints

## Create the model

Additional parameters:

```
em_sz = 200  # size of each embedding vector, usually 50 - 600
nh = 500     # number of hidden activations per layer
nl = 3       # number of layers
```
Optimal NLP optimizer parameters:
```
opt_fn = partial(optim.Adam, betas=(0.7, 0.99))
```
- Momentum doesn't work too well with these RNN models, so we use Adam with less momentum than the default 0.9.
- No known way (yet) to find the optimal Dropout parameters. Increase if overfitting, decrease if underfitting.
- Usually don't need to tune alpha, beta, or clip.
- Will have more detailed ways to avoid overfitting in the last lesson.
```
learner = md.get_model(opt_fn, em_sz, nh, nl, dropouti=0.05,
                       dropout=0.05, wdrop=0.1, dropoute=0.02, 
                       dropouth=0.05)
learner.reg_fn = partial(seq2seq_reg, alpha=2, beta=1)
learner.clip=0.3
```

## Fit the model

```
learner.fit(3e-3, 4, wds=1e-6, cycle_len=1, cycle_mult=2)
learner.save_encoder('adam1_enc')
learner.fit(3e-3, 4, wds=1e-6, cycle_len=10, 
            cycle_save_name='adam3_10')
learner.save_encoder('adam3_10_enc')
learner.fit(3e-3, 1, wds=1e-6, cycle_len=20, 
            cycle_save_name='adam3_20')
learner.load_cycle('adam3_20',0)
```
## Testing the model
Now we get to play with the language model we trained.
```
m=learner.model
ss=""". So, it wasn't quite was I was expecting, but I really liked it anyway! The best"""
s = [spacy_tok(ss)]
t=TEXT.numericalize(s)
' '.join(s[0])
". So , it was n't quite was I was expecting , but I really liked it anyway ! The best"
```
Add the methods to test a language model:
```
# Set batch size to 1
m[0].bs=1
# Turn off dropout
m.eval()
# Reset hidden state
m.reset()
# Get predictions from model
res,*_ = m(t)
# Put the batch size back to what it was
m[0].bs=bs
```
Get the next 10 predictions:
```
nexts = torch.topk(res[-1], 10)[1]
[TEXT.vocab.itos[o] for o in to_np(nexts)]
['film',
 'movie',
 'of',
 'thing',
 'part',
 '<unk>',
 'performance',
 'scene',
 ',',
 'actor']
 ```
 Generate text by itself:
 ```
print(ss,"\n")
for i in range(50):
    n=res[-1].topk(2)[1]
    n = n[1] if n.data[0]==0 else n[0]
    print(TEXT.vocab.itos[n.data[0]], end=' ')
    res,*_ = m(n[0].unsqueeze(0))
print('...')
. So, it wasn't quite was I was expecting, but I really liked it anyway! The best 
film ever ! <eos> i saw this movie at the toronto international film festival . i was very impressed . i was very impressed with the acting . i was very impressed with the acting . i was surprised to see that the actors were not in the movie . ...
 ```
