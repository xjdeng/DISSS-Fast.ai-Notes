# Lesson 4: Natural Language Processing

[[Lecture 4 Video](http://course.fast.ai/lessons/lesson4.html)] [[IPython Notebook](https://github.com/fastai/fastai/blob/master/courses/dl1/lesson4-imdb.ipynb)]

[Another article on Structured Deep Learning](https://towardsdatascience.com/structured-deep-learning-b8ca4138b848)

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
## Sentiment
Goal: use pre-trained language model and fine-tune it for sentiment classification
```
IMDB_LABEL = data.Field(sequential=False)
```
The sequential=False says the text field should be tokenized.  Also, treat each review separately and not the input as one big piece of text.
```
splits = torchtext.datasets.IMDB.splits(TEXT, IMDB_LABEL, 'data/')
t = splits[0].examples[0]
t.label, ' '.join(t.text[:16])
('pos', 'ashanti is a very 70s sort of film ( 1979 , to be precise ) .')
```
- **splits()**: torchtext method that creates train, text, and validation sets.
- Look at [lang_model-arxiv.ipynb](https://github.com/fastai/fastai/blob/master/courses/dl1/lang_model-arxiv.ipynb) on how to define fastai/torchtext datasets.

Create a ModelData object from torchtext splits:

```
md2 = TextData.from_splits(PATH, splits, bs)
m3 = md2.get_model(opt_fn, 1500, bptt, emb_sz=em_sz, n_hid=nh, 
                   n_layers=nl, dropout=0.1, dropouti=0.4,
                   wdrop=0.5, dropoute=0.05, dropouth=0.3)
m3.reg_fn = partial(seq2seq_reg, alpha=2, beta=1)
m3.load_encoder(f'adam3_20_enc')
```
Use differential learning rates and increase max gradient for clipping.
```
m3.clip=25.
lrs=np.array([1e-4,1e-3,1e-2])
m3.freeze_to(-1)
m3.fit(lrs/2, 1, metrics=[accuracy])
m3.unfreeze() #Make sure the last layer is frozen
m3.fit(lrs, 1, metrics=[accuracy], cycle_len=1)
[ 0.       0.45074  0.28424  0.88458]
[ 0.       0.29202  0.19023  0.92768]
m3.fit(lrs, 7, metrics=[accuracy], cycle_len=2, 
       cycle_save_name='imdb2')
[ 0.       0.29053  0.18292  0.93241]                        
[ 1.       0.24058  0.18233  0.93313]                        
[ 2.       0.24244  0.17261  0.93714]                        
[ 3.       0.21166  0.17143  0.93866]                        
[ 4.       0.2062   0.17143  0.94042]                        
[ 5.       0.18951  0.16591  0.94083]                        
[ 6.       0.20527  0.16631  0.9393 ]                        
[ 7.       0.17372  0.16162  0.94159]                        
[ 8.       0.17434  0.17213  0.94063]                        
[ 9.       0.16285  0.16073  0.94311]                        
[ 10.        0.16327   0.17851   0.93998]                    
[ 11.        0.15795   0.16042   0.94267]                    
[ 12.        0.1602    0.16015   0.94199]                    
[ 13.        0.15503   0.1624    0.94171]
m3.load_cycle('imdb2', 4)
accuracy(*m3.predict_with_targs())
0.94310897435897434
```
See Part 2 for how to improve this further!
