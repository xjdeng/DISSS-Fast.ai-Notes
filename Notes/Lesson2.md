# Lesson 2: Multilabel Classification

[[Lecture 3 Video](http://course.fast.ai/lessons/lesson3.html)] [[IPython Notebook](https://github.com/fastai/fastai/blob/master/courses/dl1/lesson2-image_models.ipynb)]

This one mainly builds on the [first lesson](https://github.com/xjdeng/DISSS-Fast.ai-Notes/blob/master/Notes/Lesson1.md) (single label classification).

## Required Imports

```
from fastai.imports import *
from fastai.torch_imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *
```

## Label Format
[[Lesson 3 @ 1:23:12](https://youtu.be/9C06ZPF8Uuc?t=1h23m12s)]

Each image can have multiple labels, so we can't use the Keras style of putting each image in a folder that's named its label.  Instead, we're given a [CSV table](https://www.kaggle.com/c/6322/download/train_v2.csv.zip) like this:


| image_name | tags                                      |
|------------|-------------------------------------------|
| train_0    | haze primary                              |
| train_1    | agriculture clear primary water           |
| train_2    | clear primary                             |
| train_3    | clear primary                             |
| train_4    | agriculture clear habitation primary road |
| train_5    | haze primary water                        |

### Corresponding Code:

```
from planet import f2

metrics=[f2]
f_model = resnet34

label_csvlabel_cs  = f'{PATH}train_v2.csv'
n = len(list(open(label_csv)))-1
val_idxs = get_cv_idxs(n)

def get_data(sz):
    tfms = tfms_from_model(f_model, sz, aug_tfms=transforms_top_down, max_zoom=1.05)
    return ImageClassifierData.from_csv(PATH, 'train-jpg', label_csv, tfms=tfms,
                    suffix='.jpg', val_idxs=val_idxs, test_name='test-jpg')
```

### Dataset vs Dataloader
Example:
```
x,y = next(iter(data.val_dl))
```
See Pytorch's [Datasets and Dataloaders](https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/pytorch/dataloader-and-datasets.html)

Basically, Datasets have individual images and Dataloaders are a minibatch.

## Resizing Images

[[Lesson 3 @ 1:39:55](https://youtu.be/9C06ZPF8Uuc?t=1h39m55s)]

```data = data.resize(int(sz*1.3), 'tmp')```

See the video, it saves time if you have really big images
