# Lesson 3: Structured and Time Series Data

[[Lecture 3 Video](http://course.fast.ai/lessons/lesson3.html)] [[Lecture 4 Video](http://course.fast.ai/lessons/lesson4.html)] [[IPython Notebook](https://github.com/fastai/fastai/blob/master/courses/dl1/lesson3-rossman.ipynb)]

Jeremy skipped over Feature Engineering in this lesson.  To learn more, look at his first [Machine Learning video](https://www.youtube.com/watch?v=CzdWqFTmn0Y&feature=youtu.be).

## Dropout

First, look at a sample learner:

```
Sequential(
  (0): BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True)
  (1): Dropout(p=0.5)
  (2): Linear(in_features=1024, out_features=512)
  (3): ReLU()
  (4): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True)
  (5): Dropout(p=0.5)
  (6): Linear(in_features=512, out_features=120)
  (7): LogSoftmax()
)
```

Dropout(p=x) means randomly throwing away a fraction x of the activations in a layer.  Dropout(p=0.5) means throwing away 0.5 or 50% of the activations.

This is a great way to mitigate overfitting and is loosely inspired by the way the brain works!

Why are validation losses sometimes better than training losses, esp early on?  Because dropout is turned OFF when making a prediction on the validation set (we want to use the best model we can for that.)

Use the ``ps=x`` parameter to set default dropouts for all the layers, i.e. ps=0.5.  For example:

```
learn = ConvLearner.pretrained(arch, data, ps=0.5, precompute=True)
```

But what happens when you set ps=0?

- Massive overfitting likely after even a few epochs (depending on the dataset or model structure.)
- Dropout layers won't even be aded to the model:

```
Sequential(
  (0): BatchNorm1d(4096, eps=1e-05, momentum=0.1, affine=True)
  (1): Linear(in_features=4096, out_features=512)
  (2): ReLU()
  (3): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True)
  (4): Linear(in_features=512, out_features=120)
  (5): LogSoftmax()
)
```
Can also set different dropout for different layers, i.e. ps=[0., 0.2] :
```
learn = ConvLearner.pretrained(arch, data, ps=[0., 0.2],
            precompute=True, xtra_fc=[512]); learn
```
```
Sequential(
  (0): BatchNorm1d(4096, eps=1e-05, momentum=0.1, affine=True)
  (1): Linear(in_features=4096, out_features=512)
  (2): ReLU()
  (3): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True)
  (4): Dropout(p=0.2)
  (5): Linear(in_features=512, out_features=120)
  (6): LogSoftmax()
)
```
When in doubt, use the same dropout every layer (i.e. ps = 0.25)

### xtra_fc parameter

This adds additional hiden layers.

With xtra_fc=[]:

```
learn = ConvLearner.pretrained(arch, data, ps=0., precompute=True, 
            xtra_fc=[]); learn 
```
```
Sequential(
  (0): BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True)
  (1): Linear(in_features=1024, out_features=120)
  (2): LogSoftmax()
)
```

with xtra_fc=[700, 300] :

```
learn = ConvLearner.pretrained(arch, data, ps=0., precompute=True, 
            xtra_fc=[700, 300]); learn
```
```
Sequential(
  (0): BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True)
  (1): Linear(in_features=1024, out_features=700)
  (2): ReLU()
  (3): BatchNorm1d(700, eps=1e-05, momentum=0.1, affine=True)
  (4): Linear(in_features=700, out_features=300)
  (5): ReLU()
  (6): BatchNorm1d(300, eps=1e-05, momentum=0.1, affine=True)
  (7): Linear(in_features=300, out_features=120)
  (8): LogSoftmax()
)
```

