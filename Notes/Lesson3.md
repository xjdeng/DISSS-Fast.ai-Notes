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

Dropout(p=x) means randomly throwing away a fraction x of the activations in a layer.


