# Lesson 1 Notes: Basic Image Recognition
[[Lecture Video 1](http://course.fast.ai/lessons/lesson1.html)] [[Lecture Video 2](http://course.fast.ai/lessons/lesson2.html)][[IPython Notebook](https://github.com/fastai/fastai/blob/master/courses/dl1/lesson1.ipynb)]  

## Required Fast.ai headers

```
from fastai.imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *
```

## A Basic Image Classifier
[[Video (Lesson 1 @ 20:28)](https://youtu.be/IPBSB1HLNLo?t=20m28s)]
```
arch=resnet34
data = ImageClassifierData.from_paths(PATH, tfms=tfms_from_model(arch, sz))
learn = ConvLearner.pretrained(arch, data, precompute=True)
learn.fit(0.01, 2)
```

### What these parameters mean:

- **arch**: the ImageNET architecture, i.e. Resnet34
- **PATH**: the directory containing the images.
    - Example: ```PATH = "data/dogscats"```
    - Needs to have *train* and *valid* subdirectories.
    - Each subdirectory needs a folder holding images of a respective class named that class.
    - Example: ```data/dogscats/train/cat``` and ```data/dogscats/valid/dog```
- **sz**: size that the images should be resized to.  For now, set ```sz = 224```.
- **Learning Rate**: set to 0.01 here. Will go over how to find an optimal rate later. See section below
- **Epochs**: # of times to train the model, set to 2 here. Too many epochs leads to overfitting and will take a long time. See: [[Video (Lesson 1 @ 1:18:46)](https://youtu.be/IPBSB1HLNLo?t=1h18m46s)]

### Image Classifier Output:

```
epoch      trn_loss   val_loss   accuracy                                                                              
    0      0.042222   0.028351   0.991211  
    1      0.035367   0.026421   0.991211  
```

### Make Predictions:
[[Video (Lesson 1 @ 24:46)](https://youtu.be/IPBSB1HLNLo?t=24m46s)] Note: learn.predict() only returns the logs of the predictions on the **validation** set.
```
log_preds = learn.predict()
preds = np.argmax(log_preds, axis=1)  # from log probabilities to 0 or 1
probs = np.exp(log_preds[:,1])        # pr(dog)
```
&nbsp;
&nbsp;
## Finding a Learning Rate
[[Video (Lesson 1 @ 1:11:57)](https://youtu.be/IPBSB1HLNLo?t=1h11m57s)]

Basic method (first run this): ```learn.lr_find()```  
Graph of Learning Rate vs # of Iterations: ```learn.sched.plot_lr()```[[Video (Lesson 2 @ 9:17)](https://youtu.be/JNxcznsrRb8?t=9m17s)]   
- Note: pick a learning rate slightly to the left of the lowest point.  

Graph of Learning Rate vs Loss: ```learn.sched.plot()```  

## Data Augmentation
[[Video (Lesson 2 @ 15:49)](https://youtu.be/JNxcznsrRb8?t=15m49s)]  

Increase the size of your dataset by flipping, rotating, zooming, etc. your images

```
tfms = tfms_from_model(resnet34, sz, aug_tfms=transforms_side_on, max_zoom=1.1)
data = ImageClassifierData.from_paths(PATH, tfms=tfms)
learn = ConvLearner.pretrained(arch, data, precompute=True) #Need to set it false later
learn.fit(1e-2, 1) #Learning rate and epochs can change if you like
```

Note: transforms_on_side only flips the images left or right
- Good for cats/dogs
- Don't do this for, say, letters, since their meanings will change when flipped

## Pretrained Layers (precomute = true)

[[Video (Lesson 2 @ 25:50)](https://youtu.be/JNxcznsrRb8?t=25m50s)]  

- By default when we create a learner, it sets all but the last layer to frozen. That means that it's still only updating the weights in the last layer when we call fit.
- Need to set ```learn.precompute=False``` later for data augmentation to work

```
learn.fit(1e-2, 1)
learn.precompute=False
learn.fit(1e-2, 3, cycle_len=1)
```


## Cycle Length

[[Video (Lesson 2 @ 30:18)](https://youtu.be/JNxcznsrRb8?t=30m18s)]  
- What is that cycle_len parameter? What we've done here is used a technique called stochastic gradient descent with restarts (SGDR), a variant of learning rate annealing, which gradually decreases the learning rate as training progresses. This is helpful because as we get closer to the optimal weights, we want to take smaller steps.
- However, we may find ourselves in a part of the weight space that isn't very resilient - that is, small changes to the weights may result in big changes to the loss. We want to encourage our model to find parts of the weight space that are both accurate and stable. Therefore, from time to time we increase the learning rate (this is the 'restarts' in 'SGDR'), which will force the model to jump to a different part of the weight space if the current area is "spikey".
- The number of epochs between resetting the learning rate is set by cycle_len, and the number of times this happens is refered to as the number of cycles, and is what we're actually passing as the 2nd parameter to fit().

## Fine Tuning

- By default, all layers except the final one are frozn.  To unfreeze frozen layers, use ```learn.unfreeze()```
- Differential learning rates
    - ```lr=np.array([1e-4,1e-3,1e-2])```
    - Generally speaking, the earlier layers (as we've seen) have more general-purpose features. Therefore we would expect them to need less fine-tuning for new datasets. For this reason we will use different learning rates for different layers: the first few layers will be at 1e-4, the middle layers at 1e-3, and our FC layers we'll leave at 1e-2 as before. We refer to this as differential learning rates.
- ```learn.fit(lr, 3, cycle_len=1, cycle_mult=2)```: Actually, this is 3\*cycle_len\*cycle_mult epochs.

# Summary: How to create a world-class image classifier:

[[Video (Lesson 2 @ 1:13:53)](https://youtu.be/JNxcznsrRb8?t=1h13m53s)]  
- Enable data augmentation, and precompute=True
- Use lr_find() to find highest learning rate where loss is still clearly improving
- Train last layer from precomputed activations for 1-2 epochs
- Train last layer with data augmentation (i.e. precompute=False) for 2-3 epochs with cycle_len=1
- Unfreeze all layers
- Set earlier layers to 3x-10x lower learning rate than next higher layer
- Use lr_find() again
- Train full network with cycle_mult=2 until over-fitting
