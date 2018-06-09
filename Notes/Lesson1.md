# Lesson 1 Notes: Basic Image Recognition

[[Lecture Video 1](http://course.fast.ai/lessons/lesson1.html)] [[Lecture Video 2](http://course.fast.ai/lessons/lesson2.html)][[IPython Notebook](https://github.com/fastai/fastai/blob/master/courses/dl1/lesson1.ipynb)]

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
Graph of Learning Rate vs # of Iterations: ```learn.sched.plot_lr()```  
Graph of Learning Rate vs Loss: ```learn.sched.plot()```  
