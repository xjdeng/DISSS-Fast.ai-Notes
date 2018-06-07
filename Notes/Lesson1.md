# Lesson 1 Notes: Dogs and Cats

[[Lecture Video](https://www.youtube.com/watch?v=IPBSB1HLNLo)] [[IPython Notebook](https://github.com/fastai/fastai/blob/master/courses/dl1/lesson1.ipynb)]

## A Basic Image Classifier

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
- **Learning Rate**: set to 0.01 here. Will go over how to find an optimal rate later.
- **Epochs**: # of times to train the model, set to 2 here.
