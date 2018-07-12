# Lesson 5: Collaborative Filtering

[[Lecture 4 Video](http://course.fast.ai/lessons/lesson4.html)] [[Lesson 5 Video](http://course.fast.ai/lessons/lesson5.html)] [[IPython Notebook](https://github.com/fastai/fastai/blob/master/courses/dl1/lesson5-movielens.ipynb)]

## Initial Setup

```
from fastai.learner import *
from fastai.column_data import *
```
[Download Dataset](http://files.grouplens.org/datasets/movielens/ml-latest-small.zip)

```
path='data/ml-latest-small/'
ratings = pd.read_csv(path+'ratings.csv')
```

Goal: predict user-movie combinaion.  But first we need to predict a rating.

```
movies = pd.read_csv(path+'movies.csv')
```
Create a subset for Excel:
```
g=ratings.groupby('userId')['rating'].count()
topUsers=g.sort_values(ascending=False)[:15]
g=ratings.groupby('movieId')['rating'].count()
topMovies=g.sort_values(ascending=False)[:15]
top_r = ratings.join(topUsers, rsuffix='_r', how='inner', 
                     on='userId')
top_r = top_r.join(topMovies, rsuffix='_r', how='inner', 
                   on='movieId')
pd.crosstab(top_r.userId, top_r.movieId, top_r.rating, 
            aggfunc=np.sum)
```
See this [Excel file](https://github.com/fastai/fastai/blob/master/courses/dl1/excel/collab_filter.xlsx) which uses matrix factorization/decomposition instead of Deep Learning.

For graphics of the Excel walkthrough, [see here](https://medium.com/@hiromi_suenaga/deep-learning-2-part-1-lesson-5-dd904506bee8).

## Simple Python version 
