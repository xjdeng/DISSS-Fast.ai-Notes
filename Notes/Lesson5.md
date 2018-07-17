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

First, create a validation set:

```
val_idxs = get_cv_idxs(len(ratings)) 
wd = 2e-4 
n_factors = 50
```
Create a model from a CSV file:
```
cf = CollabFilterDataset.from_csv(path, 'ratings.csv', 'userId', 'movieId', 'rating')
```
Get a learner and fit the model:
```
learn = cf.get_learner(n_factors, val_idxs, 64, opt_fn=optim.Adam)
learn.fit(1e-2, 2, wds=wd, cycle_len=1, cycle_mult=2)
```
Benchmark by taking the square root of the final validation error:
```
sqrt(0.765)
```
How to get your predictions:
```
preds = learn.predict()
```
How to plot using seaborn sns (built on matplotlib):
```
y = learn.data.val_y
sns.jointplot(preds, y, kind='hex', stat_func=None)
```
## Python Dot Products

```
a = T([[1., 2], [3, 4]])
b = T([[2., 2], [10, 10]])
(a*b).sum(1)
6
70
[torch.FloatTensor of size 2]
```
See above: the dot product is done element wise (1*2 + 2*2 = 6 and 3\*10 + 4 \*10 = 70)

## Build a Dot Product Module

```
class DotProduct (nn.Module):
   def forward(self, u, m): return (u*m).sum(1)
   
model = DotProduct()
model(a,b)
6
70
[torch.FloatTensor of size 2]
```
No need to call model.forward(a,b) !!!

## More complex module

First, we need to set up some user and movie indices which we'll later look up:
```
u_uniq = ratings.userId.unique() 
user2idx = {o:i for i,o in enumerate(u_uniq)} 
ratings.userId = ratings.userId.apply(lambda x: user2idx[x])  

m_uniq = ratings.movieId.unique() 
movie2idx = {o:i for i,o in enumerate(m_uniq)} 
ratings.movieId = ratings.movieId.apply(lambda x: movie2idx[x])  

n_users=int(ratings.userId.nunique()) 
n_movies=int(ratings.movieId.nunique())
```
Note: ```{o:i for i,o in enumerate(u_uniq)}``` reverses the indices: makes a dict where you input a key and get the corresponding index!
```
class EmbeddingDot(nn.Module):
    def __init__(self, n_users, n_movies):
        super().__init__()
        self.u = nn.Embedding(n_users, n_factors)
        self.m = nn.Embedding(n_movies, n_factors)
        self.u.weight.data.uniform_(0,0.05)
        self.m.weight.data.uniform_(0,0.05)
        
    def forward(self, cats, conts):
        users,movies = cats[:,0],cats[:,1]
        u,m = self.u(users),self.m(movies)
        return (u*m).sum(1)
```
[Kaiming Initialization](http://www.jefkine.com/deep/2016/08/08/initialization-of-deep-networks-case-of-rectifiers/) : Initializing weights to random numbers between 0 and 0.05.

Note: we don't want to manually loop through mini-batches of users and movies or else we won't get GPU acceleration (see lines 3 and 4 of ```forward()``` above.

```
x = ratings.drop(['rating', 'timestamp'],axis=1)
y = ratings['rating'].astype(np.float32)
data = ColumnarModelData.from_data_frame(path, val_idxs, x, y, ['userId', 'movieId'], 64)

wd=1e-5
model = EmbeddingDot(n_users, n_movies).cuda()
opt = optim.SGD(model.parameters(), 1e-1, weight_decay=wd, momentum=0.9)
```
**optim** : gets us Pytorch parameters.

**model.parameters()** : gives us the weights to be updated/learned.

Now fit the model.  This is more like the standard PyTorch approach without SGD with restarts or differential learning rate.
```
fit(model, data, 3, opt, F.mse_loss)
```
