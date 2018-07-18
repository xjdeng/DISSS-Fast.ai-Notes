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

## Improving the model even futher

We add bias which adjusts for popular movies and enthusiastic users

```
min_rating,max_rating = ratings.rating.min(),ratings.rating.max()
min_rating,max_rating
def get_emb(ni,nf):
    e = nn.Embedding(ni, nf)
    e.weight.data.uniform_(-0.01,0.01)
    return e
class EmbeddingDotBias(nn.Module):
    def __init__(self, n_users, n_movies):
        super().__init__()
        (self.u, self.m, self.ub, self.mb) = [get_emb(*o) for o in [
            (n_users, n_factors), (n_movies, n_factors), (n_users,1), (n_movies,1)
        ]]
        
    def forward(self, cats, conts):
        users,movies = cats[:,0],cats[:,1]
        um = (self.u(users)* self.m(movies)).sum(1)
        res = um + self.ub(users).squeeze() + self.mb(movies).squeeze()
        res = F.sigmoid(res) * (max_rating-min_rating) + min_rating
        return res
```
**F**: Pytorch functional, usually imported as **F**

**squeeze**: PyTorch's version of [numpy's broadcasting](https://docs.scipy.org/doc/numpy-1.13.0/user/basics.broadcasting.html)

If we want to squeeze ratings between 1 and 5, we put them through the sigmoid function then multiply it by 4 and add 1.

Fit the model:

```
wd=2e-4
model = EmbeddingDotBias(cf.n_users, cf.n_items).cuda()
opt = optim.SGD(model.parameters(), 1e-1, weight_decay=wd, momentum=0.9)
fit(model, data, 3, opt, F.mse_loss)
[ 0.       0.85056  0.83742]                                     
[ 1.       0.79628  0.81775]                                     
[ 2.       0.8012   0.80994]
```

## Neural Network Version

Concat user and movie embedding vectors and feet them through a neural net with one hidden layer.
```
class EmbeddingNet(nn.Module):
    def __init__(self, n_users, n_movies, nh=10, p1=0.5, p2=0.5):
        super().__init__()
        (self.u, self.m) = [get_emb(*o) for o in [
            (n_users, n_factors), (n_movies, n_factors)]]
        self.lin1 = nn.Linear(n_factors*2, nh)
        self.lin2 = nn.Linear(nh, 1)
        self.drop1 = nn.Dropout(p1)
        self.drop2 = nn.Dropout(p2)
        
    def forward(self, cats, conts):
        users,movies = cats[:,0],cats[:,1]
        x = self.drop1(torch.cat([self.u(users),self.m(movies)], dim=1))
        x = self.drop2(F.relu(self.lin1(x)))
        return F.sigmoid(self.lin2(x)) * (max_rating-min_rating+1) + min_rating-0.5
        
wd=1e-5
model = EmbeddingNet(n_users, n_movies).cuda()
opt = optim.Adam(model.parameters(), 1e-3, weight_decay=wd)
fit(model, data, 3, opt, F.mse_loss)
A Jupyter Widget
[ 0.       0.88043  0.82363]                                    
[ 1.       0.8941   0.81264]                                    
[ 2.       0.86179  0.80706]
```
## The Training Loop

Recall:
```
opt = optim.SGD(model.parameters(), 1e-1, weight_decay=wd, momentum=0.9)
```
Also see [gradient descent Excel sheet](https://github.com/fastai/fastai/blob/master/courses/dl1/excel/graddesc.xlsm)

Use [Finite Differences](https://en.wikipedia.org/wiki/Finite_difference) to calculate the gradient when doing gradient descent.

## Backpropagation
