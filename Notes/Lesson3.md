# Lesson 3: Structured and Time Series Data

[[Lecture 3 Video](http://course.fast.ai/lessons/lesson3.html)] [[Lecture 4 Video](http://course.fast.ai/lessons/lesson4.html)] [[IPython Notebook](https://github.com/fastai/fastai/blob/master/courses/dl1/lesson3-rossman.ipynb)]

Jeremy skipped over Feature Engineering in this lesson.  To learn more, look at his first [Machine Learning video](https://www.youtube.com/watch?v=CzdWqFTmn0Y&feature=youtu.be).

Also see the Fast.ai [Structured Learner solution to the Titanic challenge](https://github.com/dtylor/dtylor.github.io/blob/master/kaggle/titanic/titanic_nn.ipynb).

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

## Structured and Time Series Data

2 Types of variables:
- Categorical: can take on distinct values.  Examples are Store Types, Store id's, States, etc.
- Continuous: represented by floating point numbers. Example: temperature, price, humidity.
- Depending on the situation, some variables like Year, Month, etc. can be treated as either continuous or categorical.
- Floating point numbers like price are difficult to turn into categorical variable but integers like year can be treated as either.


### Specifying the Model

```
md = ColumnarModelData.from_data_frame(PATH, val_idx, df, 
         yl.astype(np.float32), cat_flds=cat_vars, bs=128, 
         test_df=df_test)
```
**val_idx**: specifies the validation indices.  Example:
```
val_idx = np.flatnonzero((df.index<=datetime.datetime(2014,9,17)) &
              (df.index>=datetime.datetime(2014,8,1)))
```
**df**: the independent variable (the X).  See below for how to generate this using ``proc_df()``

**yl.astype(np.float32)**: Here we enter the dependent variable (the Y).  Note: yl is the log transformation of the original "Y", and astype(np.float32) transforms it to float32.

**cat_flds**: specifies the categorical field names in df, your independent variable.  Example:
```
  cat_flds = ['Store', 'DayOfWeek', 'Year', 'Month', 'Day',
            'StateHoliday', 'CompetitionMonthsOpen', 'Promo2Weeks',
            'StoreType', 'Assortment', 'PromoInterval', 
            'CompetitionOpenSinceYear', 'Promo2SinceYear', 'State',
            'Week', 'Events', 'Promo_fw', 'Promo_bw', 
            'StateHoliday_fw', 'StateHoliday_bw', 
            'SchoolHoliday_fw', 'SchoolHoliday_bw']
```
**bs**: batch size

**df_test**: also generated using ```proc_df()```

#### proc_df(): Process Data Frame

- Pulls out dependent variable
- do_scale: Scales the data
- Handles missing values
  - Continuous: replaces with median
  - Categorical: "missing" becomes new category

Example 1:
```
df, y, nas, mapper = proc_df(joined_samp, 'Sales', do_scale=True)
yl = np.log(y)
```
Example 2:
```
df_test, _, nas, mapper = proc_df(joined_test, 'Sales', do_scale=True, skip_flds=['Id'],
                                  mapper=mapper, na_dict=nas)
```
Setting the validation set:
```
val_idx = np.flatnonzero((df.index<=datetime.datetime(2014,9,17)) &
              (df.index>=datetime.datetime(2014,8,1)))
```
Note: [np.flatnonzero()](https://docs.scipy.org/doc/numpy/reference/generated/numpy.flatnonzero.html) documentation

### Getting the Learner

```
m = md.get_learner(emb_szs, len(df.columns)-len(cat_vars),
                   0.04, 1, [1000,500], [0.001,0.01], 
                   y_range=y_range)
```
- **emb_szs**: embedding sizes (will go over later)
- 0.04: the amount of dropout we're using
- [1000, 500]: # of activations in each layer
- [0.001, 0.01]: amount of dropout at later layers
- **y_range**: defined later (in the error function section)

### Embedding Sizes
First, we need to get the sizes (cardinalities) of the categorical variables. Add 1 to account for the possibility of missing data:
```
cat_sz = [(c, len(joined_samp[c].cat.categories)+1) 
             for c in cat_vars]
```
```
cat_sz[0:5]

[('Store', 1116),
 ('DayOfWeek', 8),
 ('Year', 4),
 ('Month', 13),
 ('Day', 32)]
```
Rule of thumb for embedding size: min(cardinality/2, 50)
```
emb_szs = [(c, min(50, (c+1)//2)) for _,c in cat_sz]
emb_szs[0:5]
[(1116, 50),
 (8, 4),
 (4, 2),
 (13, 7),
 (32, 16)]
 ```

### Defining an error function

Example: Root Mean Square Percentage Error (RMSPE):

```
def inv_y(a): return np.exp(a)
def exp_rmspe(y_pred, targ):
    targ = inv_y(targ)
    pct_var = (targ - inv_y(y_pred))/targ
    return math.sqrt((pct_var**2).mean())
max_log_y = np.max(yl)
y_range = (0, max_log_y*1.2)
```
As mentioned previously, taking the log is more numerically stable and ends up being more accurate.

### Fitting the model

```
m.fit(lr, 3, metrics=[exp_rmspe])
```
Notice that the error metric is specified in metrics=[....]

Example Result:
```
[ 0.       0.02479  0.02205  0.19309]                          
[ 1.       0.02044  0.01751  0.18301]                          
[ 2.       0.01598  0.01571  0.17248]
```

### Summary of Steps

1. Create separate lists for categorical and continuous variables. Make sure they're column names in a Pandas dataframe.
2. Create a list of row indices you want in your validation set.
3. Call this line of code:
```
md = ColumnarModelData.from_data_frame(PATH, val_idx, df, 
         yl.astype(np.float32), cat_flds=cat_vars, bs=128, 
         test_df=df_test)
```
4. Create a list of tuples of how big you want each of your embeddings to be: (# of categories , # of embeddings)
5. Call get_learner.  Sample parameters:
```
m = md.get_learner(emb_szs, len(df.columns)-len(cat_vars), 0.04, 1,
                   [1000,500], [0.001,0.01], y_range=y_range)
```
6. Call ```m.fit(lr, 3, metrics=[exp_rmspe])``` (or with your parameters)
