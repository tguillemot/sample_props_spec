This is a specification to introduce data information (as `sample_weights`)
during the computation of an estimator methods(fit, score, ...) based on the
different discussion proposes on issues and PR :

- [Initial PR to talk about sample_props #4497](
    https://github.com/scikit-learn/scikit-learn/issues/4497)
- [Acceptance of sample_weights in pipeline.score #7723](
    https://github.com/scikit-learn/scikit-learn/pull/7723)
- [Establish global error state like np.seterr #4660](
    https://github.com/scikit-learn/scikit-learn/issues/4660)
- [Should cross-validation scoring take sample-weights into account? #4632](
    https://github.com/scikit-learn/scikit-learn/issues/4632)
- [Sample properties #4696](
    https://github.com/scikit-learn/scikit-learn/issues/4696)

Probably related PR:
- [Add feature_extraction.ColumnTransformer #3886](
    https://github.com/scikit-learn/scikit-learn/pull/3886)
- [Categorical split for decision tree #3346](
    https://github.com/scikit-learn/scikit-learn/pull/3346)

# 1. Requirement

These requirements are defined from the different issues and PR discussions:

- User can attach information to samples.
- Must be a DataFrame like object.
- Can be given to `fit`, `score`, `split` and every time user give X.
- Must work with every meta-estimator (`Pipeline, GridSearchCV,
    cross_val_score`).
- Can specify what sample property is used by each part of the meta-estimator.
- Must raise an error if not necessary extra information are given to an
    estimator. In the case of meta-estimator these errors are not raised.

Requirement proposed but not used by this specification:
- User can attach feature properties to samples.

# 2. Definition

Some estimator in sklearn can change their behavior when an attribute
`sample_props` is provided. `sample_props` is a dictionary
(`pandas.DataFrame` compatible) defining sample properties. The example bellow
explain how a `sample_props` can be provided to LogisticRegression to
weighted the samples:  

```python
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

digits = datasets.load_digits()
X = digits.data
y = digits.target

# Define weights used by sample_props
weights_fit = np.random.rand(X.shape[0])
weights_fit /= np.sum(weights_fit)
weights_score = np.random.rand(X.shape[0])
weights_score /= np.sum(weights_score)

logreg = LogisticRegression()

# Apply and score a LogisticRegression to the data without sample weights
logreg = logreg.fit(X, y)
score = logreg.score(X, y)
print("Score obtained without applying sample_weights: %f" % score)

# Apply LogisticRegression without sample weights and score with sample weights
logreg = logreg.fit(X, y)
score = logreg.score(X, y, sample_props={'sample_weight': weights_score})
print("Score obtained by applying sample_weights only to score: %f" % score)

# Apply and score a LogisticRegression with sample weights
log_reg = logreg.fit(X, y, sample_props={'sample_weight': weights_fit})
score = logreg.score(X, y, sample_props={'sample_weight': weights_score})
print("Score obtained by applying sample_weights to both"
      " score and fit: %f" % score)
```

When an estimator expects a mandatory `sample_props`, an error is raised for
each property not provided. Moreover **if an unintended properties is given
through `sample_props`, a warning will be launched** to prevent that the result
may be different from the one expected. For example, the following code :

```python
import numpy as np
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline

digits = datasets.load_digits()
X = digits.data
y = digits.target

logreg = LogisticRegression()

# This instruction will raise the warning
logreg = logreg.fit(X, y, sample_props={'bad_property':})
```

will **raise the warning message**: "sample_props['bad_property'] is not used by
`LogisticRegression.fit`. The results obtained may be different from the one
expected."

We provide the function `sklearn.seterr` in the case you want to change the
behavior of theses messages. Even if there are considered as warnings by
default, we recommend to change the behavior to raise as errors. You can do it
by adding the following code:

```python
sklearn.seterr(sample_props="raise")
```

Please refer to the documentation of `np.seterr` for more information.

# 3. Behavior of `sample_props` for meta-estimator

# 3.1 Common routing scheme

Meta-estimators can also change their behavior when an attribute `sample_props`
is provided. On that case, `sample_props` will be sent to any internal estimator
and function supporting the `sample_props` attribute. In other terms **all the
property defined by `sample_props` will be transmitted to each internal
functions or classes supporting `sample_props`**. For example in the following
example, the property `sample_weights` is sent through `sample_props` to
`pca.fit_transform` and `logistic.fit`:

```python
import numpy as np
from sklearn import decomposition, datasets, ensemble, linear_model, svm
from sklearn.pipeline import Pipeline

digits = datasets.load_digits()
X = digits.data
y = digits.target

logistic = linear_model.LogisticRegression()
pca = decomposition.PCA()
pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic),])

# Define weights
weights = np.random.rand(X.shape[0])
weights /= np.sum(weights)

# sample_weights is send to pca.fit_transform and logistic.fit
pipe.fit(X, sample_props={"sample_weights": weights})
```

**By contrast with the estimator, no warning will be raised by a
meta-estimator if an extra property is sent through `sample_props`.**
Anyway, errors are still raised if a mandatory property is not provided.

## 3.1 Override common routing scheme

**You can override the common routing scheme of `sample_props` by defining your
own routes through the `routing` attribute of a meta-estimator**.

**A route defines a way overrides the value of a key of `sample_props` by the
value of another key in the same `sample_props`. This modification is done
every time a method compatible with `sample_prop` is called.**

To illustrate how it works, let's consider that you want to replace the value of
`sample_weights` by the value of `another_name`, you must define the route:

```python
import numpy as np
from sklearn import decomposition, datasets, ensemble, linear_model, svm
from sklearn.pipeline import Pipeline

digits = datasets.load_digits()
X = digits.data
y = digits.target

logistic = linear_model.LogisticRegression()
pca = decomposition.PCA()

# Create a route using routing
pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic),],
                routing={'sample_weights': 'another_name'})

# Define weights
weights = np.random.rand(X.shape[0])
weights /= np.sum(weights)

# sample_props will be modified before being passed to pca.fit_transform and
# logistic.fit
pipe.fit(X, sample_props={"another_name": weights})
```

Here before calling `pca.fit_transform` or `logistic.fir`, **`sample_props` will
be copied and modified to add a value `weights` to the key `sample_weights`**.
If `sample_weights` already exists when a route is applied, its value
`sample_props['sample_weights'] will be overridden. On all cases, `another_name`
will not be modified by the route.

**The way we have defined the routes before allows to create general routes but
we provide also a way to limit the use of a route to a specific method or
estimator**.

To define a route for a specific method/estimator, you just need to add its
name:

```python
routing={'pca':{
             'sample_weights': 'pca_weights'
             },
        }
```

Here, the value of `pca_weights` overrides the `sample_props[sample_weights]`
only when `sample_props` is given to `pca`. For all other cases, `sample_props`
is sent without being modified.

**It can be useful to discard the use of a property for a particular
method/estimator. For these cases, you must define a route associating a
key to `None`** :

```python
routing={'pca':{
             'sample_weights': None
             },
        }
```

Thus, the previous route will put `sample_props['sample_weights'] = None`.

**Overriding the routing scheme can be hard at the beginning and you must
remember the priority of application of each route types**:

1. Routes applied specifically to a function/estimator
2. Routes defined globally
3. Routes defined by default

Let's consider the following code to familiarized yourself with the different
routes definitions :

```python
import numpy as np
from sklearn import datasets
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV

digits = datasets.load_digits()
X = digits.data
y = digits.target

# Define the groups used by cross_val_score
groups = np.random.randint(3, size=y.shape)

# Define the groups used by GridSearchCV
gs_groups = np.random.randint(3, size=y.shape)

# Define weights used by cross_val_score
weights = np.random.rand(X.shape[0])
weights /= np.sum(weights)

# We define GridSearchCV with its own route
grid = GridSearchCV(SGDClassifier(), params, cv=LeaveOneLabelOut(),
                    routing={'sample_groups': None,
                             'split': {'sample_groups': 'gridsearch_groups',
                                       'sample_weights': None}})

# Define the route use by cross_val_score
cross_val_score(grid, X, y, cv=LeaveOneLabelOut(),
                routing={'score': {'sample_groups': None}
                         'split': {'sample_weights': None}},
                sample_props={'grid_search_groups': gs_groups,
                              'sample_groups': groups,
                              'sample_weights': weights})
```

With this code, the `sample_props` sent to each function of GridSearchCV and
cross_val_score will be:

| function              | `sample_props`                                                                          |
|-----------------------|-----------------------------------------------------------------------------------------|
| grid.fit              | `{'grid_search_groups': gs_groups, 'sample_groups': None, 'sample_weights': weights}`  |
| grid.score            | `{'grid_search_groups': gs_groups, 'sample_groups': None, 'sample_weights': weights}`  |
| grid.split            | `{'grid_search_groups': gs_groups, 'sample_groups': gs_groups, 'sample_weights': None}`|
| cross_val_score.fit   | `{'grid_search_groups': gs_groups, 'sample_groups': groups, 'sample_weights': weights}`|
| cross_val_score.score | `{'grid_search_groups': gs_groups, 'sample_groups': None, 'sample_weights': weights}`  |
| cross_val_score.split | `{'grid_search_groups': gs_groups, 'sample_groups': groups, 'sample_weights': None}`   |
