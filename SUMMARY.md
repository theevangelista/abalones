## Problem definition
  * Informal Description: We want to predict the age of an abalone by the number of rings of it.
  * Formal Description:
    * Task(T): Predict the age of an abalone
    * Experience(E): Metrics of opened abalones and manually count of its rings
    * Performance(P): RMSE of the number of rings
  
  * Assumptions:
    * The size of an abalone increases as it get older
    * The size thus the weight of its shell increses over time

## Why solve this problem ?
For study and conservation propurses the age of an ablone is important to know so people
can take action to preserve them, but to know it, the abalone needs to be opened and have
its rings counted, which is a boring and slow process that can damage the abalone, losing it.
An automated calculation based on it's metrics can be faster and more secure to the animal.

The model will be aplicable to the location where the data was collected so it does not introduce
a location bias that might come from the way the abalones develop in the different environments

# Experiments

## Experiment Ridge Rounded

```python
import os
os.chdir('..')
```


```python
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, train_test_split
from common.eval import cross_val_rmse

np.random.seed = 70
```


```python
data = pd.read_csv('./data/dropped_correlatated.csv')

X = data.drop('rings', axis=1)
y = data['rings']
```


```python
grid = {
    'alpha': [0.01, 0.05, 0.1, 0.5, 1, 2, 3],
    'normalize': [True, False],
    'solver': ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
    }

search = GridSearchCV(Ridge(), grid, cv=10)
search.fit(X, y)
best_params = search.best_params_
print("Best parameters found: ", best_params)
```

    Best parameters found:  {'alpha': 0.5, 'normalize': True, 'solver': 'saga'}
    


```python
model = Ridge(**best_params)
scores = cross_val_rmse(model, X, y, cv=10)
print(f"Mean score {np.mean(scores)}")
```

    Mean score 2.5794337761826247
    


```python
model = Ridge(**best_params)
X_train, X_test, y_train, y_test = train_test_split(X, y)
model.fit(X_train, y_train)

```




    Ridge(alpha=0.5, copy_X=True, fit_intercept=True, max_iter=None,
       normalize=True, random_state=None, solver='saga', tol=0.001)




```python
predicts = np.around(model.predict(X_test))
results = pd.DataFrame({'true': y_test, 'predicted': predicts})
results.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>predicted</th>
      <th>true</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2603</th>
      <td>12.0</td>
      <td>9</td>
    </tr>
    <tr>
      <th>42</th>
      <td>7.0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1040</th>
      <td>12.0</td>
      <td>11</td>
    </tr>
    <tr>
      <th>1582</th>
      <td>9.0</td>
      <td>9</td>
    </tr>
    <tr>
      <th>4107</th>
      <td>8.0</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>




```python
def diff_error(y, yhat):
    "An error calculated by the difference of the rounded predicted age to the actual value"
    diff = y - yhat
    return np.mean(np.sqrt(np.square(diff)))
```


```python
msg = f"""
The mean difference from the original target rings,
using the rounded transformation of the predicted
value is: {diff_error(y_test, predicts)}

"""

print(msg)
```

    
    The mean difference from the original target rings,
    using the rounded transformation of the predicted
    value is: 1.8976076555023924
    
    
    

### Experiment Conclusion

Using the features Sex (One Hot Encoding), Diameter, and Whole Weight targeting the Rings amount.

-----------