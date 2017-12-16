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

## 1. Experiment Ridge Rounded
#### Experiment Conclusion
We modeled the experiment as a regression task with a Ridge regressor, using the features Sex (One Hot Encoding), Diameter, and Whole Weight targeting the Rings amount. **The result is far from accurate enought** to preserve the infant ones, with a mean error of 2.5 using the rounded prediction we have an error of about 3.75 years.