4a. Based on accuracy which model is the best one?
-------
It looks like LDA and QDA perform the exact same, so they are probably the best models for this application.
Although given how close the 3 top performers are it may be best to do more analysis (increase sample size).
Even upon changing the seed, I get different results, but anecdotally (3 other runs) QDA seems to be the best
then kNN then LDA. QDA being the best matches intuivitively that iris edge boundaries may not necessarily be linear.

4b. For each of the 6 other models, explain why you think it does not perform
as well as the best one.
-------

kNN: performs well because it can learn complex decision boundaries not necessarily quadratic or linear.

LDA: performs well because iris classes are relatively seperated, but may lack nuance since edge boundaries aren't
linear. so does well but not as well as QDA.

Naive Bayes: makes assumptions about feature indepence, which may not neccesarily be true

Linear Regression: does remarkably well, because the iris classes ended up being relatively linear from
feature to classes. may fail in more complex scenarios, larger sample size, or more classes and features.

Polynomial Regression (3rd then 2nd): these are the worst for the same reasons, a regression fit is simply
not the correct choice for this sort of classification task, the degree is simply too low in both cases to
model the decision boundaries for the iris dataset
