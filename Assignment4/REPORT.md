### Answers to the following questions for CompareFeatureSelectionMethods:

**a. Based on accuracy, which dimensionality reduction method worked the best?**

The **Genetic Algorithm** (Part 4) worked the best, achieving the highest accuracy of **0.9667**.

---

**b. For each of the two other methods, explain why you think they did not perform as well as the best one.**

* **PCA**: PCA transforms the original features into principal components that may lose some discriminative power, especially if only one principal component is used (as in Part 2). This transformation could have reduced the ability of the decision tree to classify the data accurately.
* **Simulated Annealing**: Although Simulated Annealing found a feature set with similar accuracy (0.9533), it is a heuristic optimization technique and might not explore the search space as efficiently as the Genetic Algorithm. It may have converged prematurely or missed more optimal feature combinations.

---

**c. Did the best dimensionality reduction method produce better accuracy than using no dimensionality reduction (Part 1)? Explain possible reasons why it did or did not.**

Yes, the Genetic Algorithm in Part 4 produced a slightly better accuracy (0.9667) than using no dimensionality reduction (0.9533). The Genetic Algorithm likely found a subset of features that better suited the Decision Tree model by balancing between the original features and the principal components from PCA, which could have captured useful variance in the data.

---

**d. Did Part 2 produce the same set of best features as Part 3? Explain possible reasons why it did or did not.**

No, Part 2 (PCA) used only the first principal component ( `z1` ), while Part 3 (Simulated Annealing) used a combination of original features ( `petal length` , `petal width` ) and transformed features ( `z1` , `z3` ). The difference arises because PCA focuses purely on variance maximization, while Simulated Annealing tries to optimize for accuracy based on feature subsets, considering both original and transformed features.

---

**e. Did Part 2 produce the same set of best features as Part 4? Explain possible reasons why it did or did not.**

No, Part 2 used only the first principal component ( `z1` ), whereas Part 4 consistently selected a combination of original and transformed features, particularly focusing on both original features ( `sepal length` , `petal width` ) and transformed features ( `z1` , `z2` , `z3` ). The Genetic Algorithm's search process was more thorough in finding the best combination of features for classification, while PCA's focus was limited to variance.

---

**f. Did Part 3 produce the same set of best features as Part 4? Explain possible reasons why it did or did not.**

No, Part 3 (Simulated Annealing) used a combination of `petal length` , `petal width` , `z1` , and `z3` , while Part 4 (Genetic Algorithm) generally converged to using `z1` , `z2` , `z3` , and some original features like `sepal length` or `petal width` . The Genetic Algorithm was likely able to explore the feature space more effectively over multiple generations, while Simulated Annealing may have gotten stuck in a local optimum.
