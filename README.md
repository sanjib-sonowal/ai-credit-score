# Credit Score using Machine Learning
Calculate Credit Score using below machine learning algorithm

### Logistic Regression
Logistic regression is a statistical technique used in machine learning for binary categorization. Logistic regression is mostly utilized for classification jobs rather than regression, despite its name.

Purpose: To predict the probability of an instance belonging to a particular class (binary classification).

Output: Produces probabilities between 0 and 1 using the logistic function (sigmoid), and a threshold is applied to make the final classification.

Model: Linear combination of input features, transformed by the logistic function.

Loss Function: Typically uses the logistic loss or cross-entropy loss.

Training: Optimized using iterative methods like gradient descent to minimize the loss function.

Use Cases: Widely used in areas like medical diagnosis, spam filtering, and credit scoring.

### Decision Tree
In machine learning, a decision tree is a predictive model that maps features (attributes) to conclusions (class labels) in the form of a tree-like structure. It's a versatile algorithm used for both classification and regression tasks.

Purpose: To make decisions by recursively splitting the data based on the most significant features.

Structure: Tree-like model with nodes representing decisions, branches representing possible outcomes, and leaves representing final class labels or numerical values.

Decision Process: At each node, the algorithm selects the feature that best splits the data, based on criteria like information gain or Gini impurity.

Training: The tree is built recursively by repeatedly choosing the best feature to split the data until a stopping condition is met (depth limit, minimum samples per leaf, etc.).

Prediction: For a new instance, it traverses the tree from the root to a leaf, assigning the class label or numerical value of that leaf.

Interpretability: Decision trees are easy to understand and interpret, making them suitable for explaining model decisions.

Use Cases: Commonly used in various domains, including finance, healthcare, and customer relationship management.

### Random Forest
In machine learning, a random forest is an ensemble learning method that combines the predictions of multiple decision trees to improve overall performance and robustness.

Ensemble Method: It builds multiple decision trees during training and merges their predictions during testing.

Bootstrap Aggregating (Bagging): Random forest uses bagging, which involves training each tree on a random subset of the training data with replacement.

Feature Randomness: At each split in a tree, only a random subset of features is considered. This adds an additional layer of randomness.

Voting Mechanism: For classification, it aggregates the predictions through voting, and for regression, it averages the predictions.

Reduced Overfitting: Random forests are less prone to overfitting compared to individual decision trees.

Parallelization: Training individual trees can be done in parallel, making random forests suitable for parallel and distributed computing.

Versatility: Effective for a wide range of tasks, including classification, regression, and feature importance analysis.

### APIs
Use the below api to train the model.

[POST] /train

Use the below api to test the model

[POST] /predict 

Payload:
{
    "algo": "logistic-regression"
}

Other supported algorithms: "decision-tree" and "random-forest"
