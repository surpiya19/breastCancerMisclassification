# breastCancerMisclassification: Case 2

### Overview

This project analyzes a machine learning model that incorrectly classified a malignant tumor as benign.

In Case 2, Patient X received a benign classification from an AI tumor diagnostic tool and was therefore not referred for a biopsy. Months later, the tumor was diagnosed as malignant. The goal of this analysis is to examine the model's decision using explainable AI methods and determine whether the prediction was consistent with the patterns the model learned from the broader dataset.

The analysis uses two XAI methods, SHAP and LIME, to understand why the model classified Patient X as benign.

### Dataset

The analysis uses the Wisconsin Breast Cancer dataset from `sklearn.datasets`.

The dataset contains measurements describing characteristics of breast cell nuclei. The target variable contains two classes:

- `0`: Malignant
- `1`: Benign

The data was divided into training and testing sets using an 80/20 split with `random_state=42`.

### Model

The assigned model is a Random Forest Classifier.

```python
model_bc = RandomForestClassifier(random_state=42)
model_bc.fit(X_train, y_train)
```

The model achieved the following results on the test set:

| Metric    | Malignant | Benign |
| --------- | --------: | -----: |
| Precision |      0.98 |   0.96 |
| Recall    |      0.93 |   0.99 |
| F1-score  |      0.95 |   0.97 |

Overall accuracy: 96%

The confusion matrix showed:

40 malignant tumors correctly classified as malignant
70 benign tumors correctly classified as benign
3 malignant tumors incorrectly classified as benign
1 benign tumor incorrectly classified as malignant

This means Patient X represents the type of false negative error made by the model.

----

#### Confusion Matrix
The confusion matrix shows the model's performance across the two classes.

| True Classification | Predicted Malignant | Predicted Benign |
| ------------------- | ------------------: | ---------------: |
| Malignant           |                  40 |                3 |
| Benign              |                   1 |               70 |

The model correctly identified most cases in both classes. However, three malignant tumors were classified as benign.
Patient X represents this type of false negative.
This provides important context for the case. The model performed well overall, but it was not perfect and failed to identify every malignant tumor.

----

#### SHAP Analysis

SHAP was used to understand how individual features contributed to Patient X's prediction.

The SHAP explanation showed that the model started with a baseline benign probability of approximately 63.2% and increased the prediction to approximately 85% based on Patient X's feature values.

The strongest features pushing the prediction toward benign were:

| Feature              | Approximate SHAP Contribution |
| -------------------- | ----------------------------: |
| Worst texture        |                         +0.05 |
| Mean concavity       |                         +0.04 |
| Worst area           |                         +0.03 |
| Worst radius         |                         +0.03 |
| Mean perimeter       |                         +0.02 |
| Mean texture         |                         +0.02 |
| Mean area            |                         +0.02 |
| Worst concave points |                         +0.02 |

The SHAP explanation indicates that Patient X's benign prediction was driven by the combined effect of several tumor characteristics.
The strongest positive contributor was worst texture, followed by features such as mean concavity, worst area, and worst radius.
Although some features pushed the prediction toward malignancy, the combined contribution of the other features moved the prediction toward benign.
This suggests that the model based its decision on the overall feature profile rather than on a single measurement.

#### LIME Analysis

LIME was used to provide a local explanation of the model's decision for Patient X.
**The strongest conditions supporting the benign classification were:**

| Feature Condition                   | LIME Weight | Direction |
| ----------------------------------- | ----------: | --------- |
| Worst texture <= 21.05              |     +0.0594 | Benign    |
| Mean texture <= 16.17               |     +0.0309 | Benign    |
| 0.10 < worst concave points <= 0.16 |     +0.0209 | Benign    |
| 0.25 < worst symmetry <= 0.28       |     +0.0201 | Benign    |
| 0.32 < radius error <= 0.47         |     +0.0185 | Benign    |
| 24.72 < area error <= 43.73         |     +0.0146 | Benign    |

**LIME also identified several characteristics that pushed the prediction toward malignancy:**

| Feature Condition              | LIME Weight | Direction |
| ------------------------------ | ----------: | --------- |
| Worst compactness > 0.34       |     -0.0192 | Malignant |
| 0.23 < worst concavity <= 0.39 |     -0.0177 | Malignant |
| 0.06 < mean concavity <= 0.13  |     -0.0164 | Malignant |
| Worst fractal dimension > 0.09 |     -0.0130 | Malignant |

LIME identifies several local feature conditions supporting the benign classification.
The strongest was worst texture <= 21.05, followed by mean texture <= 16.17.
However, LIME also identified characteristics such as higher worst compactness and worst concavity that pushed the prediction toward malignancy.
This shows that Patient X's prediction resulted from competing signals. Some characteristics supported a benign classification while others provided evidence for malignancy.

----

#### Global Importance

The Random Forest's global feature importance can be used to compare Patient X's explanation with the broader patterns learned by the model.

The most important features include characteristics related to:

Tumor area
Concave points
Tumor radius
Concavity
Tumor perimeter

Several of these features also appear in the SHAP and LIME explanations for Patient X.

This suggests that the model's prediction was influenced by features that were important to its broader classification task rather than by an obviously irrelevant feature.

----

## Defense of the models decision

The model's decision to classify Patient X's tumor as benign was not arbitrary.

The SHAP and LIME explanations show that multiple tumor characteristics contributed to the prediction. In particular, texture-related features provided strong evidence in favor of benignity.

The model also performed well across the broader test set, achieving 96% accuracy and correctly classifying 40 of 43 malignant tumors.

The confusion matrix shows that the model did make three false negative errors. Patient X represents one of these types of errors.

From the model's perspective, Patient X's combination of features more closely matched patterns associated with benign tumors in the data it learned from. The model therefore produced an 85% probability of benignity.

However, the later malignant diagnosis demonstrates an important limitation of the model.

The XAI methods explain why the model made the prediction, but they do not demonstrate that the prediction was medically correct.