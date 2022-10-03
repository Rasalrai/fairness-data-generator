# fairness-measures
The main goal of this project was to visualize the behaviour of the group fairness measures in different imbalanced settings.
Two important factors were taken into account: Minority Ratio and Group Ratio.

- Minority Ratio = minority_class / (minority_class + majority_class)
- Group Ratio = minority_group / (minority_group + majority group)

For example: positive & negative classes - we can calculate the minority ratio, females & males - we can calculate the group ratio.

Moreover, many interesting measures were proposed in the literature. In this implementation only several of them were included:
- Equal Opportunity Difference
- Statistical Parity
- Accuracy Equality Difference
- Predictive Equality Difference
- Positive Predictive Parity Difference
- Negative Predictive Parity Difference

Some additional measures were also calculated but not used in the visualization part of the project.

This project consists of two main files:
- sets_creation.py: generates error matrixes with all possible scenarios and saves them to txt/bin (we need this to visualize measures aggregated by different imbalance values)
- fairness_plots.ipynb : reads the files, creates pandas dataframe and visualizes the data.