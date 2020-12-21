# Bayesian Regression

A project for the course CS-E5710 Bayesian Data Analysis at Aalto University.

This project is about Bayesian workflow on predicting electrical output of a combined cycle power plant with the ambient conditions of the plant. The data was retrieved from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Combined+Cycle+Power+Plant) (Tüfekci, 2014).

Two Bayesian models were implemented: a linear model and a generalized additive model. It was discovered that the latter model is more suitable to the data than the former, indicated by the better performance in posterior predictive checks and approximate leave-one-out cross-validation. The results obtained from these Bayesian models reflected our uncertainty of the posterior distribution and the estimates of parameters and quantities of interest.

Contributors: Khang Nguyen, Son Le

## References

Tüfekci, P. (2014). Prediction of full load electrical power output of a base load operated combined cycle power plant using machine learning methods. International Journal of Electrical Power & Energy Systems, 60, 126-140.
