# DAG Classifier for the NCAA® March Madness ML Competition

This is exploratory work for the [Google Cloud & NCAA® ML Competition 2019](https://www.kaggle.com/c/mens-machine-learning-competition-2019) on Kaggle. The goal is to predict the outcome of games in the playoffs of the US national college basketball competition. Detailed data is available for previous games.

The winning solutions have been using gradient boosting methods with some lucky manual guesses for specific games for years. I was interested in the competition, because I thought the dataset would be useful to test out an idea I had for a while to perform DAG classification with a sequence model.

The games of the tournament can be represented as a regular DAG with the teams as edges and the games as nodes. Graph neural networks are limited to processing graphs with dozens of nodes, but we have thousands of games. A topological sort of the DAG can be processed by a sequence model thus allowing us to handle thousands of games.

The model didn't end up producing competitive results.