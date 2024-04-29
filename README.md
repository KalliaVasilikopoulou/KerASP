Neurasp Model Training Project

Neurasp Model Training Project attempts to create a new neurasp 'solver', that is, basically, a keras model extension of a given classifier, and train the whole model based on new labels, more complex than the initial ones.
The idea is that, if it is difficult for a deep learning model to be trained on a task that demands both perception and reasoning to be solved, we can split this task (and, also the structure of the model) into two parts: one that learns to classify objects based on perception and one that calculates the probability of the output (output is also a class, but a class that corresponds to the outputs of the task) based on reasoning.
The perception part of the keras model is a simple deep learning keras classifier given by the user.
The reasoning part of the keras model is constructed based on the answer sets of the corresponding asp program and the application of some probability theory.


This program creates:
 - a dataset specifically for the training of the neurasp model and saves it in the train_data folder
 - a default neurasp model, trains it and saves it in the solver folder (it also saves the classifier of the neurasp model in a separate classifier folder)
 - tokenizes for the dataset, if the user needs them, and saves them in the train_data folder



What to do:

Insert your files in the user_scripts folder!
To insert the proper files, read the instructions.txt file and follow the structure of the example files that are located in the file_examples folder. You will find the instructions.txt file and the file_examples folder in the user_scripts folder.




