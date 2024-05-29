!!! NEW VERSION !!!
The new version can handle multiple classifiers that accept different inputs and return different distributions (each classifier has a different set of classes where it assigns probabilities). The solver calculates the probabilities of the combinations of all the different classifiers' classes and, after that, it calcuclates and returns the probability distribution of the solver's output classes.


KerASP Model Training Project

KerASP Model Training Project attempts to create a new KerASP 'solver', that is, basically, a keras model extension of a given classifier, and train the whole model based on new labels, more complex than the initial ones.
The idea is that, if it is difficult for a deep learning model to be trained on a task that demands both perception and reasoning to be solved, we can split this task (and, also the structure of the model) into two parts: one that learns to classify objects based on perception and one that calculates the probability of the final outputs based on reasoning (the outputs of the reasoning part are also classes, just like the outputs of the perception part; to calculate the output probabilities of the classes of the reasoning part, we use the output probabilities of the classes of the perception part).
The perception part of the keras model is a simple deep learning keras classifier given by the user.
The reasoning part of the keras model is constructed based on the answer sets of the corresponding asp program and the application of some probability theory.


This program creates:
 - a dataset specifically for the training of the KerASP model and saves it in the train_data folder
 - a default KerASP model, trains it and saves it in the saved_models folder (it also saves the classifier of the KerASP model in the same folder)
 - tokenizers for the dataset, if the user needs them, and saves them in the train_data folder



What to do:

Insert your files in the user_scripts folder.
To insert the proper files, read the instructions.txt file and follow the structure of the example files that are located in the file_examples folder. You will find the instructions.txt file and the file_examples folder in the user_scripts folder.



