import numpy as np

from project_utils.convert_to_categorical import custom_to_categorical as to_categorical


### Set Training Parameters ###

from user_scripts.training_configurations import training_configurations

neurasp_conf = training_configurations['neurasp_conf']
classifiers_conf = neurasp_conf['classifiers_conf']

objects_of_classifier = {i: sorted(classifiers_conf[i]['objects']) for i in classifiers_conf}

try: tokenize_data = training_configurations['tokenize_data']
except KeyError: tokenize_data = False


### Import Model ###

from build_model.build_solver import Solver

solver_inst = Solver(compile_model=True)
solver_inst.model_obj_summary()
##solver_inst.visualize_model_obj()
solver_inst.plot_model_obj()

solver = solver_inst.model_obj

classifiers = solver_inst.model_obj_classifiers


### save model ###

# save solver model
solver.save('saved_models/graph_execution/solver/solver_model.h5')
print('saved solver model')

# save classifier model
for ind_classifier in classifiers_conf:
    classifiers[ind_classifier].save('saved_models/graph_execution/classifier/classifier_'+str(ind_classifier)+'_model.h5')
print('saved classifier model')


### save model weights ###

# save solver weights
solver.save_weights('saved_models/graph_execution/solver/solver_weights.h5')
print('saved solver weights')

# save classifier weights
for ind_classifier in classifiers_conf:
    classifiers[ind_classifier].save_weights('saved_models/graph_execution/classifier/classifier_'+str(ind_classifier)+'_weights.h5')
print('saved classifier weights')















