### Import Model ###

from build_model.build_solver import Solver

solver_inst = Solver(compile_model=True)
solver_inst.model_obj_summary()
##solver_inst.visualize_model_obj()
solver_inst.plot_model_obj()

solver = solver_inst.model_obj

classifier = solver_inst.model_obj_classifier


### save model ###

# save solver model
solver.save('saved_models/graph_execution/solver/solver_model.h5')
print('saved solver model')

# save classifier model
classifier.save('saved_models/graph_execution/classifier/classifier_model.h5')
print('saved classifier model')


### save model weights ###

# save solver weights
solver.save_weights('saved_models/graph_execution/solver/solver_weights.h5')
print('saved solver weights')

# save classifier weights
classifier.save_weights('saved_models/graph_execution/classifier/classifier_weights.h5')
print('saved classifier weights')















