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


### Predict on simple set ###

inputs = []

from keras.applications.vgg16 import preprocess_input
import keras.utils as image

classifier = classifiers['1']

fruits = {953: 'pineapple', 950: 'orange', 954: 'banana'}
for fruit in fruits.values():
    img_path = 'user_scripts/'+fruit+'.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor = preprocess_input(img_tensor)
    inputs.append(img_tensor)

    preds_fruit = classifier.predict(img_tensor)[0]
    for fruit_x_ind, fruit_x in fruits.items():
        print('Image of', fruit, '=> Predicted probability for', fruit_x, ':', preds_fruit[fruit_x_ind])

all_totals = [180,190,200,210,220,230,240]
preds = solver.predict(inputs)[0]
print(preds)
##layer_names = ['total_'+str(t)+'_comb_no_'+str(c)+'_prob' for t in all_totals for c in range(1,8)]
##
##import keras
##
##input_layers = []
##for layer in solver.layers:
##    if isinstance(layer, keras.layers.InputLayer):
##        input_layers.append(layer.input)
##
##for layer in solver.layers:
##    if layer.name in layer_names:
##        multer = keras.models.Model(input_layers, layer.output)
##        new_preds = multer.predict(inputs)[0]
##        print(layer.name, new_preds)

print('Predicted total:', all_totals[np.argmax(preds)], ', with probability :', preds[np.argmax(preds)])
exit()
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















