import numpy as np

### Calculate Stable Models Function ###
from clingo.control import Control
import re

from user_scripts.training_configurations import training_configurations

neurasp_conf = training_configurations['neurasp_conf']
classifiers_conf = neurasp_conf['classifiers_conf']
# sort classifiers_conf dictionary
classifiers_conf = dict(sorted(classifiers_conf.items()))

objects_of_classifier = {i: sorted(classifiers_conf[i]['objects']) for i in classifiers_conf}
classes_of_classifier = {i: sorted(classifiers_conf[i]['list_of_object_classes']) for i in classifiers_conf}

all_objects = [obj for i in classifiers_conf for obj in objects_of_classifier[i]]       # we must not have duplicates
all_objects.sort()

classifier_of_object = {}
classes_of_object = {}
for obj in all_objects:
    for i in classifiers_conf:
        if obj in objects_of_classifier[i]:
            classifier_of_object[obj] = i
            classes_of_object[obj] = classes_of_classifier[i]

list_of_possible_output_classes = neurasp_conf['list_of_possible_output_classes']

unsat_value = float('-inf')

def extract_output_class(model):

    output_class = re.findall(r'(?<=output_class\()-?\d+(?=\))', model)[0]  # only 1 value for specific topics combination 
    output_class = int(output_class) if output_class else None

    return output_class


from user_scripts.create_asp_program import asp_program_for_object_classes
def find_output_class_for_known_obj_classes(object_classes_comb):

    program = asp_program_for_object_classes
    
    for obj_class in object_classes_comb:
        program = program.replace('$$$', str(obj_class), 1)

    clingo_control = Control(["0", "--warn=none"])

    models = []

    try:
        clingo_control.add("base", [], program)
    except:
        print("\nPi': \n{}".format(program))
    clingo_control.ground([("base", [])])
    clingo_control.solve(on_model = lambda model: models.append(model.symbols(atoms=True)))
    output_class = extract_output_class(str(models[0])) if models else unsat_value

    return output_class


    
def find_all_output_classes_for_known_obj_classes(print_details=False, return_output_classes_list=False):

    if print_details:
        print('Output class possible satisfiable answers:', str(list_of_possible_output_classes)[1:-1] )
        print('Output class unsatisfiable answer:', unsat_value )

    all_output_classes = []

    import itertools
    all_object_classes_combs = list(itertools.product(*[classes_of_object[obj] for obj in all_objects]))

    for object_classes_comb in all_object_classes_combs:
        output_class = find_output_class_for_known_obj_classes(object_classes_comb)
        all_output_classes.append(output_class)
        if print_details: print('Possible output_classes for object classes combination '+str(object_classes_comb)[1:-1]+':', output_class)
        

    if return_output_classes_list:
        output_classes_list = list(set(all_output_classes))
        output_classes_list.sort()
        try:
            output_classes_list.remove(unsat_value)
        except ValueError:
            pass
        return all_output_classes, output_classes_list      # (num of satisfiable combinations of num_of_objects objects,), output_classes_list
    else:
        return all_output_classes      # (num of satisfiable combinations of num_of_objects objects,)






