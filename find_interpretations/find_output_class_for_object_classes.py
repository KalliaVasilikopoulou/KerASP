import numpy as np

### Calculate Stable Models Function ###
from clingo.control import Control
import re

from training_configurations import training_configurations

program_specs = training_configurations['program_specs']

num_of_objects = program_specs['num_of_objects']
list_of_object_classes = program_specs['list_of_object_classes']
classes_per_object = len(list_of_object_classes)

list_of_possible_output_classes = program_specs['list_of_possible_output_classes']


def extract_output_class(model):

    output_class = re.findall(r'(?<=output_class\()\d+(?=\))', model)[0]  # only 1 value for specific topics combination 
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
    output_class = extract_output_class(str(models[0])) if models else -1      # -1=unsat

    return output_class


    
def find_all_output_classes_for_known_obj_classes(print_details=False, return_output_classes_list=False):

    if print_details:
        print('Output class possible satisfiable answers:', str(list_of_possible_output_classes)[1:-1] )
        print('Output class unsatisfiable answer:', -1 )

    all_output_classes = []

    from project_utils.convert_number import number_to_base

    for num in range(classes_per_object**num_of_objects):
        object_classes_comb = number_to_base(num,classes_per_object)     # base is equal to the number of the object classes (classifier output classes)
        object_classes_comb = [0]*(num_of_objects-len(object_classes_comb))+object_classes_comb
        output_class = find_output_class_for_known_obj_classes(object_classes_comb)
        all_output_classes.append(output_class)
        if print_details: print('Possible output_classes for object classes combination '+str(object_classes_comb)[1:-1]+':', output_class)
        

    if return_output_classes_list:
        output_classes_list = list(set(all_output_classes))
        output_classes_list.sort()
        try:
            output_classes_list.remove(-1)
        except ValueError:
            pass
        return all_output_classes, output_classes_list      # (num of satisfiable combinations of num_of_objects objects,), output_classes_list
    else:
        return all_output_classes      # (num of satisfiable combinations of num_of_objects objects,)






