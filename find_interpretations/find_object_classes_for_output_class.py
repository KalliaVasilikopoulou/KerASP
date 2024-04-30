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


def extract_clause_strings(model):

    clause_strings = re.findall(r'(?<=object_class\()\d+,\d+(?=\))', model)

    return clause_strings


def simplify_clause_strings(clause_strings):
    
    clause_lists = [[int(num_str) for num_str in clause_string.split(',')] for clause_string in clause_strings]

    clause_lists.sort()

    clauses_simplified = [clause[1] for clause in clause_lists]

    return clauses_simplified


from user_scripts.create_asp_program import asp_program_for_output_class
def find_obj_classes_for_known_output_class(output_class):

    program = asp_program_for_output_class

    program = program.replace('$$$', str(output_class))

    clingo_control = Control(["0", "--warn=none"])

    models = []

    try:
        clingo_control.add("base", [], program)
    except:
        print("\nPi': \n{}".format(program))
    clingo_control.ground([("base", [])])
    clingo_control.solve(on_model = lambda model: models.append(model.symbols(atoms=True)))
    if models:
        models = [simplify_clause_strings(extract_clause_strings(str(model))) for model in models]      # (models_for_each_output_class,3)

    return models      # (models_for_each_output_class,3)


    
def find_all_obj_classes_for_known_output_classes(print_details=False, return_output_classes_list=False):

    satisfiable_output_classes = []
    models_of_all_output_classes = []
    for output_class in list_of_possible_output_classes: 
        models_of_output_class = find_obj_classes_for_known_output_class(output_class)
        if models_of_output_class:
            satisfiable_output_classes.append(output_class)
            models_of_all_output_classes.append(models_of_output_class)      # [all_satisfiable_output_classes, models_for_each_rate, 3]

    if print_details:
        print('\nOutput classes interpretations details:')

        print('\nA total of', str(len(satisfiable_output_classes)), 'output classes have satisfiable models')
        print('These classes are:', ', '.join(str(s) for s in satisfiable_output_classes))

        total_models = sum([len(models_of_output_class) for models_of_output_class in models_of_all_output_classes])
        print('\nA total of', total_models, 'object classes combinations (models) can be formed for these', str(len(satisfiable_output_classes)), 'output classes')
        
        print('\nObject classes combinations (models) that are formed for each output class:')
        for output_class, models_of_output_class in zip(satisfiable_output_classes, models_of_all_output_classes):
            print('Output class', output_class, 'corresponds to', len(models_of_output_class), 'object classes combination(s) (model(s)) totally')
        
        obj_class_count_per_object = np.zeros((num_of_objects, classes_per_object), dtype='int32')      # (objects, obj_classes)
        for models_of_output_class in models_of_all_output_classes:
            for model in models_of_output_class:
                for object_i,obj_class in enumerate(model):
                    obj_class_count_per_object[object_i][obj_class] += 1

        print('\obj_class count per object:')
        for object_i, obj_classes_count in zip(range(num_of_objects), obj_class_count_per_object):
            for obj_class, count in enumerate(obj_classes_count):
                print('obj_class', obj_class, 'was found', count, 'times in object', object_i)
            
        print('\nobj_class count for all objects:')
        obj_class_count = np.sum(obj_class_count_per_object, axis=0)
        for obj_class, count in enumerate(obj_class_count):
            print('obj_class', obj_class, 'was found', count, 'times')

        print('\nMaximum times a single obj_class was found in a single object:')
        object_max_count = np.max(obj_class_count_per_object, axis=1)
        object_max_count_obj_class = np.argmax(obj_class_count_per_object, axis=1)
        for object_i, max_count, max_count_obj_class in zip(range(num_of_objects), object_max_count, object_max_count_obj_class):
            print('obj_class', max_count_obj_class, 'was found', max_count, 'times totally in object', object_i)

        print('\nMinimum times a single obj_class was found in a single object:')
        object_min_count = np.min(obj_class_count_per_object, axis=1)
        object_min_count_obj_class = np.argmin(obj_class_count_per_object, axis=1)
        for object_i, min_count, min_count_obj_class in zip(range(num_of_objects), object_min_count, object_min_count_obj_class):
            print('obj_class', min_count_obj_class, 'was found', min_count, 'times totally in object', object_i)
            
        print('\n')

    if return_output_classes_list:
        return models_of_all_output_classes, satisfiable_output_classes      # [all_satisfiable_output_classes, models_for_each_rate, num_of_objects], [satisfiable output classes]
    else:
        return models_of_all_output_classes                              # [all_satisfiable_output_classes, models_for_each_rate, num_of_objects]







