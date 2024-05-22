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

all_objects = [obj for i in classifiers_conf for obj in objects_of_classifier[i]]       # we must not have duplicates
all_objects.sort()

list_of_possible_output_classes = neurasp_conf['list_of_possible_output_classes']


def extract_clause_strings(model):

    clause_strings = re.findall(r'(?<=object_class\()-?\d+,-?\d+(?=\))', model)

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
            print('Output class', output_class, 'corresponds to', len(models_of_output_class), 'object classes combination(s) (model(s)) totally:')
            for i in range(len(models_of_output_class)):
                model_comb = models_of_output_class[i]
                comb_list = ['object '+str(all_objects[j])+' with value '+model_comb[j] for j in range(len(all_objects))]
                comb_str = ' and '.join(comb_list)
                print('\tCombination', i, ':', comb_str)
            
        print('\n')

    if return_output_classes_list:
        return models_of_all_output_classes, satisfiable_output_classes      # [all_satisfiable_output_classes, models_for_each_rate, num_of_objects], [satisfiable output classes]
    else:
        return models_of_all_output_classes                              # [all_satisfiable_output_classes, models_for_each_rate, num_of_objects]







