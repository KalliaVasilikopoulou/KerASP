## this is the body of your program
## it will always remain the same

from user_scripts.vgg16_class_indices import vgg16_class_indices
from user_scripts.extract_prices import prices

all_items = list(prices.keys())

set_all_items = ';'.join(all_items)
set_const_values = ''
set_prices = ''

for item in all_items:
    set_const_values += '#const ' + item + ' = ' + str(vgg16_class_indices[item]) + '.\n'
    set_prices += 'price(' + item + ',' + str(prices[item]) + ').\n'

asp_program_body = '''
                ''' + set_const_values + '''

                1{contains(I,F): F=(''' + set_all_items + ''')}1 :- I=(i1;i2;i3).

                ''' + set_prices + '''

                total(X) :- X = #sum{V,I: price(F,V), contains(I,F)}.
                    '''


## this is the part of your program that takes the initial labels as inputs and returns the new label as output
## f.e. in mnist addition it takes the labels of the mnist images as inputs and returns the sum as output
## warning: you have to set '$$$' for each one of the initial labels in order for the program to function
asp_program_for_object_classes ='''
                            :- not contains(i1, $$$).
                            :- not contains(i2, $$$).
                            :- not contains(i3, $$$).

                            ''' + asp_program_body + '''

                            output_class(C) :- total(C).

                            #show output_class/1.
                               '''

## this is the part of your program that takes the new label as input and returns the initial labels as outputs
## f.e. in mnist addition it takes the sum as input and returns the possible labels of the mnist images as outputs
## warning: you have to set '$$$' for the new label in order for the program to function
asp_program_for_output_class = '''
                                :- not total($$$).
                                
                                ''' + asp_program_body + '''

                                object_class(0,C) :- contains(i1,C).
                                object_class(1,C) :- contains(i2,C).
                                object_class(2,C) :- contains(i3,C).

                                #show object_class/2.
                                '''




