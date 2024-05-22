## this is the body of your program
## it will always remain the same
asp_program_body = '''
                #const shirt = 0.
                #const trouser = 1.
                1{contains(I,D): D=(trouser;shirt)}1 :- I=(i1;i2).

                two_shirts(1) :- contains(i1, shirt), contains(i2, shirt).
                two_shirts(0) :- not two_shirts(1).
                    '''


## this is the part of your program that takes the initial labels as inputs and returns the new label as output
## f.e. in mnist addition it takes the labels of the mnist images as inputs and returns the sum as output
## warning: you have to set '$$$' for each one of the initial labels in order for the program to function
asp_program_for_object_classes ='''
                            :- not contains(i1, $$$).
                            :- not contains(i2, $$$).

                            ''' + asp_program_body + '''

                            output_class(C) :- two_shirts(C).

                            #show output_class/1.
                               '''

## this is the part of your program that takes the new label as input and returns the initial labels as outputs
## f.e. in mnist addition it takes the sum as input and returns the possible labels of the mnist images as outputs
## warning: you have to set '$$$' for the new label in order for the program to function
asp_program_for_output_class = '''
                                :- not two_shirts($$$).
                                
                                ''' + asp_program_body + '''

                                object_class(0,C) :- contains(i1,C).
                                object_class(1,C) :- contains(i2,C).

                                #show object_class/2.
                                '''




