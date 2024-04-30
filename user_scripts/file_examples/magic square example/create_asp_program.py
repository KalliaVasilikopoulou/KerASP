## this is the body of your program
## it will always remain the same
asp_program_body = '''
                % each cell has only one value
                1 { value(X,Y,I): I=1..9 } 1 :- X=0..2, Y=0..2.

                % at least 1 magic square exists
                1 { magic_square(S): S=1*3..9*3 }.

                % the sum of the numbers of any row must be equal to S
                :- magic_square(S), X=0..2, #sum{I,Y: value(X,Y,I)} != S.

                % the sum of the numbers of any column must be equal to S
                :- magic_square(S), Y=0..2, #sum{I,X: value(X,Y,I)} != S.

                % the sum of the numbers of the two diagonals must be equal to S
                :- magic_square(S), #sum{I,X: value(X,X,I)} != S.
                :- magic_square(S), #sum{I,X: value(X,2-X,I)} != S.
                    '''


## this is the part of your program that takes the initial labels as inputs and returns the new label as output
## f.e. in mnist addition it takes the labels of the mnist images as inputs and returns the sum as output
## warning: you have to set '$$$' for each one of the initial labels in order for the program to function
asp_program_for_object_classes ='''
                                #const val_1 = $$$. #const val_2 = $$$. #const val_3 = $$$.

                                % fixed values
                                :- not value(0,0,val_1). 
                                :- not value(0,1,val_2). 
                                :- not value(0,2,val_3).
                                
                                ''' + asp_program_body + '''

                                output_class(C) :- value(2,2,C).

                                #show output_class/1.
                                '''

## this is the part of your program that takes the new label as input and returns the initial labels as outputs
## f.e. in mnist addition it takes the sum as input and returns the possible labels of the mnist images as outputs
## warning: you have to set '$$$' for the new label in order for the program to function
asp_program_for_output_class = '''
                            #const val = $$$.

                            % fixed values
                            :- not value(2,2,val).

                            ''' + asp_program_body + '''

                            
                            object_class(0,C) :- value(0,0,C).
                            object_class(1,C) :- value(0,1,C).
                            object_class(2,C) :- value(0,2,C).

                            #show object_class/2.
                               '''




