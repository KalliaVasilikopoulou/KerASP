## this is the body of your program
## it will always remain the same
asp_program_body = '''
                #const politics = 0.
                #const wellness = 1.
                #const entertainment = 2.
                #const travel = 3.
                #const style_beauty = 4.
                #const parenting = 5.
                #const healthy_living = 6.
                #const queer_voices = 7.
                #const food_drink = 8.
                #const business = 9.


                % each one of the 3 articles has a single topic
                1{article_about(A,T): T=0..9}1 :- A=0..2.   % T=0..9 because this is the range of the topics' classes

                % for easier calculations, topic hit rate is multiplied by 10
                % for example: if topic hit rate is 10% = 0.1, then we multiply it by 10,
                %so hit rate becomes 0.1*10 = 1
                %this means 1 corresponds to 10% and 10 corresponds to 100%
                
                topic_hit_rate(politics, 10). topic_hit_rate(wellness, 9). topic_hit_rate(entertainment, 8). 
                topic_hit_rate(travel, 7). topic_hit_rate(style_beauty, 6).topic_hit_rate(parenting, 5). 
                topic_hit_rate(healthy_living, 4). topic_hit_rate(queer_voices, 3). topic_hit_rate(food_drink, 2). 
                topic_hit_rate(business, 1).

                % to find the newspaper actual hit rate, the hit rate that the asp progam returns must be divided by 30
                % for example, if the asp program returns newspaper hit rate 3, 
                % then we must divide this number by 30, so hit rate will become 3/30 = 10% = 0.1
                %this means 3 corresponds to 10% and 30 corresponds to 100%

                1 { newspaper_hit_rate(P): P = 3..30 } 1 :- article_about(0,T1), article_about(1,T2), article_about(2,T3),
                                                            topic_hit_rate(T1, P1), topic_hit_rate(T2, P2), topic_hit_rate(T3, P3),
                                                            P = P1+P2+P3.
                    '''


## this is the part of your program that takes the initial labels as inputs and returns the new label as output
## f.e. in mnist addition it takes the labels of the mnist images as inputs and returns the sum as output
## warning: you have to set '$$$' for each one of the initial labels in order for the program to function
asp_program_for_object_classes ='''
                                #const topic_1 = $$$. #const topic_2 = $$$. #const topic_3 = $$$.

                                :- not article_about(0,topic_1).
                                :- not article_about(1,topic_2).
                                :- not article_about(2,topic_3).
                                

                                ''' + asp_program_body + '''

                                output_class(C) :- newspaper_hit_rate(C).
                                
                                #show output_class/1.
                                '''

## this is the part of your program that takes the new label as input and returns the initial labels as outputs
## f.e. in mnist addition it takes the sum as input and returns the possible labels of the mnist images as outputs
## warning: you have to set '$$$' for the new label in order for the program to function
asp_program_for_output_class = '''
                            #const hit_rate = $$$.

                            :- not newspaper_hit_rate(hit_rate).


                            ''' + asp_program_body + '''

                            object_class(O,C) :- article_about(O,C).
                            
                            #show object_class/2.
                               '''




