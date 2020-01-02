import numpy as np
import pandas as pd
import sys

def digit_preference_first_after_dec(input,tidy_out=False):
    return(digit_preference_x_after_digit(input,2,tidy_out))

def digit_preference_second_after_dec(input,tidy_out=False):
    return(digit_preference_x_after_digit(input,3,tidy_out))

def digit_preference_x_after_digit(input,x,tidy_out):
        data = input.values
        data = data[:,:-1]
        ben = np.zeros((data.shape[0],11))
        for i in range(1, data.shape[0]):
            for j in range(0, data.shape[1]):
                if len(str(abs(data[i,j]) % 1)) > x :
                    num = int(str(abs(data[i,j]) % 1)[x])
                else:
                    num = 10
                ben[i,num] = ben[i,num] + 1
        ben = ben / data.shape[1]
        out = pd.DataFrame(ben)
        out.columns = [0,1,2,3,4,5,6,7,8,9,'None']
        out['sample_id'] = range(0,data.shape[0])
        out['labels'] = input['labels']
        if tidy_out:
            print('TIDY OUT DOES NOT CURRENTLY WORK!!! just an FYI...')
            out = pd.melt(out,["sample_id","labels"],var_name="digit",value_name="freq")
        return out
