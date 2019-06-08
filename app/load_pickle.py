import sys
import os.path
from sklearn.externals import joblib

# failcode_paths_dict = {
#     'Phenom': '../models/failcode/phenom_failcode_clf.pkl',
#     'Legacy500': '../models/failcode/legacy500_failcode_clf.pkl',
#     'Legacy600': '../models/failcode/legacy600_failcode_clf.pkl',
#     'Lineage': '../models/failcode/lineage_failcode_clf.pkl'
# }

failcode_paths_dict = {
    'Phenom': '../models/failcode/phenom_failcode_clf.pkl',
    'Legacy500': '../models/failcode/legacy500_failcode_clf.pkl',
    'Legacy600': '../models/failcode/legacy600_failcode_clf.pkl',
    'Lineage': '../models/failcode/lineage_failcode_clf.pkl'
}

finalfix_paths_dict = {
    'Phenom': '../models/finalfix/phenom_finalfix_clf.pkl',
    'Legacy500': '../models/finalfix/legacy500_finalfix_clf.pkl',
    'Legacy600': '../models/finalfix/legacy600_finalfix_clf.pkl',
    'Lineage': '../models/finalfix/lineage_finalfix_clf.pkl'
}


"""
    Function which aims to inicialize variables.
"""
def setup():
    # Specify the pickle loaded in memory. Example: failcode_Phenom
    global pickle_in_memory
    pickle_in_memory = ""

    # Failcode pickle variable
    global clf    

"""
    Function which aims to determine if is necessary load the program's pickle.
    - program: specify the program contained in JSON sended by user.
    - type: can be [failcode, ].
"""
def handle_load_pickle(program, type):
    #Avoid wasting time in empty inputs
    if program != "":
        type_program = f'{type}_{program}'
        if type_program != pickle_in_memory:
            load_clf(program, type)
    else:
        type_program = None

"""
    Function used to load the failcode pickle.
    - program: specify the program to load its pickle.
"""
def load_clf(program, type):
    global pickle_in_memory, clf

    print(f'Loading {type} pickle for {program}')

    # Update variable that indicates the pickle loaded in memory

    pickle_in_memory = f'{type}_{program}'

    # Program's pickle path
    if type == 'finalfix':
        pickle_path = os.path.join(os.path.dirname(__file__), finalfix_paths_dict.get(program))
    else:
        pickle_path = os.path.join(os.path.dirname(__file__), failcode_paths_dict.get(program))
    # Clean pickle variable 
    clf = None

    # Load pickle
    clf = joblib.load(pickle_path)


# Executed as a module
if __name__ != '__main__':
    setup()