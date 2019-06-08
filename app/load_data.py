import os.path
import pandas as pd

# Flag to assure that load just one time
finalfix_is_loaded = False
failcode_is_loaded = False

# Excel sheet with fail codes per program
failcodes_sheet_path = os.path.join(os.path.dirname(__file__),'./data/raw/failcodes.xlsx')

# Excel sheet with finalfix per program
finalfix_sheet_path = os.path.join(os.path.dirname(__file__),'./data/raw/finalfix.xlsx')

# Paths of accuracies sheet files
accuracies_filename = {
    'Phenom': 'phenom_accuracies.xlsx',
    'Legacy500': 'legacy500_accuracies.xlsx',
    'Legacy600': 'legacy600_accuracies.xlsx',
    'Lineage': 'lineage_accuracies.xlsx'
}

# Paths of samples sheet files
samples_filename = {
    'Phenom': 'phenom_samples.xlsx',
    'Legacy500': 'legacy500_samples.xlsx',
    'Legacy600': 'legacy600_samples.xlsx',
    'Lineage': 'lineage_samples.xlsx'
}


# Return a Fail Code dictionary.
# Parameter: program name.
def get_dict_failcode(program):
    data = pd.read_excel(failcodes_sheet_path, program)
    data['FAILCODE'] = data['FAILCODE'].str.upper()
    data_indexed = data.set_index('FAILCODE')
    dict = data_indexed.to_dict()['CODE']
    return dict

def get_dict_finalfix(program):
    data = pd.read_excel(finalfix_sheet_path, program)
    data['FINALFIX'] = data['FINALFIX'].str.upper()
    data_indexed = data.set_index('FINALFIX')
    dict = data_indexed.to_dict()['CODE']
    return dict
# Function used to load failcodes sheet and create dictionaries.
def load_failcodes():
    global dict_failcodes

    dict_failcodes = {
        'Phenom': get_dict_failcode('Phenom'),
        'Legacy500': get_dict_failcode('Legacy500'),
        'Legacy600': get_dict_failcode('Legacy600'),
        'Lineage': get_dict_failcode('Lineage')
    }
def load_finalfix():
    global dict_finalfix

    dict_finalfix = {
        'Phenom': get_dict_finalfix('Phenom'),
        'Legacy500': get_dict_finalfix('Legacy500'),
        'Legacy600': get_dict_finalfix('Legacy600'),
        'Lineage': get_dict_finalfix('Lineage')
    }

# Return the accuracy filename per program
def get_samples_sheet_path(program, type):
    filename = samples_filename.get(program)
    sheet_path = os.path.join(os.path.dirname(__file__), f'./data/processed/{type}/{filename}')
    return sheet_path

# Return a dictionary with samples dict per program.
# Parameter: program name.
def get_dict_samples(program, type):
    sheet_path = get_samples_sheet_path(program, type)

    # Check if the file exists
    if os.path.isfile(sheet_path):
        data = pd.read_excel(sheet_path, 'Samples')
        if type == 'failcode':
            data['FAIL CODE DESCRIPTION'] = data['FAIL CODE DESCRIPTION'].str.upper()
            data_indexed = data.set_index('FAIL CODE DESCRIPTION')
        elif type == 'finalfix':
            data['FINAL FIX DESCRIPTION'] = data['FINAL FIX DESCRIPTION'].str.upper()
            data_indexed = data.set_index('FINAL FIX DESCRIPTION')
        dict = data_indexed.to_dict()['SAMPLES']
    else:
        dict = None
    return dict

# Function used to load the sample sheets and then create the dictionaries.
def load_samples(type):
    global dict_samples_finalfix, dict_samples_failcode

    if type == 'finalfix':
        dict_samples_finalfix = {
            'Phenom': get_dict_samples('Phenom', type),
            'Legacy500': get_dict_samples('Legacy500', type),
            'Legacy600': get_dict_samples('Legacy600', type),
            'Lineage': get_dict_samples('Lineage', type)
        }
    elif type == 'failcode':
        dict_samples_failcode = {
            'Phenom': get_dict_samples('Phenom', type),
            'Legacy500': get_dict_samples('Legacy500', type),
            'Legacy600': get_dict_samples('Legacy600', type),
            'Lineage': get_dict_samples('Lineage', type)
        }

# Return the accuracy filename per program
def get_accuracies_sheet_path(program, type):
    filename = accuracies_filename.get(program)
    sheet_path = os.path.join(os.path.dirname(__file__), f'./data/processed/{type}/{filename}')
    return sheet_path

# Return a dictionary with accuracies per program.
# Parameter: program name.
def get_dict_accuracy(program, type):
    sheet_path = get_accuracies_sheet_path(program, type)

    # Check if the file exists
    if os.path.isfile(sheet_path):
        data = pd.read_excel(sheet_path, 'Accuracies')
        if type == 'failcode':
            data['FAIL CODE DESCRIPTION'] = data['FAIL CODE DESCRIPTION'].str.upper()
            data_indexed = data.set_index('FAIL CODE DESCRIPTION')
        elif type == 'finalfix':
            data['FINAL FIX DESCRIPTION'] = data['FINAL FIX DESCRIPTION'].str.upper()
            data_indexed = data.set_index('FINAL FIX DESCRIPTION')
        dict = data_indexed.to_dict()['ACCURACY']
    else:
        dict = None
    return dict

# Function used to load the accuracies sheet and then create dictionaries.
def load_accuracies(type):
    global dict_accuracies_failcode, dict_accuracies_finalfix

    if type == 'failcode':
        dict_accuracies_failcode = {
            'Phenom': get_dict_accuracy('Phenom', type),
            'Legacy500': get_dict_accuracy('Legacy500', type),
            'Legacy600': get_dict_accuracy('Legacy600', type),
            'Lineage': get_dict_accuracy('Lineage', type)
        }
    elif type == 'finalfix':
        dict_accuracies_finalfix = {
            'Phenom': get_dict_accuracy('Phenom', type),
            'Legacy500': get_dict_accuracy('Legacy500', type),
            'Legacy600': get_dict_accuracy('Legacy600', type),
            'Lineage': get_dict_accuracy('Lineage', type)
        }


# Executed as a module
if __name__ != '__main__':
    if not finalfix_is_loaded:
        finalfix_is_loaded = True
        load_finalfix()
        load_samples('finalfix')
        load_accuracies('finalfix')  
    if not failcode_is_loaded:    
        failcode_is_loaded = True
        load_failcodes()
        load_samples('failcode')
        load_accuracies('failcode')   