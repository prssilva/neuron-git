from flask import Flask, jsonify, request
from flask import abort
from flask import make_response
from flask_cors import CORS
import json
from app.preprocessing import *
from app.failcode_classifier import *
from app.finalfix_classifier import *
from app.load_pickle import *
from app.load_data import *

def create_app():
    app = Flask(__name__)
    return app

app = create_app()
CORS(app)

'''
    Melhora a forma padrão do 404 do flask (HTML)
    Responde com JSON informando o erro
'''

@app.errorhandler(400)
def invalid_request(error):
    return make_response(jsonify({'error': {'message':error}}), 400)

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': {'message':error}}), 404)

@app.errorhandler(500)
def server_error(error):
    return make_response(jsonify({'error': {'message':error}}), 500)

@app.errorhandler(502)
def bad_gateway(error):
    return make_response(jsonify({'error': {'message':error}}), 502)

@app.errorhandler(503)
def service_unavailable(error):
    return make_response(jsonify({'error': {'message':error}}), 503)


@app.route('/')
def index():
    print(dict_failcodes.keys())
    return "Hello, World! <br><br> I'm a simple Flask app."

@app.route('/api/classifier/failcode', methods=['POST'])
def failcode_classifier_route():
    data = request.json

    try:
        # Check if is necessary to load the program's pickle
        program = data.get('program')
        handle_load_pickle(program, 'failcode')
        # Data to classify
        to_classify = data.get('excel')    
        # Classify:
        result = init_classifier_failcode(to_classify, program)
    #input error
    except TypeError:
        #Missing program in json
        return invalid_request("Missing program")
    except KeyError:
        #Missing input Excel in json
        return invalid_request("Missing classifier input data")
    #Resources not found but will be available after bug fix
    except FileNotFoundError:
        #Model's pickle not found
        return not_found("Classifier model not found")
    #Internal server error
    except AttributeError:
        #Accuracies' or samples' excels not found
        return server_error("Accuracy or samples excel not found")


    return result, 200, {"Content-Type": "application/json"}

@app.route('/api/classifier/finalfix', methods=['POST'])
def finalfix_classifier_route():
    data = request.json

    try:
        # Check if is necessary to load the program's pickle
        program = data.get('program')
        handle_load_pickle(program, 'finalfix')
        # Data to classify
        to_classify = data.get('excel')    
        # Classify:
        result = init_classifier_finalfix(to_classify, program)
    #input error
    except TypeError:
        #Missing program in json
        return invalid_request("Missing program")
    except KeyError:
        #Missing input Excel in json
        return invalid_request("Missing classifier input data")
    #Resources not found but will be available after bug fix
    except FileNotFoundError:
        #Model's pickle not found
        return not_found("Classifier model not found")
    #Internal server error
    except AttributeError:
        #Accuracies' or samples' excels not found
        return server_error("Accuracy or samples excel not found")

    return result, 200, {"Content-Type": "application/json"}


if __name__ == '__main__':
    app.run(debug=False)