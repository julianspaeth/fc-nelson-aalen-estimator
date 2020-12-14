import queue as q

import pandas as pd
import redis
import rq
import yaml
from flask import Blueprint, jsonify, request, current_app

from fc_app.kaplan_meier import calculate_global, calculate_local_counts, read_input, write_results
from redis_util import redis_set, redis_get, get_step, set_step

pool = redis.BlockingConnectionPool(host='localhost', port=6379, db=0, queue_class=q.Queue)
r = redis.Redis(connection_pool=pool)

# setting 'available' to False --> no data will be send around.
# Change it to True later to send data from the coordinator to the clients or vice versa.
redis_set('available', False)

# The various steps of the KM app. This list is not really used and only an overview.
STEPS = ['start', 'setup', 'local_calculation', 'waiting', 'global_calculation', 'broadcast_results', 'write_results',
         'finalize', 'finished']

# Initializes the app with the first step
set_step('start')
# Initialize the local and global data
redis_set('local_data', None)
redis_set('global_data', [])

# Set the paths of the input and output dir
INPUT_DIR = "/mnt/input"
OUTPUT_DIR = "/mnt/output"

api_bp = Blueprint('api', __name__)
tasks = rq.Queue('fc_tasks', connection=r)


@api_bp.route('/status', methods=['GET'])
def status():
    """
    GET request to /status, if True is returned a GET data request will be send
    :return: JSON with key 'available' and value True or False and 'finished' value True or False
    """
    available = redis_get('available')
    current_app.logger.info('[API] /status GET request ' + str(available) + ' - [STEP]: ' + str(get_step()))

    if get_step() == 'start':
        current_app.logger.info('[STEP] start')
        current_app.logger.info('[API] Federated Kaplan-Meier Estimator')

    elif get_step() == 'local_calculation':
        current_app.logger.info('[STEP] local_calculation')
        local_counts = calculate_local_counts()

        if redis_get('is_coordinator'):
            # if this is the coordinator, directly add the local counts and number of samples to the global_data list
            global_data = redis_get('global_data')
            dfs = {}
            for key in local_counts.keys():
                dfs[key] = pd.DataFrame.from_dict(local_counts[key], dtype=float)
            global_data.append(dfs)
            redis_set('global_data', global_data)
            current_app.logger.info('[STEP] : waiting_for_clients')
            set_step('waiting')
        else:
            # if this is a client, set the local counts and number of samples to local_data and set available to true
            redis_set('local_data', local_counts)
            current_app.logger.info('[STEP] waiting_for_coordinator')
            redis_set('available', True)
            set_step('waiting')

    elif get_step() == 'waiting':
        current_app.logger.info('[STEP] waiting')
        if redis_get('is_coordinator'):
            current_app.logger.info('[API] Coordinator checks if data of all clients has arrived')
            # check if all clients have sent their data already
            has_client_data_arrived()
        else:
            # the clients wait for the coordinator to finish
            current_app.logger.info('[API] Client waiting for coordinator to finish')

    elif get_step() == 'global_calculation':
        # as soon as all data has arrived the global calculation starts
        current_app.logger.info('[STEP] global_calculation')
        calculate_global()
        set_step("broadcast_results")

    elif get_step() == 'broadcast_results':
        # as soon as the global km was calculated, the result is broadcasted to the clients
        current_app.logger.info('[STEP] broadcast_results')
        global_km = redis_get('global_km')
        current_app.logger.info(global_km)
        redis_set('available', True)
        set_step('write_results')

    elif get_step() == 'write_results':
        # The global km is written to the output directory
        current_app.logger.info('[STEP] write_results')
        write_results(redis_get('global_km'), OUTPUT_DIR)
        current_app.logger.info('[API] Finalize client')
        if redis_get('is_coordinator'):
            # The coordinator is already finished now
            redis_set('finished', [True])
        # Coordinator and clients continue with the finalize step
        set_step("finalize")

    elif get_step() == 'finalize':
        current_app.logger.info('[STEP] finalize')
        current_app.logger.info("[API] Finalize")
        if redis_get('is_coordinator'):
            # The coordinator waits until all clients have finished
            if have_clients_finished():
                current_app.logger.info('[API] Finalize coordinator.')
                set_step('finished')
            else:
                current_app.logger.info('[API] Not all clients have finished yet.')
        else:
            # The clients set available true to signal the coordinator that they have written the results.
            redis_set('available', True)

    elif get_step() == 'finished':
        # All clients and the coordinator set available to False and finished to True and the computation is done
        current_app.logger.info('[STEP] finished')
        return jsonify({'available': False, 'finished': True})

    return jsonify({'available': True if available else False, 'finished': False})


@api_bp.route('/data', methods=['GET', 'POST'])
def data():
    """
    GET request to /data sends data to coordinator
    POST request to /data pulls data from coordinator
    :return: GET request: JSON with key 'data' and value data
             POST request: JSON True
    """
    if request.method == 'POST':
        current_app.logger.info('[API] /data POST request')
        if redis_get('is_coordinator'):
            # Get data from clients (as coordinator)
            if get_step() != 'finalize':
                # Get local counts of the clients
                global_data = redis_get('global_data')
                dfs = {}
                client_counts = request.get_json(True)['data']
                for key in client_counts.keys():
                    dfs[key] = pd.DataFrame.from_dict(client_counts[key], dtype=float)
                global_data.append(dfs)
                redis_set('global_data', global_data)
                return jsonify(True)
            else:
                # Get Finished flags of the clients
                request.get_json(True)
                finish = redis_get('finished')
                finish.append(request.get_json(True)['finished'])
                redis_set('finished', finish)
                return jsonify(True)
        else:
            # Get global result from coordinator (as client)
            current_app.logger.info('GET GLOBAL RESULT FROM COORDINATOR!')
            current_app.logger.info('[API] ' + str(request.get_json()))
            redis_set('global_km', request.get_json(True)['global_km'])
            current_app.logger.info('[API] ' + str(redis_get('global_km')))
            set_step('write_results')
            return jsonify(True)

    elif request.method == 'GET':
        current_app.logger.info('[API] /data GET request')
        if not redis_get('is_coordinator'):
            # send data to coordinator (as client)
            if get_step() != 'finalize':
                # Send local KM to the coordinator
                current_app.logger.info('[API] send data to coordinator')
                redis_set('available', False)
                local_data = redis_get('local_data')

                return jsonify({'data': local_data})
            else:
                # Send finish flag to the coordinator
                current_app.logger.info('[API] send finish flag to coordinator')
                redis_set('available', False)
                set_step('finished')
                return jsonify({'finished': True})
        else:
            # broadcast data to clients (as coordinator)
            current_app.logger.info('[API] broadcast data from coordinator to clients')
            redis_set('available', False)
            global_km = redis_get('global_km')
            current_app.logger.info(global_km)
            return jsonify({'global_km': global_km})

    else:
        current_app.logger.info('[API] Wrong request type, only GET and POST allowed')
        return jsonify(True)


@api_bp.route('/setup', methods=['POST'])
def setup():
    """
    set setup params, id is the id of the client, coordinator is True if the client is the coordinator,
    in global_data the data from all clients (including the coordinator) will be aggregated,
    clients is a list of all ids from all clients, nr_clients is the number of clients involved in the app
    :return: JSON True
    """
    set_step('setup')
    current_app.logger.info('[STEP] setup')
    read_config()
    retrieve_setup_parameters()
    input_data = read_input(INPUT_DIR)
    if input_data is None:
        current_app.logger.info('[API] no data was found.')
        return jsonify(False)
    else:
        redis_set('files', input_data)
        set_step("local_calculation")
        return jsonify(True)


def retrieve_setup_parameters():
    """
    Retrieve the setup parameters and store them in the redis store
    :return: None
    """
    current_app.logger.info('[API] Retrieve Setup Parameters')
    setup_params = request.get_json()
    current_app.logger.info(setup_params)
    redis_set('id', setup_params['id'])
    is_coordinator = setup_params['master']
    redis_set('is_coordinator', is_coordinator)
    if is_coordinator:
        redis_set('global_data', [])
        redis_set('finished', [])
        redis_set('clients', setup_params['clients'])
        redis_set('nr_clients', len(setup_params['clients']))


def has_client_data_arrived():
    """
    Checks if the data of all clients has arrived.
    :return: None
    """
    current_app.logger.info('[API] Coordinator checks if data of all clients has arrived')
    global_data = redis_get('global_data')
    nr_clients = redis_get('nr_clients')
    current_app.logger.info('[API] ' + str(len(global_data)) + "/" + str(nr_clients) + " clients have sent their data.")
    if len(global_data) == nr_clients:
        current_app.logger.info('[API] Data of all clients has arrived')
        set_step('global_calculation')
    else:
        current_app.logger.info('[API] Data of at least one client is still missing')


def have_clients_finished():
    """
    Checks if the all clients have finished.
    :return: True if all clients have finished, False otherwise
    """
    current_app.logger.info('[API] Coordinator checks if all clients have finished')
    finish = redis_get('finished')
    nr_clients = redis_get('nr_clients')
    current_app.logger.info('[API] ' + str(len(finish)) + "/" + str(nr_clients) + " clients have finished already.")
    if len(finish) == nr_clients:
        current_app.logger.info('[API] All clients have finished.')
        return True
    else:
        current_app.logger.info('[API] At least one client did not finish yet-')
        return False


def read_config():
    with open(INPUT_DIR + '/config.yml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)['fc_kaplan_meier']

        redis_set('input_filename', config['files']['input'])
        redis_set('survival_function_filename', config['files']['output']['survival_function'])
        redis_set('survival_plot_filename', config['files']['output']['survival_plot'])
        redis_set('logrank_test_filename', config['files']['output']['logrank_test'])

        redis_set('duration_col', config['parameters']['duration_col'])
        redis_set('event_col', config['parameters']['event_col'])
        redis_set('category_col', config['parameters']['category_col'])
        redis_set('differential_privacy', config['parameters']['differential_privacy'])
        redis_set('multipletesting_method', config['parameters']['multipletesting_method'])
