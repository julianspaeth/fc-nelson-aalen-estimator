import pickle
import queue as q
import redis

pool = redis.BlockingConnectionPool(host='localhost', port=6379, db=0, queue_class=q.Queue)
r = redis.Redis(connection_pool=pool)


def redis_get(key: str):
    """
    Gets the value of a certain key from the redis store
    :param key: Key of redis store
    :return: The corresponding value behind the key. None if there is no such key.
    """
    if key in r:
        return pickle.loads(r.get(key))
    else:
        return None


def redis_set(key: str, value: any):
    """
    Saves a value under the corresponding key in the redis store.
    :param key: Key to access the value later using redis_get.
    :param value: Value which is stored behind the corresponding key.
    :return: None
    """
    r.set(key, pickle.dumps(value))


def set_step(next_step):
    """
    Sets the step key in the redis store with the next_step value.
    :param next_step: Value of the next step
    :return: None
    """
    redis_set('step', next_step)


def get_step():
    """
    Gets the current step of the redis store
    :return: Value of the current step.
    """
    return redis_get('step')
