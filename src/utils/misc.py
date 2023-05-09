def set_random_state_and_save():
    import random
    import numpy as np

    state = random.getstate(), np.random.get_state()

    return state


def restore_random_state(state):
    import random
    import numpy as np

    random.setstate(state[0])
    np.random.set_state(state[1])
