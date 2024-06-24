import os

MODEL_PARAMS = {
    'n_estimators': 100,
    'random_state': 42,
    'warm_start': True
}

CHECKPOINT_DIR = 'ckpt/'


def get_checkpoint_path(params):
    filename = 'rf_checkpoint_' + \
        '_'.join([f"{key}_{value}" for key, value in params.items()]) + '.pkl'
    return os.path.join(CHECKPOINT_DIR, filename)
