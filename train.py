import numpy as np
import h5py
import tensorflow as tf
from tensorflow.python.client import device_lib
import keras
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import os
import pickle
import argparse
import json
import time

import util
import loader
import models

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
# print(device_lib.list_local_devices())
# print(keras.backend.floatx())


def transfer_weights(model, weights_path, ensemble_load=False, wait_for_load=False, ens_id=None, sleeptime=600):
    td = None
    conv1d = None
    td_name = None
    conv1d_name = None
    for layer in model.layers:
        if layer.name.find('time_distributed') != -1:
            td = layer.layer
            td_name = layer.name
            break
    for layer in td.layers:
        if layer.name.find('conv1d') != -1:
            conv1d = layer
            conv1d_name = layer.name
            break
    model_borehole = conv1d.get_weights()[0].shape[1] == 64

    if ensemble_load:
        weights_path = os.path.join(weights_path, f'{ens_id}')

    # If weight file does not exists, wait until it exists. Intended for ensembles. Warning: Can deadlock program.
    if wait_for_load:
        if os.path.isfile(weights_path):
            target_object = weights_path
        else:
            target_object = os.path.join(weights_path, 'hist.pkl')

        while not os.path.exists(target_object):
            print(f'File {target_object} for weight transfer missing. Sleeping for {sleeptime} seconds.')
            time.sleep(sleeptime)

    if os.path.isdir(weights_path):
        last_weight = sorted([x for x in os.listdir(weights_path) if x[:6] == 'event-'])[-1]
        weights_path = os.path.join(weights_path, last_weight)

    with h5py.File(weights_path, 'r') as weights:
        weights_borehole = weights[td_name][conv1d_name]['kernel:0'].shape[1] == 64
        weights_dict = generate_weights_dict(weights)
    del_list = []
    for weight in weights_dict:
        if weight[:9] == 'embedding':
            del_list += [weight]
    for del_element in del_list:
        del weights_dict[del_element]
    if model_borehole and not weights_borehole:
        # Take same weights for borehole as for top sensor and rescale
        combine_weights = np.concatenate([weights_dict[f'{conv1d_name}/kernel:0'], weights_dict[f'{conv1d_name}/kernel:0']], axis=1)
        combine_weights /= 2
        weights_dict[f'{conv1d_name}/kernel:0'] = combine_weights
    if not model_borehole and weights_borehole:
        # Only take weights for the surface sensor and rescale
        combine_weights = weights_dict[f'{conv1d_name}/kernel:0'][:, :32, :]
        combine_weights *= 2
        weights_dict[f'{conv1d_name}/kernel:0'] = combine_weights
    new_weights = []
    transferred = 0
    for i, weight in enumerate(model.weights):
        name = weight.name
        if name in weights_dict:
            new_weights += [weights_dict[name]]
            transferred += 1
        else:
            new_weights += [model.get_weights()[i]]
    print(f'Transferred {transferred} of {len(model.weights)} weights')
    model.set_weights(new_weights)


def generate_weights_dict(weights, name=None):
    weights_dict = {}
    for key in weights.keys():
        if isinstance(weights[key], h5py.Dataset):
            weights_dict[f'{name}/{key}'] = weights[key].value
        else:
            weights_dict.update(generate_weights_dict(weights[key], key))
    return weights_dict


def seed_np_tf(seed=42):
    np.random.seed(seed)
    tf.set_random_seed(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--test_run', action='store_true')  # Test run with less data
    parser.add_argument('--no_multiprocessing', action='store_true')  # Prevents certain deadlocks
    parser.add_argument('--continue_ensemble', action='store_true')  # Continues a stopped ensemble training
    args = parser.parse_args()
    config = json.load(open(args.config, 'r'))

    seed_np_tf(config.get('seed', 42))

    training_params = config['training_params']
    generator_params = training_params.get('generator_params', [training_params.copy()])

    if not os.path.isdir(training_params['weight_path']):
        os.mkdir(training_params['weight_path'])
    listdir = os.listdir(training_params['weight_path'])
    if not args.continue_ensemble and listdir:
        if len(listdir) != 1 or listdir[0] != 'config.json':
            raise ValueError(f'Weight path needs to be empty. ({training_params["weight_path"]})')

    with open(os.path.join(training_params['weight_path'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    print('Loading data')
    if args.test_run:
        limit = 300
    else:
        limit = None

    if not isinstance(training_params['data_path'], list):
        training_params['data_path'] = [training_params['data_path']]

    assert len(generator_params) == len(training_params['data_path'])

    overwrite_sampling_rate = training_params.get('overwrite_sampling_rate', None)

    full_data_train = [loader.load_events(data_path, limit=limit,
                                          parts=(True, False, False),
                                          shuffle_train_dev=generator.get('shuffle_train_dev', False),
                                          custom_split=generator.get('custom_split', None),
                                          min_mag=generator.get('min_mag', None),
                                          mag_key=generator.get('key', 'MA'),
                                          overwrite_sampling_rate=overwrite_sampling_rate,
                                          decimate_events=generator.get('decimate_events', None))
                       for data_path, generator in zip(training_params['data_path'], generator_params)]
    full_data_dev = [loader.load_events(data_path, limit=limit,
                                        parts=(False, True, False),
                                        shuffle_train_dev=generator.get('shuffle_train_dev', False),
                                        custom_split=generator.get('custom_split', None),
                                        min_mag=generator.get('min_mag', None),
                                        mag_key=generator.get('key', 'MA'),
                                        overwrite_sampling_rate=overwrite_sampling_rate,
                                        decimate_events=generator.get('decimate_events', None))
                     for data_path, generator in zip(training_params['data_path'], generator_params)]

    event_metadata_train = [d[0] for d in full_data_train]
    data_train = [d[1] for d in full_data_train]
    metadata_train = [d[2] for d in full_data_train]
    event_metadata_dev = [d[0] for d in full_data_dev]
    data_dev = [d[1] for d in full_data_dev]
    metadata_dev = [d[2] for d in full_data_dev]

    sampling_rate = metadata_train[0]['sampling_rate']
    assert all(m['sampling_rate'] == sampling_rate for m in metadata_train + metadata_dev)
    waveforms = data_train[0]['waveforms']

    max_stations = config['model_params']['max_stations']

    config['model_params']['n_datasets'] = len(data_train)

    ensemble = config.get('ensemble', 1)

    super_config = config.copy()
    super_training_params = training_params.copy()
    super_model_params = config['model_params'].copy()

    for ens_id in range(ensemble):
        if ensemble > 1:
            print(f'Starting ensemble member {ens_id + 1}/{ensemble}')
            seed_np_tf(ens_id)

            config = super_config.copy()
            config['ens_id'] = ens_id
            training_params = super_training_params.copy()
            training_params['weight_path'] = os.path.join(training_params['weight_path'], f'{ens_id}')
            config['training_params'] = training_params
            config['model_params'] = super_model_params.copy()

            if training_params.get('ensemble_rotation', False):
                # Rotated by angles between 0 and pi/4
                config['model_params']['rotation'] = np.pi / 4 * ens_id / (ensemble - 1)

            if args.continue_ensemble and os.path.isdir(training_params['weight_path']):
                hist_path = os.path.join(training_params['weight_path'], 'hist.pkl')
                if os.path.isfile(hist_path):
                    continue
                else:
                    raise ValueError(f'Can not continue unclean ensemble. Checking for {hist_path} failed.')

            if not os.path.isdir(training_params['weight_path']):
                os.mkdir(training_params['weight_path'])

            with open(os.path.join(training_params['weight_path'], 'config.json'), 'w') as f:
                json.dump(config, f, indent=4)

        print('Building model')
        single_station_model, full_model = models.build_transformer_model(**config['model_params'],
                                                                          trace_length=data_train[0]['waveforms'][0].shape[1])

        if 'single_station_model_path' in training_params:
            print('Loading single station model')
            single_station_model.load_weights(training_params['single_station_model_path'])
        elif 'transfer_model_path' not in training_params:
            optimizer = keras.optimizers.Adam(lr=training_params['lr'], clipnorm=training_params['clipnorm'])
            single_station_model.compile(loss=models.mixture_density_loss, optimizer=optimizer)
            key = generator_params[0]['key']
            filter_single_station_by_pick = training_params.get('filter_single_station_by_pick', False)

            x_train = np.concatenate(data_train[0]['waveforms'], axis=0)
            x_dev = np.concatenate(data_dev[0]['waveforms'], axis=0)
            y_train = np.concatenate([np.full(x.shape[0], mag) for x, mag in
                                      zip(data_train[0]['waveforms'], event_metadata_train[0][key])])
            y_dev = np.concatenate([np.full(x.shape[0], mag) for x, mag in
                                    zip(data_dev[0]['waveforms'], event_metadata_dev[0][key])])

            train_mask = (x_train != 0).any(axis=(1, 2))
            dev_mask = (x_dev != 0).any(axis=(1, 2))
            if filter_single_station_by_pick:
                picks_train = np.concatenate(data_train[0]['p_picks'], axis=0)
                train_mask = np.logical_and(train_mask, picks_train < 3000)
                picks_dev = np.concatenate(data_dev[0]['p_picks'], axis=0)
                dev_mask = np.logical_and(dev_mask, picks_dev < 3000)
            x_train = x_train[train_mask]
            y_train = y_train[train_mask]
            x_dev = x_dev[dev_mask]
            y_dev = y_dev[dev_mask]

            cutout = (
                sampling_rate * (5 + generator_params[0]['cutout_start']), sampling_rate * (5 + generator_params[0]['cutout_end']))

            train_generator = util.DataGenerator(x_train, np.expand_dims(np.expand_dims(y_train, axis=1), axis=2),
                                                 batch_size=generator_params[0]['batch_size'], cutout=cutout,
                                                 label_smoothing=True)
            validation_generator = util.DataGenerator(x_dev, np.expand_dims(np.expand_dims(y_dev, axis=1), axis=2),
                                                      batch_size=generator_params[0]['batch_size'], cutout=cutout, oversample=3)

            # Only save weights due to open issue:
            # https://github.com/matterport/Mask_RCNN/issues/308

            filepath = os.path.join(training_params['weight_path'], 'single-station-{epoch:02d}.hdf5')
            checkpoint = ModelCheckpoint(filepath, monitor='val_loss', save_weights_only=True, verbose=1,
                                         save_best_only=True,
                                         mode='min')
            lr_decay = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=4, factor=0.3, verbose=1)

            if 'workers' in training_params:
                workers = training_params['workers']
            else:
                workers = 10
            use_multiprocessing = workers > 1

            single_station_model.fit_generator(generator=train_generator,
                                               validation_data=validation_generator,
                                               epochs=training_params['epochs_single_station'],
                                               use_multiprocessing=use_multiprocessing,
                                               workers=workers,
                                               callbacks=[checkpoint, lr_decay])

            # Free memory
            del x_train
            del x_dev

        if 'load_model_path' in training_params:
            print('Loading full model')
            full_model.load_weights(training_params['load_model_path'])

        if 'transfer_model_path' in training_params:
            print('Transfering model weights')
            ensemble_load = training_params.get('ensemble_load', False)
            wait_for_load = training_params.get('wait_for_load', False)
            transfer_weights(full_model, training_params['transfer_model_path'],
                             ensemble_load=ensemble_load, wait_for_load=wait_for_load, ens_id=ens_id)

        def location_loss(y_true, y_pred):
            return models.mixture_density_loss(y_true, y_pred, eps=1e-5, d=3)

        no_event_token = config['model_params'].get('no_event_token', False)
        optimizer = keras.optimizers.Adam(lr=training_params['lr'], clipnorm=training_params['clipnorm'])
        if not no_event_token:
            losses = {'magnitude': models.mixture_density_loss, 'location': location_loss}
        else:
            losses = {}

        n_pga_targets = config['model_params'].get('n_pga_targets', 0)

        if n_pga_targets:

            def pga_loss(y_true, y_pred):
                return models.time_distributed_loss(y_true, y_pred, models.mixture_density_loss, mean=True,
                                                    kwloss={'mean': False})

            losses['pga'] = pga_loss

        full_model.compile(loss=losses, loss_weights=training_params['loss_weights'], optimizer=optimizer)

        train_generators = []
        validation_generators = []

        for i, generator_param_set in enumerate(generator_params):
            cutout = (
                sampling_rate * (5 + generator_param_set['cutout_start']), sampling_rate * (5 + generator_param_set['cutout_end']))

            generator_param_set['transform_target_only'] = generator_param_set.get('transform_target_only', True)

            train_generators += [util.PreloadedEventGenerator(data=data_train[i],
                                                              event_metadata=event_metadata_train[i],
                                                              coords_target=True,
                                                              label_smoothing=True,
                                                              station_blinding=True,
                                                              cutout=cutout,
                                                              pga_targets=n_pga_targets,
                                                              max_stations=max_stations,
                                                              sampling_rate=sampling_rate,
                                                              no_event_token=no_event_token,
                                                              **generator_param_set)]

            old_oversample = generator_param_set.get('oversample', 1)
            generator_param_set['oversample'] = 4
            validation_generators += [util.PreloadedEventGenerator(data=data_dev[i],
                                                                   event_metadata=event_metadata_dev[i],
                                                                   coords_target=True,
                                                                   station_blinding=True,
                                                                   cutout=cutout,
                                                                   pga_targets=n_pga_targets,
                                                                   max_stations=max_stations,
                                                                   sampling_rate=sampling_rate,
                                                                   no_event_token=no_event_token,
                                                                   **generator_param_set)]
            generator_param_set['oversample'] = old_oversample

        if len(train_generators) == 0:
            train_generator = train_generators[0]
            validation_generator = validation_generators[0]
        else:
            dataset_bias = config['model_params'].get('dataset_bias', False)
            train_generator = util.JointGenerator(train_generators, shuffle=True, dataset_id=dataset_bias)
            validation_generator = util.JointGenerator(validation_generators, shuffle=True, dataset_id=dataset_bias)

        filepath = os.path.join(training_params['weight_path'], 'event-{epoch:02d}.hdf5')
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', save_weights_only=True, verbose=1, save_best_only=True,
                                     mode='min')
        patience = training_params.get('lr_decay_patience', 6)
        lr_decay = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=patience, factor=0.3, verbose=1)
        logdir = os.path.join('logs/scalars/', training_params['weight_path'])
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

        workers = training_params.get('workers', 10)
        use_multiprocessing = workers > 1

        callbacks = [checkpoint, lr_decay]
        if not args.test_run:
            callbacks += [tensorboard_callback]

        if args.test_run or args.no_multiprocessing:
            use_multiprocessing = False
            workers = 1

        hist = full_model.fit_generator(generator=train_generator,
                                        validation_data=validation_generator,
                                        epochs=training_params['epochs_full_model'],
                                        use_multiprocessing=use_multiprocessing,
                                        workers=workers,
                                        callbacks=callbacks)

        pickle.dump(hist.history, open(os.path.join(training_params['weight_path'], 'hist.pkl'), 'wb'))
