import glob
import h5py
import numpy as np
import pandas as pd
import obspy
import os
from obspy import UTCDateTime
import warnings
import time
from tqdm import tqdm
from scipy.stats import norm
try:
    import keras
except ImportError:
    keras = None
    print('Running without keras. Functionality might be limited.')

# warnings.simplefilter("ignore", UserWarning)

D2KM = 111.19492664455874


def resample_trace(trace, sampling_rate):
    if trace.stats.sampling_rate == sampling_rate:
        return
    if trace.stats.sampling_rate % sampling_rate == 0:
        trace.decimate(int(trace.stats.sampling_rate / sampling_rate))
    else:
        trace.resample(sampling_rate)

if keras is not None:
    class DataGenerator(keras.utils.Sequence):
        def __init__(self, X, y, batch_size=32, cutout=None, sliding_window=False, windowlen=3000,
                     shuffle=True, label_smoothing=False, oversample=1):
            self.batch_size = batch_size
            self.y = y
            self.dist = None
            self.X = X
            self.cutout = cutout
            self.sliding_window = sliding_window  # If true, selects sliding windows instead of cutout. Uses cutout as values for end of window.
            self.windowlen = windowlen  # Length of window for sliding window
            assert self.X.shape[0] == self.y.shape[0]
            self.shuffle = shuffle
            self.oversample = oversample
            self.indexes = np.arange(self.X.shape[0])
            self.label_smoothing = label_smoothing
            self.on_epoch_end()

        def __len__(self):
            return int(np.floor(self.oversample * self.X.shape[0] / self.batch_size))

        def __getitem__(self, index):
            # Generate indexes of the batch
            indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

            # Find list of IDs
            X = self.X[indexes, :]
            y = self.y[indexes]

            if self.cutout:
                if self.sliding_window:
                    windowlen = self.windowlen
                    window_end = np.random.randint(max(windowlen, self.cutout[0]), min(X.shape[1], self.cutout[1]) + 1)
                    X = X[:, window_end - windowlen : window_end]
                else:
                    X[:, np.random.randint(*self.cutout):] = 0

            if self.label_smoothing:
                y += (y > 4) * np.random.randn(y.shape[0]).reshape(y.shape) * (y - 4) * 0.05

            return X, y

        def on_epoch_end(self):
            self.indexes = np.repeat(np.arange(self.X.shape[0]), self.oversample)
            if self.shuffle:
                np.random.shuffle(self.indexes)


    class PreloadedEventGenerator(keras.utils.Sequence):
        def __init__(self, data, event_metadata, key='MA', batch_size=32, cutout=None,
                     sliding_window=False, windowlen=3000, shuffle=True,
                     coords_target=True, oversample=1, pos_offset=(-21, -69),
                     label_smoothing=False, station_blinding=False, magnitude_resampling=3,
                     pga_targets=None, adjust_mean=True, transform_target_only=False,
                     max_stations=None, trigger_based=None, min_upsample_magnitude=2,
                     disable_station_foreshadowing=False, selection_skew=None, pga_from_inactive=False,
                     integrate=False, sampling_rate=100.,
                     select_first=False, fake_borehole=False, scale_metadata=True, pga_key='pga',
                     pga_mode=False, p_pick_limit=5000, coord_keys=None, upsample_high_station_events=None,
                     no_event_token=False, pga_selection_skew=None, **kwargs):
            if kwargs:
                print(f'Unused parameters: {", ".join(kwargs.keys())}')
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.waveforms = data['waveforms']
            self.metadata = data['coords']
            self.event_metadata = event_metadata
            if pga_key in data:
                self.pga = data[pga_key]
            else:
                print('Found no PGA values')
                self.pga = [np.zeros(x.shape[0]) for x in self.waveforms]
            self.key = key
            self.cutout = cutout
            self.sliding_window = sliding_window  # If true, selects sliding windows instead of cutout. Uses cutout as values for end of window.
            self.windowlen = windowlen  # Length of window for sliding window
            self.coords_target = coords_target
            self.oversample = oversample
            self.pos_offset = pos_offset
            self.label_smoothing = label_smoothing
            self.station_blinding = station_blinding
            self.magnitude_resampling = magnitude_resampling
            self.pga_targets = pga_targets
            self.adjust_mean = adjust_mean
            self.transform_target_only = transform_target_only
            if max_stations is None:
                max_stations = self.waveforms.shape[1]
            self.max_stations = max_stations
            self.trigger_based = trigger_based
            self.disable_station_foreshadowing = disable_station_foreshadowing
            self.selection_skew = selection_skew
            self.pga_from_inactive = pga_from_inactive
            self.pga_selection_skew = pga_selection_skew
            self.integrate = integrate
            self.sampling_rate = sampling_rate
            self.select_first = select_first
            self.fake_borehole = fake_borehole
            self.scale_metadata = scale_metadata
            self.upsample_high_station_events = upsample_high_station_events
            self.no_event_token = no_event_token

            if 'p_picks' in data:
                self.triggers = data['p_picks']
            else:
                print('Found no picks')
                self.triggers = [np.zeros(x.shape[0]) for x in self.waveforms]

            # Extend samples to include all pga targets in each epoch
            # PGA mode is only for evaluation, as it adds zero padding to the input/pga target!
            self.pga_mode = pga_mode
            self.p_pick_limit = p_pick_limit

            self.base_indexes = np.arange(len(self.waveforms))
            self.reverse_index = None
            if magnitude_resampling > 1:
                magnitude = self.event_metadata[key].values
                for i in np.arange(min_upsample_magnitude, 9):
                    ind = np.where(np.logical_and(i < magnitude, magnitude <= i + 1))[0]
                    self.base_indexes = np.concatenate(
                        (self.base_indexes, np.repeat(ind, int(magnitude_resampling ** (i - 1) - 1))))

            if self.upsample_high_station_events is not None:
                new_indexes = []
                for ind in self.base_indexes:
                    n_stations = self.waveforms[ind].shape[0]
                    new_indexes += [ind for _ in range(n_stations // self.upsample_high_station_events + 1)]
                self.base_indexes = np.array(new_indexes)

            if pga_mode:
                new_base_indexes = []
                self.reverse_index = []
                c = 0
                for idx in self.base_indexes:
                    # This produces an issue if there are 0 pga targets for an event.
                    # As all input stations are always pga targets as well, this should not occur.
                    #
                    num_samples = (len(self.pga[idx]) - 1) // pga_targets + 1
                    new_base_indexes += [(idx, i) for i in range(num_samples)]
                    self.reverse_index += [c]
                    c += num_samples
                self.reverse_index += [c]
                self.base_indexes = new_base_indexes

            self.indexes = np.arange(len(self.waveforms))
            if coord_keys is None:
                self.coord_keys = detect_location_keys(event_metadata.columns)
            else:
                self.coord_keys = coord_keys

            self.on_epoch_end()

        def __len__(self):
            return int(np.ceil(self.indexes.shape[0] / self.batch_size))

        def __getitem__(self, index):
            # Generate indexes of the batch
            indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
            true_batch_size = len(indexes)
            if self.pga_mode:
                pga_indexes = [x[1] for x in indexes]
                indexes = [x[0] for x in indexes]

            waveforms = np.zeros((true_batch_size, self.max_stations) + self.waveforms[0].shape[1:])
            true_max_stations_in_batch = max(max([self.metadata[idx].shape[0] for idx in indexes]), self.max_stations)
            metadata = np.zeros((true_batch_size, true_max_stations_in_batch) + self.metadata[0].shape[1:])
            pga = np.zeros((true_batch_size, true_max_stations_in_batch))
            full_p_picks = np.zeros((true_batch_size, true_max_stations_in_batch))
            p_picks = np.zeros((true_batch_size, self.max_stations))
            reverse_selections = []

            # Find list of IDs
            for i, idx in enumerate(indexes):
                if len(self.waveforms[idx]) <= self.max_stations:
                    waveforms[i, :len(self.waveforms[idx])] = self.waveforms[idx]
                    metadata[i, :len(self.metadata[idx])] = self.metadata[idx]
                    pga[i, :len(self.pga[idx])] = self.pga[idx]
                    p_picks[i, :len(self.triggers[idx])] = self.triggers[idx]
                    reverse_selections += [[]]
                else:
                    if self.selection_skew is None:
                        selection = np.arange(0, len(self.waveforms[idx]))
                        np.random.shuffle(selection)
                    else:
                        tmp_p_picks = self.triggers[idx].copy()
                        mask = np.logical_and(tmp_p_picks <= 0, tmp_p_picks > self.p_pick_limit)
                        tmp_p_picks[mask] = min(np.max(tmp_p_picks), self.p_pick_limit)
                        coeffs = np.exp(-tmp_p_picks / self.selection_skew)
                        coeffs *= np.random.random(coeffs.shape)
                        coeffs[self.triggers[idx] == 0] = 0
                        coeffs[self.triggers[idx] > self.waveforms[0].shape[1]] = 0
                        selection = np.argsort(-coeffs)

                    if self.select_first:
                        selection = np.argsort(self.triggers[idx])

                    metadata[i, :len(selection)] = self.metadata[idx][selection]
                    pga[i, :len(selection)] = self.pga[idx][selection]
                    full_p_picks[i, :len(selection)] = self.triggers[idx][selection]

                    tmp_reverse_selection = [0 for _ in selection]
                    for j, s in enumerate(selection):
                        tmp_reverse_selection[s] = j
                    reverse_selections += [tmp_reverse_selection]

                    selection = selection[:self.max_stations]
                    waveforms[i] = self.waveforms[idx][selection]
                    p_picks[i] = self.triggers[idx][selection]

            magnitude = self.event_metadata.iloc[indexes][self.key].values.copy()

            target = None
            if self.coords_target:
                target = self.event_metadata.iloc[indexes][self.coord_keys].values

            org_waveform_length = waveforms.shape[2]
            if self.cutout:
                if self.sliding_window:
                    windowlen = self.windowlen
                    window_end = np.random.randint(max(windowlen, self.cutout[0]), min(waveforms.shape[2], self.cutout[1]) + 1)
                    waveforms = waveforms[:, :, window_end - windowlen: window_end]

                    cutout = window_end
                    if self.adjust_mean:
                        waveforms -= np.mean(waveforms, axis=2, keepdims=True)
                else:
                    cutout = np.random.randint(*self.cutout)
                    if self.adjust_mean:
                        waveforms -= np.mean(waveforms[:, :, :cutout+1], axis=2, keepdims=True)
                    waveforms[:, :, cutout:] = 0
            else:
                cutout = waveforms.shape[2]

            if self.trigger_based:
                # Remove waveforms for all stations that did not trigger yet to avoid knowledge leakage
                p_picks[p_picks <= 0] = org_waveform_length  # Ensure that stations without P picks do not show data
                waveforms[cutout < p_picks, :, :] = 0

            if self.integrate:
                waveforms = np.cumsum(waveforms, axis=2) / self.sampling_rate

            # Reshape magnitude to match dimension number of MDN output
            magnitude = np.expand_dims(np.expand_dims(magnitude, axis=-1), axis=-1)
            # Center location on mean of locations
            if self.coords_target:
                metadata, target = self.location_transformation(metadata, target)
            else:
                metadata = self.location_transformation(metadata)

            if self.label_smoothing:
                magnitude += (magnitude > 4) * np.random.randn(magnitude.shape[0]).reshape(magnitude.shape) * (
                        magnitude - 4) * 0.05

            if not self.pga_from_inactive and not self.pga_mode:
                metadata = metadata[:, :self.max_stations]
                pga = pga[:, :self.max_stations]

            if self.pga_targets:
                pga_values = np.zeros(
                    (true_batch_size, self.pga_targets))
                pga_targets = np.zeros((true_batch_size, self.pga_targets, 3))
                if self.pga_mode:
                    for i in range(waveforms.shape[0]):
                        pga_index = pga_indexes[i]
                        if len(reverse_selections[i]) > 0:
                            sorted_pga = pga[i, reverse_selections[i]]
                            sorted_metadata = metadata[i, reverse_selections[i]]
                        else:
                            sorted_pga = pga[i]
                            sorted_metadata = metadata[i]
                        pga_values_pre = sorted_pga[pga_index * self.pga_targets:(pga_index + 1) * self.pga_targets]
                        pga_values[i, :len(pga_values_pre)] = pga_values_pre
                        pga_targets_pre = sorted_metadata[pga_index * self.pga_targets:(pga_index + 1) * self.pga_targets, :]
                        if pga_targets_pre.shape[-1] == 4:
                            pga_targets_pre = pga_targets_pre[:, (0, 1, 3)]
                        pga_targets[i, :len(pga_targets_pre), :] = pga_targets_pre
                else:
                    pga[np.logical_or(np.isnan(pga), np.isinf(pga))] = 0  # Ensure only legal PGA values are selected
                    for i in range(waveforms.shape[0]):
                        active = np.where(pga[i] != 0)[0]
                        if len(active) == 0:
                            raise ValueError(f'Found event without PGA idx={indexes[i]}')
                        while len(active) < self.pga_targets:
                            active = np.repeat(active, 2)
                        if self.pga_selection_skew is not None:
                            active_p_picks = full_p_picks[i, active]
                            mask = np.logical_and(active_p_picks <= 0, active_p_picks > self.p_pick_limit)
                            active_p_picks[mask] = min(np.max(active_p_picks), self.p_pick_limit)
                            coeffs = np.exp(-active_p_picks / self.pga_selection_skew)
                            coeffs *= np.random.random(coeffs.shape)
                            active = active[np.argsort(-coeffs)]
                        else:
                            np.random.shuffle(active)

                        samples = active[:self.pga_targets]
                        if metadata.shape[-1] == 3:
                            pga_targets[i] = metadata[i, samples, :]
                        else:
                            full_targets = metadata[i, samples]
                            pga_targets[i] = full_targets[:, (0, 1, 3)]
                        pga_values[i] = pga[i, samples]
                # Last two dimensions to match shape for keras loss
                pga_values = pga_values.reshape((true_batch_size, self.pga_targets, 1, 1))

            metadata = metadata[:, :self.max_stations]

            if self.station_blinding:
                mask = np.zeros(waveforms.shape[:2], dtype=bool)

                for i in range(waveforms.shape[0]):
                    active = np.where((waveforms[i] != 0).any(axis=(1, 2)))[0]
                    if len(active) == 0:
                        active = np.zeros(1, dtype=int)
                    blind_length = np.random.randint(0, len(active))
                    np.random.shuffle(active)
                    blind = active[:blind_length]
                    mask[i, blind] = True

                waveforms[mask] = 0
                metadata[mask] = 0

            # To avoid that stations without a trigger are masked, we can set a value in the waveforms to non-zero.
            # Thereby we keep the information that the station did not trigger yet.
            # On the other hand this might leak information that a station is still going to trigger
            stations_without_trigger = (metadata != 0).any(axis=2) & (waveforms == 0).all(axis=(2, 3))
            if self.disable_station_foreshadowing:
                metadata[stations_without_trigger] = 0
            else:
                waveforms[stations_without_trigger, 0, 0] += 1e-9

            # Avoid completely zero events, leading to NaN values in energy loss
            mask = np.logical_and((metadata == 0).all(axis=(1, 2)), (waveforms == 0).all(axis=(1, 2, 3)))
            waveforms[mask, 0, 0, 0] = 1e-9
            metadata[mask, 0, 0] = 1e-9

            if self.fake_borehole and waveforms.shape[3] == 3:
                waveforms = np.concatenate([np.zeros_like(waveforms), waveforms], axis=3)
                metadata_new = np.zeros(metadata.shape[:-1] + (4,))
                metadata_new[:, :, 0] = metadata[:, :, 0]
                metadata_new[:, :, 1] = metadata[:, :, 1]
                metadata_new[:, :, 3] = metadata[:, :, 2]
                metadata = metadata_new

            inputs = [waveforms, metadata]
            outputs = []
            if not self.no_event_token:
                outputs += [magnitude]

                if self.coords_target:
                    target = np.expand_dims(target, axis=-1)
                    outputs += [target]

            if self.pga_targets:
                inputs += [pga_targets]
                outputs += [pga_values]

            return inputs, outputs

        def on_epoch_end(self):
            self.indexes = np.repeat(self.base_indexes.copy(), self.oversample, axis=0)
            if self.shuffle:
                np.random.shuffle(self.indexes)

        def location_transformation(self, metadata, target=None):
            transform_target_only = self.transform_target_only
            metadata = metadata.copy()

            metadata_old = metadata
            metadata = metadata.copy()
            mask = (metadata == 0).all(axis=2)
            if target is not None:
                target[:, 0] -= self.pos_offset[0]
                target[:, 1] -= self.pos_offset[1]
            metadata[:, :, 0] -= self.pos_offset[0]
            metadata[:, :, 1] -= self.pos_offset[1]

            # Coordinates to kilometers (assuming a flat earth, which is okay close to equator)
            if self.scale_metadata:
                metadata[:, :, :2] *= D2KM
            if target is not None:
                target[:, :2] *= D2KM

            metadata[mask] = 0

            if self.scale_metadata:
                metadata /= 100
            if target is not None:
                target /= 100

            if transform_target_only:
                metadata = metadata_old

            if target is None:
                return metadata
            else:
                return metadata, target


    class JointGenerator(keras.utils.Sequence):
        def __init__(self, generators=(), shuffle=True, dataset_id=False, fake_id=None):
            assert len(generators)
            self.generators = generators
            self.indexes = None
            self.shuffle = shuffle
            self.dataset_id = dataset_id
            self.fake_id = fake_id
            self.on_epoch_end()

        def __len__(self):
            return sum(len(generator) for generator in self.generators)

        def __getitem__(self, index):
            generator_id, batch_id = self.indexes[index]
            batch_inp, batch_out = self.generators[generator_id][batch_id]
            if self.dataset_id:
                if self.fake_id is None:
                    dataset_id = np.ones((batch_inp[0].shape[0], 1)) * generator_id
                else:
                    dataset_id = np.ones((batch_inp[0].shape[0], 1)) * self.fake_id
                batch_inp += [dataset_id]
            return batch_inp, batch_out

        def on_epoch_end(self):
            self.indexes = []
            for i, generator in enumerate(self.generators):
                self.indexes += [(i, j) for j in range(len(generator))]
            if self.shuffle:
                np.random.shuffle(self.indexes)


    class CutoutGenerator(keras.utils.Sequence):
        def __init__(self, generator, times, sampling_rate):
            self.generator = generator
            self.times = times
            self.sampling_rate = sampling_rate
            self.indexes = None
            self.on_epoch_end()

        def __len__(self):
            return len(self.generator) * len(self.times)

        def __getitem__(self, index):
            time, batch_id = self.indexes[index]
            cutout = int(self.sampling_rate * (time + 5))
            self.generator.cutout = (cutout, cutout + 1)
            return self.generator[batch_id]

        def on_epoch_end(self):
            self.indexes = []
            for time in self.times:
                self.indexes += [(time, i) for i in range(len(self.generator))]


    def gaussian_mixture(x, params, eps=1e-6, memory_save=True, fortran=True):
        if fortran and not np.isfortran(params):
            params = np.array(params, order='F')
        if not memory_save:
            return gaussian_mixture_fully_vectorized(x, params, eps)
        if params.ndim == 2:
            alpha = params[:, 0]
            mu = params[:, 1]
            sigma = np.maximum(params[:, 2], eps)
            density = np.zeros_like(x)
            for i in range(alpha.shape[0]):
                density += alpha[i] * 1 / (np.sqrt(2 * np.pi) * sigma[i]) * np.exp(-(x - mu[i]) ** 2 / (2 * sigma[i] ** 2))
        elif params.ndim == 3:
            alpha = np.expand_dims(params[:, :, 0], axis=0)
            mu = np.expand_dims(params[:, :, 1], axis=0)
            sigma = np.expand_dims(np.maximum(params[:, :, 2], eps), axis=0)
            x = np.reshape(x, (-1, 1))
            density = np.zeros((x.shape[0], alpha.shape[1]))
            for i in range(alpha.shape[2]):
                density += alpha[:, :, i] * 1 / (np.sqrt(2 * np.pi) * sigma[:, :, i]) * np.exp(-(x - mu[:, :, i]) ** 2 / (2 * sigma[:, :, i] ** 2))
            density = density.T
        else:
            raise ValueError('Params ndim must be 2 or 3')

        return density


def gaussian_mixture_fully_vectorized(x, params, eps=1e-6):
    if params.ndim == 2:
        alpha = np.reshape(params[:, 0], (1, -1))
        mu = np.reshape(params[:, 1], (1, -1))
        sigma = np.reshape(np.maximum(params[:, 2], eps), (1, -1))
        x = np.reshape(x, (-1, 1))
        density = alpha * 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
        density = np.sum(density, axis=1)
    elif params.ndim == 3:
        alpha = np.expand_dims(params[:, :, 0], axis=0)
        mu = np.expand_dims(params[:, :, 1], axis=0)
        sigma = np.expand_dims(np.maximum(params[:, :, 2], eps), axis=0)
        x = np.reshape(x, (-1, 1, 1))
        density = alpha * 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
        density = np.sum(density, axis=2).T
    else:
        raise ValueError('Params ndim must be 2 or 3')

    return density


def filter_shard(events, shard_id, shards):
    if not shards:
        return events
    if isinstance(events, pd.DataFrame):
        events = events.iloc[shard_id::shards]
    else:
        return events[shard_id::shards]
    return events


def merge_hdf5(inputs, output, event_id, sort_key=None):
    inputs = sorted(glob.glob(inputs))
    print(inputs)
    metadata = {}
    delete_keys = []
    existing_keys = []
    catalog = None
    with h5py.File(output, 'w') as fout:
        gout_data = fout.create_group('data')
        gout_meta = fout.create_group('metadata')
        for inp in tqdm(inputs):
            with h5py.File(inp, 'r') as fin:
                for key in fin['metadata'].keys():
                    if key == 'event_metadata':
                        continue
                    tmp = fin['metadata'][key].value
                    if key in metadata:
                        if isinstance(metadata[key] == tmp, (bool, np.bool_, np.bool)):
                            assert metadata[key] == tmp
                        else:
                            assert all(metadata[key] == tmp)
                    metadata[key] = tmp

                for event in fin['data']:
                    existing_keys += [event]
                    if event in gout_data:
                        delete_keys += [event]
                        del gout_data[event]
                        continue
                    gout_event = gout_data.create_group(event)
                    for ds in fin['data'][event]:
                        gout_event.create_dataset(ds, data=fin['data'][event][ds].value)

            if catalog is None:
                catalog = pd.read_hdf(inp, 'metadata/event_metadata')
            else:
                catalog = pd.concat([catalog, pd.read_hdf(inp, 'metadata/event_metadata')])

        for key, val in metadata.items():
            gout_meta.create_dataset(key, data=val)

    if sort_key is None:
        sort_key = event_id

    catalog = catalog.sort_values(by=sort_key)
    catalog = catalog[~catalog[event_id].isin(delete_keys)]
    catalog = catalog[catalog[event_id].isin(existing_keys)]
    print(f'Removed {len(delete_keys)} redundant events')
    catalog.to_hdf(output, key='metadata/event_metadata', mode='a', encoding='utf-8', format='table')


def detect_location_keys(columns):
    candidates = [['LAT', 'Latitude(°)', 'Latitude'],
                  ['LON', 'Longitude(°)', 'Longitude'],
                  ['DEPTH', 'JMA_Depth(km)', 'Depth(km)', 'Depth/Km']]

    coord_keys = []
    for keyset in candidates:
        for key in keyset:
            if key in columns:
                coord_keys += [key]
                break

    if len(coord_keys) != len(candidates):
        raise ValueError('Unknown location key format')

    return coord_keys


def wait_for_file(path, sleep_seconds=600, silent=False):
    while not os.path.exists(path):
        if not silent:
            print(f'File {path} for weight transfer missing. Sleeping for {sleep_seconds} seconds.')
        time.sleep(sleep_seconds)


def normalize_gain(stream, inv, key='ACC'):
    delete_traces = []
    for trace in stream:
        try:
            resp = inv.get_response(trace.id, trace.stats.starttime)
        except:
            # get_response throws plain Exception, therefore bare except is required
            print(f'Missing response for {trace.id}')
            delete_traces += [trace]
            continue
        try:
            sensitivity = np.abs(resp.get_evalresp_response_for_frequencies([1.], key))[0]
            reported_sensitivity = resp.instrument_sensitivity.value
            if np.abs(sensitivity - reported_sensitivity) / max(sensitivity, reported_sensitivity) > 0.05:
                print(f'Sensitivity mismatch for station {trace.stats.network}.{trace.stats.station}. '
                      f'Reported: {reported_sensitivity}\tComputed: {sensitivity}\tFactor: {reported_sensitivity / sensitivity}')
                sensitivity = reported_sensitivity
            trace.data = trace.data / sensitivity
        except ValueError:
            # Illegal RESP format
            trace.data = trace.data * 0.0

    for trace in delete_traces:
        stream.remove(trace)


def generator_from_config(config, data, event_metadata, time, batch_size=64, sampling_rate=100, pga=False, dataset_id=None):
    training_params = config['training_params']

    if dataset_id is not None:
        generator_params = training_params.get('generator_params', [training_params.copy()])[dataset_id]
    else:
        generator_params = training_params.get('generator_params', [training_params.copy()])[0]

    noise_seconds = generator_params[0].get('noise_seconds', 5)
    cutout = int(sampling_rate * (noise_seconds + time))
    cutout = (cutout, cutout + 1)

    n_pga_targets = config['model_params'].get('n_pga_targets', 0)
    max_stations = config['model_params']['max_stations']
    generator_params['magnitude_resampling'] = 1
    generator_params['batch_size'] = batch_size
    generator_params['transform_target_only'] = generator_params.get('transform_target_only', True)
    generator_params['upsample_high_station_events'] = None
    if generator_params.get('coord_keys', None) is not None:
        raise NotImplementedError('Fixed coordinate keys are not implemented in location evaluation')
    generator_params['translate'] = False
    generator = PreloadedEventGenerator(data=data,
                                        event_metadata=event_metadata,
                                        coords_target=True,
                                        cutout=cutout,
                                        pga_targets=n_pga_targets,
                                        max_stations=max_stations,
                                        sampling_rate=sampling_rate,
                                        select_first=True,
                                        shuffle=False,
                                        pga_mode=pga,
                                        **generator_params)
    if dataset_id is not None and config['model_params'].get('dataset_bias', False):
        generator = JointGenerator([generator], shuffle=False, dataset_id=True, fake_id=dataset_id)
    return generator


def bin_search_quantile(pred, quantile, vmin=0, vmax=10):
    min_val = vmin * np.ones((pred.shape[0], 1))
    max_val = vmax * np.ones((pred.shape[0], 1))

    for _ in range(14):
        # 14 iterations mean approximation error is below 0.001
        mean_val = (min_val + max_val) / 2
        prob = np.sum(
            pred[:, :, 0] * (1 - norm.cdf((mean_val - pred[:, :, 1]) / pred[:, :, 2])),
            axis=-1, keepdims=True)
        min_val = np.where(prob > quantile, mean_val, min_val)
        max_val = np.where(prob <= quantile, mean_val, max_val)

    pred = np.squeeze(min_val, axis=-1)

    return pred