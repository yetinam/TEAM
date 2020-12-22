import argparse
import os
import json
import numpy as np
from scipy import signal
from geopy.distance import geodesic
import pickle
from pyrocko import cake
from tqdm import tqdm
import pandas as pd

import loader
import util


traveltime_model = cake.load_model('prem-no-ocean.m')


class KuyukAllen:
    """
    Early Magnitude estimation
    Based on: Kuyuk, Huseyin Serdar, and Richard M. Allen. "A global approach to provide magnitude estimates for earthquake early warning alerts." Geophysical research letters 40.24 (2013): 6329-6333.
    """
    def __init__(self, sampling_rate=100, sigma=0.31, length_weight=False, z_idx=2, snr_threshold=3, p_slack=1,
                 s_slack=0.5, min_window=3, time_before=5, pd_length=4, offset=0, velocity=False):
        self.sampling_rate = sampling_rate
        self.sigma = sigma
        self.length_weight = length_weight
        self.z_idx = z_idx
        self.snr_threshold = snr_threshold
        self.p_slack = p_slack
        self.s_slack = s_slack
        self.min_window = min_window
        self.time_before = time_before
        self.pd_length = pd_length
        self.velocity = velocity
        if offset == 'italy':
            self.offset = 0.3
        elif offset == 'japan':
            self.offset = 0.5
        else:
            self.offset = 0

        # Fixed and derived parameters
        self.coeff = [1.23, 1.38, 5.39]
        self.magnitude_buckets = np.arange(0, 10, 0.1)
        self.sos = signal.butter(3, (0.5, 3), 'bandpass', output='sos', fs=sampling_rate)
        self.norm_const = 1 / np.sqrt(2 * np.pi * sigma ** 2)

    def aggregate(self, predictions):
        return np.sum((predictions * self.magnitude_buckets), axis=-1)

    def predict_full(self, times, event_metadata, data):
        full_predictions = []
        for (_, event), waveforms, picks, coords in tqdm(zip(event_metadata.iterrows(), data['waveforms'], data['p_picks'],
                                                             data['coords']), total=len(event_metadata)):
            full_predictions += [self.predict(times, waveforms, picks, coords, event)]
        return np.array(full_predictions)

    def predict(self, times, waveforms, picks, coords, event):
        """
        Predict magnitude for one event at multiple times
        Calculate multiple times at once as the expensive steps (filtering, distance caluculation) are identical for all times
        """
        ztraces = waveforms[:, :, self.z_idx]
        if not self.velocity:
            ztraces = np.cumsum(ztraces, axis=1)
            ztraces = signal.detrend(ztraces, axis=1)
            ztraces -= np.mean(ztraces[:, :400], axis=1, keepdims=True)
            ztraces /= self.sampling_rate
        ztraces = np.cumsum(ztraces, axis=1)
        ztraces /= self.sampling_rate
        ztraces = signal.sosfilt(self.sos, ztraces, axis=1)

        idxs = []
        for i, (trace, pick) in enumerate(zip(ztraces, picks)):
            if pick >= len(trace) or pick == 0:
                continue
            val = np.max(np.abs(ztraces[i, pick:pick + 4 * self.sampling_rate]))
            noise = np.max(np.abs(ztraces[i, pick - 2 * self.sampling_rate:pick - self.sampling_rate]))
            if noise * self.snr_threshold > val:
                continue
            idxs += [i]

        if not idxs:
            return np.nan * np.ones((len(times), self.magnitude_buckets.shape[0]))

        ztraces = ztraces[idxs]
        picks = picks[idxs]
        coords = coords[idxs]

        coord_keys = util.detect_location_keys(event.keys())
        ev_coord = event[coord_keys]

        s_picks = np.zeros_like(picks)
        for i, (st_coord, pick) in enumerate(zip(coords, picks)):
            p_pred = generate_pick(st_coord, ev_coord, 'p')
            s_pred = generate_pick(st_coord, ev_coord, 's')
            s_picks[i] = int(pick + (s_pred - p_pred) * self.sampling_rate)

        dists = np.ones(picks.shape[0])
        for i, coord in enumerate(coords):
            dists[i] = geodesic(coord[:2], ev_coord[:2]).km

        full_pred = np.ones((len(times), self.magnitude_buckets.shape[0]))
        for j, t in enumerate(times):
            t = (t + self.time_before) * self.sampling_rate
            t_pred = np.ones((ztraces.shape[0], self.magnitude_buckets.shape[0]))

            for i, (trace, p_pick, s_pick) in enumerate(zip(ztraces, picks, s_picks)):
                p0 = int(p_pick - self.sampling_rate * self.p_slack)
                p1 = int(min(p_pick + self.pd_length * self.sampling_rate, s_pick - self.s_slack * self.sampling_rate, t))
                if p1 - p0 < self.min_window * self.sampling_rate:
                    continue
                if p_pick == 0:
                    continue

                pd = np.log10(np.max(np.abs(trace[p0:p1]))) + 2
                c1, c2, c3 = self.coeff
                mag = c1 * pd + c2 * np.log10(dists[i]) + c3 + self.offset

                t_pred[i] = self.norm_const * np.exp(-((mag - self.magnitude_buckets) / self.sigma) ** 2)

                if self.length_weight:
                    t_pred[i] **= ((p1 - p0) / self.sampling_rate - self.min_window + 0.5)

            full_pred[j] = np.prod(t_pred, axis=0)

        full_pred /= np.sum(full_pred, axis=1, keepdims=True)
        full_pred[(full_pred == 1 / self.magnitude_buckets.shape[0]).all(axis=1)] = np.nan

        return full_pred


def generate_pick(coords_station, coords_event, phase_key):
    dist = geodesic(coords_station[:2], coords_event[:2]).m

    phase = cake.PhaseDef(phase_key.lower())
    arrivals = traveltime_model.arrivals([dist * cake.m2d], phases=phase, zstart=max(coords_event[2] * 1e3, 0), zstop=max(-coords_station[2] * 1e3, 0))

    if not arrivals:
        phase = cake.PhaseDef(phase_key.upper())
        arrivals = traveltime_model.arrivals([dist * cake.m2d], phases=phase, zstart=max(coords_event[2] * 1e3, 0), zstop=max(-coords_station[2] * 1e3, 0))

    if not arrivals:
        coords_event_new = list(coords_event)
        coords_event_new[2] += 0.1
        return generate_pick(coords_station, coords_event_new, phase_key)

    return arrivals[0].t


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--test_run', action='store_true')  # Test run with less data
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = json.load(f)
    os.makedirs(config['output_path'], exist_ok=True)
    with open(os.path.join(config['output_path'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    if args.test_run:
        limit = 300
    else:
        limit = None

    shuffle_train_dev = config.get('shuffle_train_dev', False)
    custom_split = config.get('custom_split', None)
    data_keys = config.get('data_keys', None)
    times = config.get('times', (0.5, 1, 2, 4, 8, 16, 25))

    for act_set in ['dev', 'test']:
        event_metadata, data, metadata = loader.load_events(config['data_path'], limit=limit,
                                                            parts=(False, act_set == 'dev', act_set == 'test'),
                                                            shuffle_train_dev=shuffle_train_dev,
                                                            custom_split=custom_split, data_keys=data_keys)

        if 'additional_data' in config:
            event_metadata_add, data_add, _ = loader.load_events(config['additional_data'],
                                                                 parts=(True, True, True))
            event_metadata = pd.concat([event_metadata, event_metadata_add])
            for t_key in data.keys():
                if t_key in data_add:
                    data[t_key] += data_add[t_key]

        for method, method_args in zip(config['methods'], config['method_args']):
            print(f'STARTING:\t{method}\t{act_set}')

            predictor = globals()[method](**method_args)
            full_predictions = predictor.predict_full(times, event_metadata, data)
            agg_predictions = predictor.aggregate(full_predictions)

            output_dir = os.path.join(config['output_path'], method, act_set)
            os.makedirs(output_dir, exist_ok=True)

            output_data = (times, agg_predictions, full_predictions)

            with open(os.path.join(output_dir, 'predictions.pkl'), 'wb') as pred_file:
                pickle.dump(output_data, pred_file)
