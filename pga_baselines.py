import numpy as np
from geopy.distance import geodesic
from tqdm import tqdm
import argparse
import json
import os
import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression
from collections import defaultdict
from scipy import stats

from mag_baselines import KuyukAllen
import loader
import util


def plum(event_metadata, data, pga_thresholds, training_data, radius=15, alpha=None):
    pga_times = data['pga_times']
    coords = data['coords']

    pred_times = []
    for ev_times, ev_coords in zip(pga_times, tqdm(coords)):
        if np.isnan(ev_times).all():
            pred_times += [np.expand_dims(ev_times.copy(), axis=-1)]
            continue
        dist_matrix = calc_dist_matrix(ev_coords)
        ev_pred_times = 1e6 * np.ones_like(ev_times)
        for i in range(ev_times.shape[0]):
            for j in range(ev_times.shape[1]):
                if not np.isnan(ev_times[i, j]):
                    neighbors = dist_matrix[i] <= radius
                    ev_pred_times[neighbors, j] = np.minimum(ev_pred_times[neighbors, j], ev_times[i, j])
        ev_pred_times[ev_pred_times >= 1e6] = np.nan
        pred_times += [np.expand_dims(ev_pred_times, axis=-1)]
    return pred_times


def estimated_point_source(event_metadata, data, pga_thresholds, training_data, region=None,
                           start_time=1, end_time=25, dt=0.1, alpha=(0.3, 0.4, 0.5, 0.6, 0.7),
                           **kwargs):
    if region is None:
        raise ValueError("Region must be set")

    pga_thresholds = np.log10(9.81 * pga_thresholds)

    waveforms = data['waveforms']
    picks = data['p_picks']
    coords = data['coords']
    stations = data['stations']

    mag_estimator = KuyukAllen(offset=region, **kwargs)
    print('Calibrating GMPE')
    gmpe = calibrate_gmpe(*training_data, region)
    mag_key = gmpe.mag_key

    pred_full = []

    print('Predicting')
    pred_iter = tqdm(zip(waveforms, picks, coords, stations, event_metadata.iterrows()), total=len(waveforms))
    for ev_waveforms, ev_picks, ev_coords, ev_stations, (_, event) in pred_iter:
        times = np.arange(start_time, end_time, dt)
        pred = mag_estimator.predict(times, ev_waveforms, ev_picks, ev_coords, event)
        mean_pred = np.sum((pred * mag_estimator.magnitude_buckets), axis=-1)

        dists = calc_station_event_dist(ev_coords, event, single=True)

        dummy_metadata = []
        for mag in mean_pred:
            tmp_event = event.copy()
            tmp_event[mag_key] = mag
            dummy_metadata += [tmp_event]

        dummy_metadata = pd.DataFrame(dummy_metadata)

        pga_pred = gmpe.predict(event_metadata=dummy_metadata,
                                dists=len(dummy_metadata) * [dists],
                                stations=len(dummy_metadata) * [ev_stations],
                                alpha=alpha)

        times = np.pad(times, (1, 0), mode='constant')  # Zero pad to have no warning option
        times = times.astype(int)

        pga_pred = [-10 * np.ones_like(pga_pred[0])] + pga_pred  # Pad pga preds with value not exeeding threshold

        pga_pred = np.concatenate([np.expand_dims(x, axis=0) for x in pga_pred])  # time, station

        ev_time_pred = np.zeros((ev_picks.shape[0], len(pga_thresholds), len(alpha)), dtype=float)
        for j, _ in enumerate(alpha):
            for i, level in enumerate(pga_thresholds):
                warning = np.argmax(pga_pred[:, :, j] > level, axis=0)
                warning = times[warning]
                ev_time_pred[:, i, j] = warning
        ev_time_pred[ev_time_pred == 0] = np.nan
        pred_full += [ev_time_pred]

    return pred_full


def true_point_source(event_metadata, data, pga_thresholds, training_data, region=None, alpha=(0.3, 0.4, 0.5, 0.6, 0.7)):
    """
    Estimate warnings based on a GMPE with the true source parameters
    Assume source is fully known at the moment of the first P arrival
    """
    if region is None:
        raise ValueError("Region must be set")

    pga_thresholds = np.log10(9.81 * pga_thresholds)

    coords = data['coords']
    stations = data['stations']
    picks = data['p_picks']

    print('Calibrating GMPE')
    gmpe = calibrate_gmpe(*training_data, region)

    print('Calculating dists')
    dists = calc_station_event_dist(coords, event_metadata)
    pga_pred = gmpe.predict(event_metadata=event_metadata, dists=dists, stations=stations, alpha=alpha)

    print('Predicting')
    pred_full = []
    pred_iter = tqdm(zip(picks, pga_pred), total=len(picks))
    for ev_picks, ev_pga in pred_iter:
        ev_time_pred = np.zeros((ev_picks.shape[0], len(pga_thresholds), len(alpha)), dtype=float)
        for j, _ in enumerate(alpha):
            for i, level in enumerate(pga_thresholds):
                ev_time_pred[ev_pga[:, j] > level, i, j] = 1
        ev_time_pred[ev_time_pred == 0] = np.nan
        ev_time_pred -= 1
        pred_full += [ev_time_pred]

    return pred_full


def calibrate_gmpe(event_metadata, coords, pgas, stations, region):
    if region == 'italy':
        mag_key = 'Magnitude'
    elif region == 'japan':
        mag_key = 'M_J'
    else:
        raise ValueError(f'Magnitude key for region {region} unknown')

    dists = calc_station_event_dist(coords, event_metadata)

    gmpe = GMPECuaHeaton(mag_key=mag_key, params={'c1': 1.48, 'c2': 1.11}, region=region)
    gmpe.fit(pgas, dists, event_metadata, stations, iterations=10)
    return gmpe


def calc_station_event_dist(coords, event_metadata, single=False):
    dists = []
    if single:
        coord_keys = util.detect_location_keys(event_metadata.keys())
        event_iter = zip([coords], [(None, event_metadata)])
    else:
        coord_keys = util.detect_location_keys(event_metadata.columns)
        event_iter = zip(coords, event_metadata.iterrows())

    for coord, (_, event) in event_iter:
        ev_lat, ev_lon, ev_depth = event[coord_keys]
        dist = np.ones(coord.shape[0])
        for i, station_coords in enumerate(coord):
            dist[i] = geodesic(station_coords[:2], [ev_lat, ev_lon]).km
        dists += [dist]

    if single:
        return dists[0]
    else:
        return dists


def calc_dist_matrix(coords):
    dist = np.zeros((coords.shape[0], coords.shape[0]))
    for i in range(coords.shape[0]):
        for j in range(i):
            dist[i, j] = geodesic(coords[i, :2], coords[j, :2]).km
    dist = dist + dist.T  # Calculate only lower triangle explicitly
    return dist


def convert_pga_times(data, metadata):
    new_pga_times = []
    for pga_times in data['pga_times']:
        pga_times = pga_times.astype(float)
        pga_times[pga_times == 0] = np.nan
        pga_times[~np.isnan(pga_times)] = pga_times[~np.isnan(pga_times)] / metadata['sampling_rate'] - metadata['time_before']
        new_pga_times += [pga_times]
    data['pga_times'] = new_pga_times


class GMPECuaHeaton:
    def __init__(self, mag_key='Magnitude', params=None, region='italy', surpress_warnings=False):
        self.mag_key = mag_key
        if params is None:
            self.params = {'a1': 1.7788,
                           'a2': 0.1074,
                           'b': 0.0006,
                           'c1': 1.0966,
                           'c2': -0.1149,
                           'd': -1.6543,
                           'e': -4.1808}
        else:
            self.params = params.copy()
        self.region = region
        self.model = None
        self.sigma = None
        self.station_bias = {}
        self.surpress_warnings = surpress_warnings

    def save(self, path):
        data = {'mag_key': self.mag_key,
                'params': self.params,
                'region': self.region,
                'station_bias': self.station_bias,
                'sigma': self.sigma}
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.mag_key = data['mag_key']
        self.params = data['params']
        self.region = data['region']
        self.station_bias = data['station_bias']
        self.sigma = data['sigma']

    def fit(self, pgas, dists, event_metadata, stations=None, iterations=1):
        if iterations > 1 and stations is None:
            raise ValueError("Can't run method iteratively without station correction")
        if stations is None:
            self.station_bias = None

        coord_keys = util.detect_location_keys(event_metadata.columns)

        mag_list = [np.ones_like(dist) * mag for dist, mag in zip(dists, event_metadata[self.mag_key])]
        depth_list = [np.ones_like(dist) * depth for dist, depth in zip(dists, event_metadata[coord_keys[2]])]

        r = np.concatenate(dists)
        mag = np.concatenate(mag_list)
        depth = np.concatenate(depth_list)
        stations = np.concatenate(stations)

        if self.region == 'japan':
            h_d = np.where(depth < 20, 5, 40)
            h_d = np.where(depth > 200, depth, h_d)
            r = np.sqrt(r ** 2 + h_d ** 2)
        elif self.region == 'italy':
            h_d = np.where(depth < 20, 5, 50)
            r = np.sqrt(r ** 2 + h_d ** 2)

        cm = self.params['c1'] * np.exp(self.params['c2'] * np.maximum(0, mag - 5)) * (np.arctan(mag - 5) + np.pi / 2)
        rcm = r + cm

        if self.region == 'japan':
            m2 = np.maximum(0, mag - 6) ** 2
        elif self.region == 'italy':
            m2 = np.maximum(0, mag - 4) ** 2
        m2 = m2.reshape(-1, 1)

        mag = mag.reshape(-1, 1)
        rcm = rcm.reshape(-1, 1)
        predictors = np.concatenate([mag, m2, np.minimum(rcm, 5000), np.log10(rcm)], axis=1)
        target = np.concatenate(pgas)

        if self.region == 'japan':
            mask = r < np.maximum(0, mag[:, 0] - 3.5) * 200
            mask = np.logical_and(mask, r > 20)
        elif self.region == 'italy':
            mask = r < np.maximum(0, mag[:, 0] - 3.0) * 50
            mask = np.logical_and(mask, r > 5)
            mask = np.logical_and(mask, target < 1.7)  # Delete faulty points

        mask = np.logical_and(mask, ~np.isnan(target))
        mask = np.logical_and(mask, ~np.isinf(target))
        mask = np.logical_and(mask, ~(np.isnan(predictors).any(axis=1)))
        mask = np.logical_and(mask, ~(np.isinf(predictors).any(axis=1)))

        target = target[mask]
        predictors = predictors[mask]
        stations = stations[mask]

        corr = np.zeros_like(target)

        for i in range(iterations):
            print(f'Iteration {i+1}')

            self.model = LinearRegression()
            self.model.fit(predictors, target - corr)
            
            pred = self.model.predict(predictors)
            diff = target - corr - pred
            self.sigma = np.sqrt(np.mean(diff ** 2))
            print(f'RMSE: {self.sigma}')
            
            if stations is not None:
                self.update_station_bias(predictors, target, stations)
                corr = np.array([self.station_bias[station] for station in stations])

        self.params['a1'] = self.model.coef_[0]
        self.params['a2'] = self.model.coef_[1]
        self.params['b'] = self.model.coef_[2]
        self.params['d'] = self.model.coef_[3]
        self.params['e'] = self.model.intercept_
    
    def predict(self, dists, event_metadata, stations=None, alpha=None):
        if stations is None and self.station_bias is not None and not self.surpress_warnings:
            print('Warning: station information missing')

        coord_keys = util.detect_location_keys(event_metadata.columns)

        mag_list = [np.ones_like(dist) * mag for dist, mag in zip(dists, event_metadata[self.mag_key])]
        depth_list = [np.ones_like(dist) * depth for dist, depth in zip(dists, event_metadata[coord_keys[2]])]

        r = np.concatenate(dists)
        mag = np.concatenate(mag_list)
        depth = np.concatenate(depth_list)

        if self.region == 'japan':
            h_d = np.where(depth < 20, 5, 40)
            h_d = np.where(depth > 200, depth, h_d)
            r = np.sqrt(r ** 2 + h_d ** 2)
        elif self.region == 'italy':
            h_d = np.where(depth < 20, 5, 50)
            r = np.sqrt(r ** 2 + h_d ** 2)

        cm = self.params['c1'] * np.exp(self.params['c2'] * np.maximum(0, mag - 5)) * (np.arctan(mag - 5) + np.pi / 2)
        rcm = r + cm

        if self.region == 'japan':
            m2 = np.maximum(0, mag - 6) ** 2
        elif self.region == 'italy':
            m2 = np.maximum(0, mag - 4) ** 2

        pred_full = self.params['a1'] * mag + self.params['a2'] * m2 + self.params['b'] * np.minimum(rcm, 5000) + self.params['d'] * np.log10(rcm) + self.params['e']

        if stations is not None and self.station_bias is not None:
            corr = []
            for station in np.concatenate(stations):
                if station in self.station_bias:
                    corr += [self.station_bias[station]]
                else:
                    corr += [0]
            corr = np.array(corr)
            pred_full += corr

        if self.region == 'italy':
            pred_full = np.maximum(pred_full, -4)

        if alpha is not None:
            pred_full = pred_full.reshape(pred_full.shape + (1,)).repeat(len(alpha), axis=-1)

            for i, alp in enumerate(alpha):
                pred_full[:, i] += stats.norm.ppf(alp) * self.sigma

        cuts = np.cumsum([0] + [x.shape[0] for x in dists])
        pga_pred = []
        for start, end in zip(cuts[:-1], cuts[1:]):
            pga_pred += [pred_full[start:end]]

        return pga_pred

    def update_station_bias(self, predictors, target, stations):
        pred = self.model.predict(predictors)
        diff = target - pred

        station_diffs = defaultdict(list)

        for station, d in zip(stations, diff):
            station_diffs[station] += [d]

        for station, bias in station_diffs.items():
            self.station_bias[station] = np.mean(bias)

    def print_params(self):
        for param, val in self.params.items():
            print(f'{param}\t{val}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--test_run', action='store_true')  # Test run with less data
    args = parser.parse_args()
    config = json.load(open(args.config, 'r'))
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
    training_keys = config.get('training_keys', None)
    training_parts = config.get('training_parts', (True, False, False))

    for act_set in ['dev', 'test']:
        event_metadata, data, metadata = loader.load_events(config['data_path'], limit=limit,
                                                            parts=(False, act_set == 'dev', act_set == 'test'),
                                                            shuffle_train_dev=shuffle_train_dev,
                                                            custom_split=custom_split, data_keys=data_keys)
        if training_keys is not None:
            event_metadata_train, data_train, metadata_train = loader.load_events(
                config['data_path'], limit=limit,
                parts=training_parts,
                shuffle_train_dev=shuffle_train_dev,
                custom_split=custom_split, data_keys=training_keys)
            training_data = [event_metadata_train] + [data_train[key] for key in training_keys]
        else:
            training_data = None

        pga_thresholds = metadata['pga_thresholds']
        key = config.get('magnitude_key', 'M_J')
        if key not in event_metadata.columns:
            raise ValueError(f'Magnitude key {key} not in event metadata')
        coord_keys = util.detect_location_keys(event_metadata.columns)
        convert_pga_times(data, metadata)

        for method, method_args in zip(config['methods'], config['method_args']):
            print(f'STARTING:\t{method}\t{act_set}')
            pred_times = globals()[method](event_metadata, data, pga_thresholds, training_data, **method_args)
            full_predictions = []

            for tmp_pred_times, tmp_true_times, tmp_coords, (_, event) in zip(pred_times,
                                                                              data['pga_times'],
                                                                              data['coords'],
                                                                              event_metadata.iterrows()):
                coords_event = event[coord_keys]

                dist = np.zeros(tmp_coords.shape[0])
                for j, station_coords in enumerate(tmp_coords):
                    dist[j] = geodesic(station_coords[:2], coords_event[:2]).km
                dist = np.sqrt(dist ** 2 + coords_event[2] ** 2)  # Epi- to hypocentral distance

                full_predictions += [(tmp_pred_times, tmp_true_times, dist)]

            output_dir = os.path.join(config['output_path'], method, act_set)
            os.makedirs(output_dir, exist_ok=True)
            alpha = method_args.get('alpha', (0.3, 0.4, 0.5, 0.6, 0.7))
            with open(os.path.join(output_dir, 'predictions.pkl'), 'wb') as pred_file:
                pickle.dump(full_predictions, pred_file)
