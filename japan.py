import pandas as pd
import numpy as np
import getpass
from obspy.geodetics import gps2dist_azimuth
from tqdm import tqdm
import time
import argparse
from obspy import UTCDateTime
import os
import tarfile
import h5py
import obspy
from pyrocko import cake
from collections import defaultdict
from geopy.distance import geodesic
import requests

from util import filter_shard, merge_hdf5

traveltime_model = cake.load_model('prem-no-ocean.m')
pick_offset = 16


def generate_pick(coords_station, coords_event, phase_key, event_time):
    dist = gps2dist_azimuth(lat1=coords_station[0], lat2=coords_event[0], lon1=coords_station[1], lon2=coords_event[1])[0]

    phase = cake.PhaseDef(phase_key.lower())
    arrivals = traveltime_model.arrivals([dist * cake.m2d], phases=phase, zstart=coords_event[2] * 1e3, zstop=max(-coords_station[2] * 1e3, 0))

    if not arrivals:
        phase = cake.PhaseDef(phase_key.upper())
        arrivals = traveltime_model.arrivals([dist * cake.m2d], phases=phase, zstart=coords_event[2] * 1e3, zstop=max(-coords_station[2] * 1e3, 0))

    if not arrivals:
        coords_event_new = coords_event.copy()
        coords_event_new[2] += 0.1
        return generate_pick(coords_station, coords_event_new, phase_key, event_time)

    return event_time + pick_offset + arrivals[0].t


def get_user_pass(user=None, password=None):
    if user is None:
        print('Login credentials for NEID required')
        user = input('User: ')
        password = getpass.getpass()
    elif user is not None and password is None:
        print('Login credentials for NEID required')
        print(f'User: {user}')
        password = getpass.getpass()
    return user, password


def download_kiknet_events(kiknet_files, output_dir, sleep_between_calls=1.0):
    base_url = 'https://www.kyoshin.bosai.go.jp/kyoshin/pubdata/'
    kiknet_files = [line.strip() for line in open(kiknet_files, 'r') if line.strip()]

    with requests.session() as session:
        session.auth = get_user_pass()

        downloaded = 0
        skipped = 0
        file_iterator = tqdm(kiknet_files, desc=f'Skipped: {skipped:5d}\tDownloaded: {downloaded:5d}')
        for file in file_iterator:
            url = f'{base_url}kik/alldata/{file[:4]}/{file[4:6]}/{file}'
            output_path = os.path.join(output_dir, os.path.basename(file))
            if os.path.isfile(output_path):
                skipped += 1
            else:
                status_code = 0
                while status_code != 200:
                    r = session.get(url)
                    status_code = r.status_code
                    if status_code == 401:
                        print('User/password seems to be incorrect. Please reenter:')
                        session.auth = get_user_pass()
                    elif status_code != 200:
                        raise requests.HTTPError('Download failed with HTTP code {status_code}')

                with open(output_path, 'wb') as fout:
                    fout.write(r.content)
                downloaded += 1
                time.sleep(sleep_between_calls)

            file_iterator.set_description(f'Skipped: {skipped:5d}\tDownloaded: {downloaded:5d}')


def resample_trace(trace, old_sampling_rate, target_sampling_rate):
    if old_sampling_rate == target_sampling_rate:
        return trace
    if int(old_sampling_rate) % int(target_sampling_rate) == 0:
        s = int(old_sampling_rate) // int(target_sampling_rate)
        return trace[::s]
    raise NotImplementedError(f'Can not resample from {old_sampling_rate} to {target_sampling_rate}')


def parse_channel(path, demean=True, rem_resp=True, rewrite_params=True, tarball=None):
    metadata = {'channel': path[-3:]}

    def time_rewrite(x):
        x = x.replace(' ', ',')
        return x + '.00'

    rewrites = (('Station Code', 'station', str),
                ('Station Lat.', 'lat', float),
                ('Station Long.', 'lon', float),
                ('Station Height(m)', 'depth', lambda x: float(x)/1000),
                ('Sampling Freq(Hz)', 'sampling_rate', lambda x: float(x[:-2])),
                ('Duration Time(s)', 'duration', float),
                ('Record Time', 'start_time', lambda x: UTCDateTime.strptime(x, '%Y/%m/%d %H:%M:%S')),
                ('Mag.', 'M_J', float),
                ('Lat.', 'Latitude(°)', float),
                ('Long.', 'Longitude(°)', float),
                ('Depth. (km)', 'Depth(km)', float),
                ('Origin Time', 'Origin_Time(JST)', time_rewrite))

    del_params = ['Dir.', 'Last Correction', 'Max. Acc. (gal)']

    def open_channel_file():
        if tarball is None:
            return open(path, 'rb')
        else:
            return tarball.extractfile(path)

    with open_channel_file() as f:
        for line in f:
            line = line.decode()
            if line.split()[0].strip() == 'Memo.':
                # data starts here
                break
            key = line[:18].strip()
            val = line[18:].strip()
            metadata[key] = val
        waveforms = []
        for line in f:
            line = line.strip().split()
            if not line:
                continue
            waveforms += [int(v) for v in line]
    waveforms = np.array(waveforms, dtype='float64')
    if demean:
        waveforms -= np.mean(waveforms)
    if rem_resp:
        scale_string = metadata['Scale Factor']
        p1 = scale_string.find('(')
        p2 = scale_string.find('/')
        scale = float(scale_string[:p1]) / float(scale_string[p2+1:])
        scale /= 100.  # Scale from gal to m/(s**2)
        waveforms *= scale
        metadata['scale'] = scale
        del metadata['Scale Factor']
    if rewrite_params:
        for old, new, fnc in rewrites:
            metadata[new] = fnc(metadata[old])
            del metadata[old]
        for param in del_params:
            del metadata[param]
    return metadata, np.array(waveforms)


def parse_station(path, channels=('NS1', 'EW1', 'UD1', 'NS2', 'EW2', 'UD2'), sampling_rate=100, tarball=None):
    metadata = {}
    waveforms = None
    for i, channel in enumerate(channels):
        tmp_metadata, tmp_waveforms = parse_channel(path + '.' + channel, tarball=tarball)
        del tmp_metadata['channel']
        tmp_waveforms = resample_trace(tmp_waveforms, tmp_metadata['sampling_rate'], sampling_rate)
        tmp_metadata['sampling_rate'] = sampling_rate
        if not metadata:
            metadata = tmp_metadata.copy()
            del metadata['scale']
            del metadata['depth']
            waveforms = np.zeros(tmp_waveforms.shape + (len(channels),))
        for key in tmp_metadata.keys():
            if key in ('scale', 'depth'):
                continue
            if metadata[key] != tmp_metadata[key]:
                raise ValueError(f'Record time mismatch for key {key}')
        c = channel[-1]
        metadata[f'scale{c}'] = tmp_metadata['scale']
        metadata[f'depth{c}'] = tmp_metadata['depth']
        waveforms[:, i] = tmp_waveforms
    return metadata, waveforms


def translate_channel(channel):
    if channel[0] == 'U':
        return 'Z'
    else:
        return channel[0]


def parse_event(path, channels=('NS1', 'EW1', 'UD1', 'NS2', 'EW2', 'UD2'), sampling_rate=100, to_obspy=False):
    tarball = None
    if path[-7:] == '.tar.gz':
        tarball = tarfile.open(path, 'r')
        stations_paths = [x.name[:-4] for x in tarball.getmembers() if x.name[-4:] == '.EW1']
    else:
        stations_paths = [os.path.join(path, x[:-4]) for x in os.listdir(path) if x[-4:] == '.EW1']
    waveforms = None
    event_metadata = {}
    for i, station in enumerate(stations_paths):
        tmp_metadata, tmp_waveforms = parse_station(station, channels, sampling_rate=sampling_rate, tarball=tarball)
        if waveforms is None:
            waveforms = np.zeros((len(stations_paths),) + tmp_waveforms.shape)
            stations = np.zeros((len(stations_paths),), dtype='S10')
            coords = np.zeros((len(stations_paths), 4))
            start_times = np.zeros(len(stations_paths), dtype=int)
            scales = np.zeros((len(stations_paths), 2))
        assert sampling_rate == tmp_metadata['sampling_rate']
        if tmp_waveforms.shape[0] > waveforms.shape[1]:
            diff = tmp_waveforms.shape[0] - waveforms.shape[1]
            waveforms = np.pad(waveforms, ((0, 0), (0, diff), (0, 0)), mode='constant', constant_values=0)
        waveforms[i, :tmp_waveforms.shape[0]] = tmp_waveforms
        stations[i] = tmp_metadata['station']
        coords[i] = [tmp_metadata['lat'], tmp_metadata['lon'], tmp_metadata['depth1'], tmp_metadata['depth2']]
        scales[i] = [tmp_metadata['scale1'], tmp_metadata['scale2']]
        start_times[i] = tmp_metadata['start_time'].timestamp

    for metadata_key in ['Latitude(°)', 'Longitude(°)', 'Depth(km)', 'M_J', 'Origin_Time(JST)']:
        event_metadata[metadata_key] = tmp_metadata[metadata_key]

    if not to_obspy:
        return waveforms, stations, coords, scales, start_times, event_metadata
    else:
        translated_channels = [translate_channel(c) for c in channels]

        traces = []
        for i, _ in enumerate(waveforms):
            base_stats = {'sampling_rate': sampling_rate,
                          'station': stations[i].decode(),
                          'starttime': UTCDateTime(start_times[i])}
            for j in range(6):
                stats = base_stats.copy()
                if j < 3:
                    stats['station'] += '_B'  # Borehole sensor
                stats['channel'] = f'HL{translated_channels[j]}'
                trace = obspy.core.trace.Trace(data=waveforms[i, :, j], header=stats)
                traces += [trace]
        stream = obspy.core.stream.Stream(traces)
        return stream, coords, scales, stations


def select_ground_motion_window(dist, diff, event_mag, sampling_rate=100):
    start_idx = (pick_offset - 5) * sampling_rate + diff
    # Second term approximates P travel time to project back to source time
    # Third term approximates Rayleigh wave speed
    end_idx = int(start_idx - (dist / 8 * sampling_rate) + (dist / 2.8 * sampling_rate) + 10 * sampling_rate)
    # Account roughly for longer event durations
    if event_mag > 6:
        end_idx += 5 * sampling_rate
    if event_mag > 7:
        end_idx += 15 * sampling_rate
    if event_mag > 8:
        end_idx += 80 * sampling_rate
    return start_idx, end_idx


def get_dists(coords, coords_event):
    dist = np.zeros(coords.shape[0])
    for j, station_coords in enumerate(coords):
        dist[j] = geodesic(station_coords[:2], coords_event[:2]).km
    dist = np.sqrt(dist ** 2 + coords_event[2] ** 2)  # Epi- to hypocentral distance

    return dist


def align_and_extract(waveforms, stations, coords, scales, start_times, coords_event, event_time, event_mag,
                      sampling_rate=100, time_before=5, time_after=25,
                      pga_thresholds=(0.01, 0.02, 0.05, 0.1, 0.2),
                      pick_validity_check=False, validity_w=4, validity_limit=5):
    p_picks = start_times.copy()

    if pick_validity_check:
        predicted_p_picks = np.zeros(len(coords))
        for i, _ in enumerate(coords):
            predicted_p_picks[i] = generate_pick(coords[i, (0, 1, 2)], coords_event, 'p', event_time)

        diff = p_picks - predicted_p_picks
        diff = diff - np.mean(diff)
        sort_diff = np.sort(diff)

        w = validity_w

        if len(sort_diff) < 2 * w:
            diff_limit = sort_diff[0]
            if len(sort_diff) > 2 and sort_diff[0] < sort_diff[2] - 3000:
                diff_limit = sort_diff[2] - 5

        else:
            if np.min(sort_diff[w:] - sort_diff[:-w]) > validity_limit:
                diff_limit = sort_diff[0]
            else:
                idx = np.argmax(sort_diff[w:] - sort_diff[:-w] < validity_limit)
                diff_limit = sort_diff[idx] - 5

        if np.any(diff < diff_limit):
            print(f'Rewriting {np.sum(diff < diff_limit)} picks due to validity check')
            # Shift affected picks to match with pick limit
            p_picks = (p_picks - np.minimum(diff - diff_limit, 0)).astype(int)

    start_times = (start_times - np.min(p_picks)) * sampling_rate
    p_picks = (p_picks - np.min(p_picks)) * sampling_rate

    pga = np.zeros(waveforms.shape[0])
    pgv = np.zeros(waveforms.shape[0])
    pga_times = np.zeros((waveforms.shape[0], len(pga_thresholds)))

    dists = get_dists(coords, coords_event)

    tmp_waveforms = np.cumsum(waveforms, axis=1) / sampling_rate

    for j in range(waveforms.shape[0]):
        start_idx, end_idx = select_ground_motion_window(dist=dists[j], diff=(p_picks - start_times)[j],
                                                         event_mag=event_mag, sampling_rate=sampling_rate)

        hor_acc = np.sqrt(waveforms[j, start_idx:end_idx, 3] ** 2 + waveforms[j, start_idx:end_idx, 4] ** 2)
        pga[j] = np.max(hor_acc)
        pgv[j] = np.max(
            np.sqrt(tmp_waveforms[j, start_idx:end_idx, 3] ** 2 + tmp_waveforms[j, start_idx:end_idx, 4] ** 2))

        # PGA Threshold are given as fractions of g
        if pga_thresholds is not None:
            for i, pre_threshold in enumerate(pga_thresholds):
                threshold = 9.81 * pre_threshold
                hor_acc = np.pad(hor_acc, (1, 0), 'constant')
                pga_times[j, i] = np.argmax(hor_acc > threshold)  # Outputs index of the first exceedance and 0 if it is never exceeded
            pga_times[j, pga_times[j, :] == 0] = np.nan
            pga_times[j, :] -= 1
            pga_times[j, :] += start_idx
        else:
            pga_times = None

    pga = np.log10(pga)
    pgv = np.log10(pgv)

    samples_before = sampling_rate * time_before
    samples_after = sampling_rate * time_after
    offset = 15 * sampling_rate

    aligned_waveforms = np.zeros((waveforms.shape[0], samples_before + samples_after, waveforms.shape[2]))

    if pga_times is not None:
        pga_times = pga_times - offset + samples_before + start_times.reshape((-1, 1))

    for i in range(waveforms.shape[0]):
        left = max(offset - samples_before - start_times[i], 0)
        right = offset + samples_after - start_times[i]
        right_pad = 0  # Catch case that the start of the data is available, but not the end
        if right > waveforms.shape[1]:
            right_pad = right - waveforms.shape[1]
            right = waveforms.shape[1]
        right = max(right, 0)
        data = waveforms[i, left:right]
        if left == right:
            data = np.zeros((1, waveforms.shape[-1]))
        aligned_waveforms[i, -data.shape[0] - right_pad:samples_before + samples_after - right_pad] = data

    p_picks += samples_before
    return aligned_waveforms, stations, coords, p_picks, pga, pga_times, pgv, scales


def extract_events(input_path, output, channels=('NS1', 'EW1', 'UD1', 'NS2', 'EW2', 'UD2'),
                   sampling_rate=100, time_before=5, time_after=25, shards=None, shard_id=None,
                   pga_thresholds=(0.01, 0.02, 0.05, 0.1, 0.2), pick_validity_check=True, validity_w=4, validity_limit=5):
    events = sorted([x[:-11] for x in os.listdir(input_path) if x[-11:] == '.kik.tar.gz'])
    events = filter_shard(events, shard_id, shards)

    if shard_id is not None:
        output = f'{output[:-5]}_{shard_id}.hdf5'

    with h5py.File(output, 'w') as output_file:
        g_data = output_file.create_group('data')
        g_meta = output_file.create_group('metadata')
        g_meta.create_dataset('channels', (len(channels),), 'S4', np.array(channels, dtype='S4'))
        g_meta.create_dataset('sampling_rate', data=sampling_rate)
        g_meta.create_dataset('time_before', data=time_before)
        g_meta.create_dataset('time_after', data=time_after)
        g_meta.create_dataset('pga_thresholds', data=pga_thresholds)
        if pick_validity_check:
            g_meta.create_dataset('pick_validity_check', data=True)
            g_meta.create_dataset('validity_w', data=validity_w)
            g_meta.create_dataset('validity_limit', data=validity_limit)

        event_iterator = tqdm(events)
        catalog_prior = defaultdict(list)

        for event in event_iterator:
            event_name = event

            if event_name in g_data:
                del g_data[event_name]
                print(f'Duplicate events removed {event_name}')
                continue

            path = os.path.join(input_path, event_name + '.kik.tar.gz')
            event_data = parse_event(path, channels=channels, sampling_rate=sampling_rate)
            event_metadata = event_data[-1]

            for dict_key, dict_val in event_metadata.items():
                catalog_prior[dict_key] += [dict_val]
            catalog_prior['KiK_File'] += [event]

            coords_event = [event_metadata['Latitude(°)'],
                            event_metadata['Longitude(°)'],
                            event_metadata['Depth(km)']]
            event_time = UTCDateTime.strptime(event_metadata['Origin_Time(JST)'], '%Y/%m/%d,%H:%M:%S.%f').timestamp

            waveforms, stations, coords, p_picks, pga, pga_times, pgv, scales = \
                align_and_extract(*event_data[:-1],
                                  event_mag=event_metadata['M_J'],
                                  coords_event=coords_event,
                                  event_time=event_time,
                                  sampling_rate=sampling_rate,
                                  time_before=time_before,
                                  time_after=time_after,
                                  pga_thresholds=pga_thresholds,
                                  pick_validity_check=pick_validity_check,
                                  validity_w=validity_w,
                                  validity_limit=validity_limit)
            if waveforms is not None:
                g_event = g_data.create_group(event_name)
                g_event.create_dataset('waveforms', data=waveforms)
                g_event.create_dataset('stations', stations.shape, stations.dtype, stations)
                g_event.create_dataset('coords', data=coords)
                g_event.create_dataset('p_picks', data=p_picks)
                g_event.create_dataset('pga', data=pga)
                g_event.create_dataset('pga_times', data=pga_times)
                g_event.create_dataset('pgv', data=pgv)
                g_event.create_dataset('scales', data=scales)

    catalog = pd.DataFrame(catalog_prior)
    catalog.to_hdf(output, key='metadata/event_metadata', mode='a')


def verify_args(args, params, action):
    for param in params:
        if getattr(args, param) is None:
            raise ValueError(f'{param} needs to be set for action {action}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', type=str, required=True)
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--catalog', type=str)
    parser.add_argument('--shards', type=int)
    parser.add_argument('--shard_id', type=int)
    parser.add_argument('--no_pick_validity_check', action='store_false')
    args = parser.parse_args()
    action = args.action

    if action == 'download_events':
        verify_args(args, ['output', 'catalog'], action)
        download_kiknet_events(kiknet_files=args.catalog, output_dir=args.output)

    if action == 'extract_events':
        verify_args(args, ['output', 'input'], action)
        extract_events(args.input, args.output,
                       shards=args.shards, shard_id=args.shard_id, pick_validity_check=args.no_pick_validity_check)

    if action == 'merge_hdf5':
        verify_args(args, ['input', 'output'], action)
        merge_hdf5(args.input, args.output, 'KiK_File')
