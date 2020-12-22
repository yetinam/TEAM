import numpy as np
import pandas as pd
import h5py


class TrainDevTestSplitter:
    @staticmethod
    def run_method(event_metadata, name, shuffle_train_dev, parts):
        mask = np.zeros(len(event_metadata), dtype=bool)

        split_methods = {'test_2016': TrainDevTestSplitter.test_2016,
                         'test_2011': TrainDevTestSplitter.test_2011,
                         'no_test': TrainDevTestSplitter.no_test}

        if name is None or name == '':
            test_set = TrainDevTestSplitter.default(event_metadata)
        elif name in split_methods:
            test_set = split_methods[name](event_metadata)
        else:
            raise ValueError(f'Unknown split function: {name}')

        b1 = int(0.6 / 0.7 * np.sum(~test_set))
        train_set = np.zeros(np.sum(~test_set), dtype=bool)
        train_set[:b1] = True

        if shuffle_train_dev:
            np.random.seed(len(event_metadata))  # The same length of data always gets split the same way
            np.random.shuffle(train_set)

        if parts[0] and parts[1]:
            mask[~test_set] = True
        elif parts[0]:
            mask[~test_set] = train_set
        elif parts[1]:
            mask[~test_set] = ~train_set
        if parts[2]:
            mask[test_set] = True

        return mask

    @staticmethod
    def default(event_metadata):
        test_set = np.zeros(len(event_metadata), dtype=bool)
        b2 = int(0.7 * len(event_metadata))
        test_set[b2:] = True
        return test_set

    @staticmethod
    def test_2016(event_metadata):
        test_set = np.array([x[:4] == '2016' for x in event_metadata['Time']])
        return test_set

    @staticmethod
    def test_2011(event_metadata):
        test_set = np.array([x[:4] == '2011' for x in event_metadata['Origin_Time(JST)']])
        return test_set


    @staticmethod
    def no_test(event_metadata):
        return np.zeros(len(event_metadata), dtype=bool)


def load_events(data_paths, limit=None, parts=None, shuffle_train_dev=False, custom_split=None, data_keys=None,
                overwrite_sampling_rate=None, min_mag=None, mag_key=None, decimate_events=None):
    if min_mag is not None and mag_key is None:
        raise ValueError('mag_key needs to be set to enforce magnitude threshold')
    if isinstance(data_paths, str):
        data_paths = [data_paths]
    if len(data_paths) > 1:
        raise NotImplementedError('Loading partitioned data is currently not supported')
    data_path = data_paths[0]

    event_metadata = pd.read_hdf(data_path, 'metadata/event_metadata')
    if min_mag is not None:
        event_metadata = event_metadata[event_metadata[mag_key] >= min_mag]
    for event_key in ['KiK_File', '#EventID', 'EVENT']:
        if event_key in event_metadata.columns:
            break

    if limit:
        event_metadata = event_metadata.iloc[:limit]
    if parts:
        mask = TrainDevTestSplitter.run_method(event_metadata, custom_split, shuffle_train_dev, parts=parts)
        event_metadata = event_metadata[mask]

    if decimate_events is not None:
        event_metadata = event_metadata.iloc[::decimate_events]

    metadata = {}
    data = {}

    with h5py.File(data_path, 'r') as f:
        for key in f['metadata'].keys():
            if key == 'event_metadata':
                continue
            metadata[key] = f['metadata'][key].value

        if overwrite_sampling_rate is not None:
            if metadata['sampling_rate'] % overwrite_sampling_rate != 0:
                raise ValueError(f'Overwrite sampling ({overwrite_sampling_rate}) rate must be true divisor of sampling'
                                 f' rate ({metadata["sampling_rate"]})')
            decimate = metadata['sampling_rate'] // overwrite_sampling_rate
            metadata['sampling_rate'] = overwrite_sampling_rate
        else:
            decimate = 1

        skipped = 0
        contained = []
        for _, event in event_metadata.iterrows():
            event_name = str(event[event_key])
            if event_name not in f['data']:
                skipped += 1
                contained += [False]
                continue
            contained += [True]
            g_event = f['data'][event_name]
            for key in g_event:
                if data_keys is not None and key not in data_keys:
                    continue
                if key not in data:
                    data[key] = []
                if key == 'waveforms':
                    data[key] += [g_event[key][:, ::decimate, :]]
                else:
                    data[key] += [g_event[key].value]
                if key == 'p_picks':
                    data[key][-1] //= decimate

        data_length = None
        for val in data.values():
            if data_length is None:
                data_length = len(val)
            assert len(val) == data_length

        if len(contained) < len(event_metadata):
            contained += [True for _ in range(len(event_metadata) - len(contained))]
        event_metadata = event_metadata[contained]
        if skipped > 0:
            print(f'Skipped {skipped} events')

    return event_metadata, data, metadata
