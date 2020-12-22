import numpy as np
import pandas as pd
import argparse
import h5py
import cartopy.io.shapereader as shpreader
import shapely.geometry as sgeom
from shapely.ops import unary_union
from shapely.prepared import prep
from tqdm import tqdm


land_shp_fname = shpreader.natural_earth(resolution='50m',
                               category='physical', name='land')

land_geom = unary_union(list(shpreader.Reader(land_shp_fname).geometries()))
land = prep(land_geom)


def is_land(x, y):
   return land.contains(sgeom.Point(x, y))


def filter_events(event_metadata, min_mag, max_depth):
    mask = event_metadata['Depth(km)'] <= max_depth
    mask = np.logical_and(mask, event_metadata['M_J'] >= min_mag)
    mask = mask.values

    lons = event_metadata['Longitude(°)'].values[mask]
    lats = event_metadata['Latitude(°)'].values[mask]

    eq_on_land = [is_land(lon, lat) for lon, lat in zip(lons, lats)]
    mask[mask] = eq_on_land

    return event_metadata[mask]


def relocate_to_italy(coords, event):
    event = event.copy()
    coords = coords[:, :3].copy()

    v_nnw = np.array([0.75, -0.5])
    v_ene = np.array([0.5, 0.75])

    scale_nnw = 0.4
    scale_ene = 0.15

    base_lat_shift = - event['Latitude(°)'] + 42.6
    base_lon_shift = - event['Longitude(°)'] + 13.3

    shift = np.array([base_lat_shift, base_lon_shift])
    shift += np.random.randn() * scale_nnw * v_nnw
    shift += np.random.randn() * scale_ene * v_ene

    coords[:, :2] += shift
    event['Latitude(°)'] += shift[0]
    event['Longitude(°)'] += shift[1]

    return coords, event


def transfer_metadata(fin, fout, additional):
    gout_meta = fout.create_group('metadata')

    for key in fin['metadata'].keys():
        if key == 'event_metadata':
            continue
        val = fin['metadata'][key].value
        gout_meta.create_dataset(key, data=val)

    for key, val in additional.items():
        gout_meta.create_dataset(key, data=val)


def transfer_data(fin, fout, event_metadata):
    gout_data = fout.create_group('data')

    new_events = []

    for _, event in tqdm(event_metadata.iterrows(), total=len(event_metadata)):
        event_str = str(event['KiK_File'])
        gout_event = gout_data.create_group(event_str)
        for ds in fin['data'][event_str]:
            val = fin['data'][event_str][ds].value
            if ds == 'coords':
                val, event = relocate_to_italy(val, event)
                new_events += [event]
            if ds == 'waveforms':
                val = val[:, :, 3:]
            gout_event.create_dataset(ds, data=val)

    return pd.DataFrame(new_events)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--min_mag', type=float, default=5)
    parser.add_argument('--max_depth', type=float, default=10)
    args = parser.parse_args()

    event_metadata = pd.read_hdf(args.input, 'metadata/event_metadata')

    event_metadata = filter_events(event_metadata, min_mag=args.min_mag, max_depth=args.max_depth)

    with h5py.File(args.input, 'r') as fin:
        with h5py.File(args.output, 'w') as fout:
            transfer_metadata(fin, fout, {'min_mag': args.min_mag,
                                          'max_depth': args.max_depth,
                                          'source_file': args.input})
            event_metadata = transfer_data(fin, fout, event_metadata)

    event_metadata.to_hdf(args.output, key='metadata/event_metadata', mode='a', encoding='utf-8', format='table')