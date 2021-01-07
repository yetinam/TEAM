# The Transformer earthquake alerting model (TEAM)
Implementation for early warning (TEAM) and for location and magnitude estimation (TEAM-LM)

## Installation

We recommend using conda. TEAM requires python version 3.6 or higher.
Please install python requirements with `pip install -r requirements.txt`.
Note that this does not install GPU support for tensorflow.
If required, GPU support needs to be installed manually.

## Training

Model and training configurations are defined in json-files.
Please consult the folders `magloc_configs` and `pga_configs` for example configurations.

To start model training use:
```
python train.py --config [CONFIG].json
```

To test a config by running a model with only few data points, the command line flag `--test_run` can be used.

The training saves model weights to the given weight path.
In addition, it writes logs for tensorboard to `/logs/scalars`.

## Citation

When using TEAM or TEAM-ML please reference the associated publications:
```
@article{munchmeyer2020team,
  title={The transformer earthquake alerting model: A new versatile approach to earthquake early warning},
  author={M{\"u}nchmeyer, Jannes and Bindi, Dino and Leser, Ulf and Tilmann, Frederik},
  journal={Geophysical Journal International},
  year={2020},
  doi={10.1093/gji/ggaa609}
}

@article{munchmeyer2021teamlm,
  title={Earthquake magnitude and location estimation from real time seismic waveforms with a transformer network},
  author={M{\"u}nchmeyer, Jannes and Bindi, Dino and Leser, Ulf and Tilmann, Frederik},
  journal={arXiv preprint arXiv:2101.02010},
  year={2021}
}
```

## Config options

The configurations are split into model and training parameters.
Furthermore, there are three global parameters, the random seed `seed`, the model type, which currently needs to be set to `transformer`, and `ensemble`, the size of the ensemble.
By default, no ensemble but a single model is trained.
Note that not all parameter combinations are possible for both theoretical and implementation restrictions, and might lead to crashes.

### Model parameters

Parameter | Default value | Description
---- | ---- | ---- 
max_stations | 25 | Maximum number of stations in training
waveform_model_dims | (500, 500, 500) | Dimensions of the MLP in the feature extractor
output_mlp_dims | (150, 100, 50, 30, 10) | Dimensions of the MLP in the mixture density output for magnitude and PGA
output_location_dims | (150, 100, 50, 50, 50) | Dimensions of the MLP in the mixture density output for location
wavelength | ((0.01, 10), (0.01, 10), (0.01, 10)) | Wavelength ranges for the position embeddings (Latitude, Longitude, Depth)
mad_params | {"n_heads": 10, "att_dropout": 0.0, "initializer_range": 0.02} | Parameters for the multi-head self-attention
ffn_params | {'hidden_dim': 1000} | Parameters for the Transformer feed-forward layer
transformer_layers | 6 | Number of transformer layers
hidden_dropout | 0.0 | Transformer hidden dropout
activation | 'relu' | Activation function for CNNs and MLPs
n_pga_targets | 0 | Number of PGA targets
location_mixture | 5 | Size of the Gaussian mixture for location
pga_mixture | 5 | Size of the Gaussian mixture for PGA
magnitude_mixture | 5 | Size of the Gaussian mixture for magnitude
borehole | False | Whether the data contains borehole measurements
bias_mag_mu | 1.8 | Bias initializer for magnitude mu
bias_mag_sigma | 0.2 | Bias initializer for magnitude sigma
bias_loc_mu | 0 | Bias initializer for location mu
bias_loc_sigma | 1 | Bias initializer for location sigma
event_token_init_range | None | Initializer for event token. Defaults to ones, if value is None
dataset_bias | False | Adds a scalar bias term to the output for joint training on multiple datasets
no_event_token | False | Removes event token, disables magnitude and location estimation
downsample | 5 | Downsampling factor for the first CNN layer
rotation | None | Rotation to be applied to latitude and longitude before the position embedding
rotation_anchor | None | Point to rotate around
skip_transformer | False | Replace the transformer by a pooling layer
alternative_coords_embedding | False | Concatenate position instead of adding position embeddings

### Training parameters

To accommodate joint training, parameters are split into general training parameters and generator parameters.
For training on a single data set all generator parameters can directly be given in the training parameter array.
For joint training a list of generator parameter dictionaries needs to be given.
Check the given configs for examples.

#### General training parameters
Parameter | Default value | Description
---- | ---- | ---- 
weight_path | - | Path to save model weights. Needs to be empty.
data_path | - | Path to the training data. If given a list, the model assumes joint training on multiple datasets.
overwrite_sampling_rate | - | If given, all data is resampled to the given sampling rate. Needs to be a divisor of the sampling rate given in the data.
ensemble_rotation | False | If position embeddings between the different ensemble member should be rotated. 
single_station_model_path | - | Weights of the initial model for the feature extraction. If not given, the model will train a single station model first to initialize the feature extraction.
lr | - | Learning rate
clipnorm | - | Norm for gradient clipping
filter_single_station_by_pick | False | For single station training only train on traces containing a pick.
workers | 10 | Number of parallel workers for data preprocessing
epochs_single_station | - | Number of training epochs for single station model
load_model_path | - | Initial weights for model. Not recommended, use transfer_model_path instead.
transfer_model_path | - | Initial weights for model. Also transfers weights between models with and without borehole data.
ensemble_load | False | Load weights for each ensemble member from the corresponding member of another ensemble.
wait_for_load | False | Wait if weight file does not exist. Otherwise raises an exception.
loss_weights | - | Loss weights given as a dict. Depending on the model configuration required parameters are `magnitude`, `location` and `pga`.
lr_decay_patience | 6 | Patience for learning rate decay
epochs_full_model | - | Number of training epochs for full model

#### Generator params
Parameter | Default value | Description
---- | ---- | ----
key | - | Key of the magnitude value in the event metadata
batch_size | 32 | Size of training batches
cutout | None | Value range for temporal blinding given as a tuple of sample idices
shuffle | True | Shuffle order of events
coords_target | True | Return target coordinates as outputs
oversample | 1 | Number of times to show each event per epoch
pos_offset | (-21, -69) | Scalar shift applied to latitude and longitude
label_smoothing | False | Enables label smoothing for large magnitudes
station_blinding | False | Randomly zeros out stations in each training example
magnitude_resampling | 3 | Factor to upsample number of large magnitude events
adjust_mean | True | Sets mean of all waveform traces to zero. Disabling this will cause a knowledge leak!
transform_target_only | False | Only transform coordinates of target coordinates, but not of station coordinates
trigger_based | False | Disable data from stations without trigger 
min_upsample_magnitude | 2 | Minimum magnitude to upsample event above this magnitude
disable_station_foreshadowing | False | Zeros coordinates for stations without data
selection_skew | None | If given, prefers station closer to event
pga_from_inactive | False | Predict PGA for stations without waveforms too
integrate | False | Integrate waveform traces
select_first | False | Only use closest stations
fake_borehole | False | Adds 3 artifical channels to fake borehole data
scale_metadata | True | Rescale coordinates. Not required with position embeddings.
pga_key | pga | Key for the PGA values in the data set 
p_pick_limit | 5000 | Maximum pick to assume for selection skew. Ensures probability of selection is positive for all stations.
coord_keys | None | Keys for the event coordinates in the event metadata. If none will be detected automatically. 
upsample_high_station_events | None | Factor to upsample events recorded at many stations
pga_selection_skew | None | Similar to selection_skew, but for PGA targets
shuffle_train_dev | False | Shuffle events between training and development set
custom_split | None | Use custom split instead of temporal 60:10:30 split. Custom splits are defined in `loader.py`.
min_mag | None | Only use events with at least this magnitude
decimate_events | None | Integer k, if given only load every kth event.

## Evaluation

For evaluating a model use `python evaluate.py --experiment_path [WEIGHTS_PATH]`.
To evaluate PGA estimation as well us `--pga`, to evaluate warning times use `--head_times`.
By default, the development set is evaluated.
To evaluate the test set use the `--test` flag.
Certain further detail options are documented in the python file. 

The evaluation creates a evaluation subfolder in the weights path, containing a statistics file, multiple plots and a prediction file.
The statistics file includes for each target (magnitude, location, PGA) and each time step the values of performance metrics.
Values are:
- R2, RMSE, MAE (magnitude)
- Hypocentral RMSE and MAE, Epicentral RMSE and MAE (location)
- R2, RMSE, MAE (PGA)

The predictions are a pickle file containg a list consisting of:
- Evaluation times
- Magnitude predictions. Numpy array with shape (times, events, mixture, (alpha, mu, sigma)).
- Location predictions. Numpy array with shape (times, events, mixture, (alpha, mu latitude, mu longitude, mu depth, sigma latitude, sigma longitude, sigma depth))
- PGA predictions. List containing one entry for each time, containing list of events. Each event is a numpy array with shape (station, mixture, (alpha, mu, sigma)) 
- Warning time results, list of events, each event containing:
    - times of predicted warnings, array with shape (stations, PGA thresholds, alpha)
    - times of actual warnings, array with shape (stations, PGA thresolds)
    - distance of stations to event, array with shape (stations,)+
- Values of alpha
    
## Datasets

The dataset for Italy is available at [10.5880/GFZ.2.4.2020.004](https://doi.org/10.5880/GFZ.2.4.2020.004).
The dataset for Chile will shortly be published.
To obtain the dataset for Japan, please run the following commands.
Obtaining the data requires an account with [NIED](https://www.kyoshin.bosai.go.jp/).
The download script will prompt for you login credentials.
```
python japan.py --action download_events --catalog resources/kiknet_events --output [OUTPUT FOLDER]
python japan.py --action extract_events --input [DATA FOLDER] --output [HDF5 OUTPUT PATH]
```
The download sometimes crashed, due to connection issues to NIED.
It can be resumed by simply restarting the download job. 

The extraction can be calculated in parallel using sharding.
To this end use the flag `--shards [NUMBER OF SHARDS]` and start jobs with `--shard_id` between `0` and `[NUMBER OF SHARDS] - 1`.
Run all shards with the same configuration, the output path will be adjusted automatically.
Use `python japan.py --action merge_hdf5 --input [PATH OF ALL SHARDS] --output [HDF5 OUTPUT PATH]`.

## Baselines

Baseline implementations for magnitude estimation and early warning are contained in `mag_baselines.py` and `pga_baselines.py`.
For reference on the usage please see the samples configs in `mag_baseline_configs` and `pga_baseline_configs` and the implementation.
