import numpy as np
np.random.seed(123)

import pandas as pd
from math import sqrt, ceil

import h5py

from sklearn.utils import shuffle
import tensorflow as tf

from tfbio.data import Featurizer, make_grid, rotate
import tfbio.net

import os.path

import matplotlib as mpl
mpl.use('agg')

import seaborn as sns
sns.set_style('white')
sns.set_context('paper')
sns.set_color_codes()

import time
timestamp = time.strftime('%Y-%m-%dT%H:%M:%S')


import argparse
parser = argparse.ArgumentParser(
    description='Make graph of predictions',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
io_group = parser.add_argument_group('I/O')
io_group.add_argument('--data_file', '-d', required=True, type=str,
                      help='data file to run through the network (e.g. our_train.hdf)')
io_group.add_argument('--network', '-n', required=True, type=str,
                      help='network to run the data file through (e.g. pafnucy_retrained)')
io_group.add_argument('--out', '-o', required=True, type=str,
                      help='output file (e.g. our_graph.pdf)')
io_group.add_argument('--color', '-c', required=False, default='b', type=str,
                      help='color of points in graph (e.g. blue)')
args = parser.parse_args()

# load and use the new model
graph = tf.Graph()

with graph.as_default():
    saver = tf.train.import_meta_graph("{}.meta".format(args.network))

x = graph.get_tensor_by_name('input/structure:0')
y = graph.get_tensor_by_name('output/prediction:0')
t = graph.get_tensor_by_name('input/affinity:0')
keep_prob = graph.get_tensor_by_name('fully_connected/keep_prob:0')
mse = graph.get_tensor_by_name('training/mse:0')

predictions = []

coords = []
features = []
affinity = []
ids = []

x_ = []

with h5py.File(args.data_file, 'r') as f:
    for pdb_id in f:
        dataset = f[pdb_id]
        my_coords = dataset[:, :3]
        my_features = dataset[:, 3:]

        coords.append(my_coords)
        features.append(my_features)
        affinity.append(dataset.attrs['affinity'].mean())
        ids.append(pdb_id)
        grid = make_grid(my_coords, my_features)
        x_.append(grid)

x_ = np.vstack(x_)

featurizer = Featurizer()
columns = {name: i for i, name in enumerate(featurizer.FEATURE_NAMES)}

# normalize charges
charges = []
for feature_data in features:
    charges.append(feature_data[..., columns['partialcharge']])

charges = np.concatenate([c.flatten() for c in charges])

m = charges.mean()
std = charges.std()


ids = np.array(ids)
affinity = np.reshape(affinity, (-1, 1))
size = len(affinity)

def get_batch(rotation=0):
    global coords, features, std
    x = []
    for i, idx in enumerate(range(size)):
        coords_idx = rotate(coords[idx], rotation)
        features_idx = features[idx]
        x.append(make_grid(coords_idx, features_idx,
                 grid_resolution=1.0,
                 max_dist=10.0))
    x = np.vstack(x)
    x[..., columns['partialcharge']] /= std
    return x

with tf.Session(graph=graph) as session:
    saver.restore(session, args.network)
    print('predictions with loaded model:',
          session.run(y, feed_dict={x: x_, keep_prob: 1.0}))
    tf.set_random_seed(123)

    pred = np.zeros((size, 1))

    pred, mse_dataset = session.run(
        [y, mse],
        feed_dict={x: get_batch(),
                   t: affinity,
                   keep_prob: 1.0}
    )

    predictions = pd.DataFrame(data={'pdbid': ids,
                                     'real': affinity[:, 0],
                                     'predicted': pred[:, 0]})
    rmse = sqrt(mse_dataset)


#predictions = pd.concat(predictions, ignore_index=True)
#predictions.to_csv('our_predictions.csv', index=False)

grid = sns.jointplot('real', 'predicted', data=predictions, color=args.color,
                     space=0.0, xlim=(0, 16), ylim=(0, 16),
                     annot_kws={'title': '(rmse=%.3f)' % rmse})

image = tfbio.net.custom_summary_image(grid.fig)
grid.fig.savefig(args.out)

print("RMSE: {}".format(rmse))
