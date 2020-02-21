# Modified from
# https://gist.github.com/marta-sd/bd359e5047e7bc1abb8ba5bb65799e35

# before running this script clone Pafnucy's repository and create the environment:
# $ git clone https://gitlab.com/cheminfIBB/pafnucy
# $ cd pafnucy
# $ conda env create -f environment_gpu.yml


import pandas as pd
import csv
import numpy as np
import h5py
import tensorflow as tf

from sklearn.metrics import mean_squared_error
from tfbio.data import make_grid, rotate, Featurizer
import tfbio.net

import argparse
import math

import matplotlib as mpl
mpl.use('agg')

import seaborn as sns
sns.set_style('white')
sns.set_context('paper')
sns.set_color_codes()

# From https://stackoverflow.com/a/4602224
def unison_shuffled_copies(a, b, c, d, e):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p], c[p], d[p], e[p]

import time
timestamp = time.strftime('%Y-%m-%dT%H:%M:%S')
def_out = "results/retrained-{}".format(timestamp)

parser = argparse.ArgumentParser(
    description='Retrain pafnucy',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
io_group = parser.add_argument_group('I/O')
io_group.add_argument('--data_file', '-d', required=True, type=str,
                      help='data file for training the network (e.g. our_train.hdf)')
io_group.add_argument('--out', '-o', required=False, type=str, default=def_out,
                      help='name of network to output (e.g. pafnucy_retrained)')

tr_group = parser.add_argument_group('Training')
tr_group.add_argument('-k', required=False, type=int, default=10,
                      help='number of groups to split the dataset into (default is 10)')
tr_group.add_argument('--learning_rate', '-lr', default=1e-5, type=float,
                      help='learning rate (default is 1e-5)')
tr_group.add_argument('--num_epochs', '-e', default=20, type=int,
                      help='number of epochs')
tr_group.add_argument('--rotations', '-r', metavar='R', default=list(range(24)),
                      type=int, nargs='+',
                      help='rotations to perform')
args = parser.parse_args()

def initialize_session():
    global graph, saver, x, y, t, keep_prob, train
    # load Pafnucy
    graph = tf.Graph()

    with graph.as_default():
        saver = tf.train.import_meta_graph('results/batch5-2017-06-05T07:58:47-best.meta')

    # get placeholders for input, prediction and target
    x = graph.get_tensor_by_name('input/structure:0')
    y = graph.get_tensor_by_name('output/prediction:0')
    t = graph.get_tensor_by_name('input/affinity:0')
    keep_prob = graph.get_tensor_by_name('fully_connected/keep_prob:0')
    cost = graph.get_tensor_by_name('training/cost:0')
    global_step = graph.get_tensor_by_name('training/global_step:0')

    with graph.as_default():
        to_run = [v for v in tf.trainable_variables()
                  if 'fully_connected' in v.name
                  or 'output' in v.name]
        with tf.variable_scope('training'):
            optimizer = tf.train.AdamOptimizer(args.learning_rate,
                                               name='re-optimizer')
            retrain = optimizer.minimize(cost, var_list=to_run,
                                         global_step=global_step,
                                         name='retrain')
    train = graph.get_tensor_by_name('training/retrain:0')


def parse_name(name):
    name = name[7:]  # remove ligand_ prefix
    pocket, ligand = name.split('_')
    pocket_parts = pocket.split('.')
    reconstructed_pocket = pocket_parts[0]
    for i in range(1, len(pocket_parts)):
        if i == 1:
            sep = '.'
        elif i == 2:
            sep = ' '
        else:
            sep = '/'
        reconstructed_pocket += sep + pocket_parts[i]
    return reconstructed_pocket, ligand

# load some data
featurizer = Featurizer()

print('\n---- FEATURES ----\n')
print('atomic properties:', featurizer.FEATURE_NAMES)

columns = {name: i for i, name in enumerate(featurizer.FEATURE_NAMES)}

x_ = []
y_ = []
ids = []
coords = []
features = []
with h5py.File(args.data_file, 'r') as f:
    for name in f:
        n_coords = (f[name][:, :3])
        n_features = (f[name][:, 3:])
        coords.append(n_coords)
        features.append(n_features)
        grid = make_grid(n_coords, n_features)
        x_.append(grid)
        y_.append(f[name].attrs['affinity'])
        ids.append(name)

x_ = np.vstack(x_)
y_ = np.reshape(y_, (-1, 1))
ids = np.vstack(ids)
coords = np.array(coords)
features = np.array(features)

# Run ititial network
initialize_session()
with tf.Session(graph=graph) as session:
    saver.restore(session, 'results/batch5-2017-06-05T07:58:47-best')
    initial_prediction = session.run(y, feed_dict={x: x_, keep_prob: 1.0})
    rmse = math.sqrt(mean_squared_error(y_, initial_prediction))
    print('-' * 20)
    print('\033[1mResults for initial network\033[0m')
    print("RMSE: {}".format(rmse))
    print()

# shuffle for k-fold cross validation
x_, y_, ids, coords, features = unison_shuffled_copies(x_, y_, ids, coords,
        features)
pairs = [parse_name(np.asscalar(name)) for name in ids]

# normalize charges
charges = []
for feature_data in features:
    charges.append(feature_data[..., columns['partialcharge']])

charges = np.concatenate([c.flatten() for c in charges])

m = charges.mean()
std = charges.std()
print('charges: mean=%s, sd=%s' % (m, std))
print('use sd as scaling factor')

def get_rotation(rotation):
    x = []
    for i in range(len(coords)):
        coords_i = rotate(coords[i], rotation)
        x.append(make_grid(coords_i, features[i]))
    x = np.vstack(x)
    x[..., columns['partialcharge']] /= std
    return x


# partition dataset into k groups
group_size = len(x_) // args.k
start = 0
end = group_size
groups = []
for i in range(args.k - 1):
    groups.append((start, end))
    start += group_size
    end += group_size
end = len(x_)
groups.append((start, end))

# re-train Pafnucy
train_errors = []
test_errors = []
saved_train_errors = []
saved_test_errors = []
predictions = []

group_num = 0
for begin, end in groups:
    initialize_session()
    with tf.Session(graph=graph) as session:
        session.run(tf.global_variables_initializer())
        # Start with fresh model each time
        saver.restore(session, 'results/batch5-2017-06-05T07:58:47-best')

        # Partition training and test sets
        test_x = x_[begin:end]
        test_y = y_[begin:end]
        train_x = np.concatenate((x_[:begin], x_[end:]))
        train_y = np.concatenate((y_[:begin], y_[end:]))

        # Retrain
        for _ in range(args.num_epochs):
            for rotation in args.rotations:
                x_rot = get_rotation(rotation)
                train_x_rot = np.concatenate((x_rot[:begin], x_rot[end:]))
                session.run(train, feed_dict={x: train_x_rot, t: train_y,
                    keep_prob: 1.0})

        # Calculate error
        train_pred = session.run(y, feed_dict={x: train_x, keep_prob: 1.0})
        test_pred = session.run(y, feed_dict={x: test_x, keep_prob: 1.0})
        train_rmse = math.sqrt(mean_squared_error(train_y, train_pred))
        test_rmse = math.sqrt(mean_squared_error(test_y, test_pred))
        train_errors.append(train_rmse)
        test_errors.append(test_rmse)
        print('-' * 20)
        print("\033[1mResults for group {}\033[0m".format(group_num))
        print("Training RMSE: {}".format(train_rmse))
        print("Test RMSE: {}".format(test_rmse))

        fname = "{}-{}".format(args.out, group_num)
        saver.save(session, fname)

    # Calculate error for saved model

    # load and use the new model
    new_graph = tf.Graph()

    with new_graph.as_default():
        saver = tf.train.import_meta_graph("{}.meta".format(fname))

    x = new_graph.get_tensor_by_name('input/structure:0')
    y = new_graph.get_tensor_by_name('output/prediction:0')
    keep_prob = new_graph.get_tensor_by_name('fully_connected/keep_prob:0')

    with tf.Session(graph=new_graph) as new_session:
        saver.restore(new_session, fname)
        new_train_pred = new_session.run(y, feed_dict={x: train_x,
                keep_prob: 1.0})
        new_test_pred = new_session.run(y, feed_dict={x: test_x,
                keep_prob: 1.0})
        new_train_rmse = math.sqrt(mean_squared_error(train_y, train_pred))
        new_test_rmse = math.sqrt(mean_squared_error(test_y, test_pred))
        saved_train_errors.append(new_train_rmse)
        saved_test_errors.append(new_test_rmse)
        print("Saved Model's Training RMSE: {}".format(new_train_rmse))
        print("Saved Model's Test RMSE: {}".format(new_test_rmse))
        print()
        all_pred = new_session.run(y, feed_dict={x: x_, keep_prob: 1.0})
        predictions.append(all_pred)

    group_num += 1


print('-' * 20)
print("\033[1mAggregate results\033[0m".format(group_num))
print("Training RMSE: {}".format(sum(train_errors) / args.k))
print("Test RMSE: {}".format(sum(test_errors) / args.k))
print("Saved Training RMSE: {}".format(sum(saved_train_errors) / args.k))
print("Saved Test RMSE: {}".format(sum(saved_test_errors) / args.k))

# save predictions
csv_fname = "{}-predictions.csv".format(args.out)
with open(csv_fname, mode='w') as f:
    fieldnames = ['Group', 'Pocket', 'Ligand', 'Actual', 'Initial']
    fieldnames.extend(["Model {}".format(i) for i in range(args.k)])
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    group_num = 0
    for begin, end in groups:
        for i in range(begin, end):
            pocket, ligand = pairs[i]
            actual = np.asscalar(y_[i])
            initial = np.asscalar(initial_prediction[i])
            row = {
                'Group': group_num,
                'Pocket': pocket,
                'Ligand': ligand,
                'Actual': actual,
                'Initial': initial,
            }
            for j in range(args.k):
                row["Model {}".format(j)] = np.asscalar(predictions[j][i])
            writer.writerow(row)
        group_num += 1

# save error report
error_csv_fname = "{}-rmse.csv".format(args.out)
with open(error_csv_fname, mode='w') as f:
    fieldnames = ['Model', 'Train RMSE', 'Test RMSE']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(args.k):
        row = {
            'Model': i,
            'Train RMSE': saved_train_errors[i],
            'Test RMSE': saved_test_errors[i],
        }
        writer.writerow(row)
    row = {
        'Model': 'aggregate',
        'Train RMSE': sum(saved_train_errors) / args.k,
        'Test RMSE': sum(saved_test_errors) / args.k,
    }
    writer.writerow(row)

# save hyperparameters
param_csv_fname = "{}-hyperparameters.csv".format(args.out)
with open(param_csv_fname, mode='w') as f:
    fieldnames = ['k', 'num_epochs', 'rotations']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    row = {
        'k': args.k,
        'num_epochs': args.num_epochs,
        'rotations': str(args.rotations),
    }
    writer.writerow(row)

# make graphs
train_color = 'b'
test_color = 'g'

group_num = 0
for begin, end in groups:
    # Train graph
    train_ids = [np.asscalar(ids[i]) for i in range(len(y_))
                 if i not in range(begin, end)]
    train_real = [np.asscalar(y_[i]) for i in range(len(y_))
                  if i not in range(begin, end)]
    train_pred = [np.asscalar(predictions[group_num][i])
                  for i in range(len(y_)) if i not in range(begin, end)]
    preds = pd.DataFrame(data={'pdbid': train_ids,
                               'real': train_real,
                               'predicted': train_pred})
    grid = sns.jointplot('real', 'predicted', data=preds,
            color=train_color, space=0.0, xlim=(0, 16), ylim=(0, 16),
            annot_kws={'title':
                "(rmse=%.3f)".format(saved_train_errors[group_num])})
    image = tfbio.net.custom_summary_image(grid.fig)
    grid.fig.savefig("{}-plot-train-model{}.pdf".format(args.out, group_num))

    # Test graph
    test_ids = [np.asscalar(ids[i]) for i in range(begin, end)]
    test_real = [np.asscalar(y_[i]) for i in range(begin, end)]
    test_pred = [np.asscalar(predictions[group_num][i])
                 for i in range(begin, end)]
    preds = pd.DataFrame(data={'pdbid': test_ids,
                               'real': test_real,
                               'predicted': test_pred})
    grid = sns.jointplot('real', 'predicted', data=preds,
            color=test_color, space=0.0, xlim=(0, 16), ylim=(0, 16),
            annot_kws={'title':
                "(rmse=%.3f)".format(saved_test_errors[group_num])})
    image = tfbio.net.custom_summary_image(grid.fig)
    grid.fig.savefig("{}-plot-test-model{}.pdf".format(args.out, group_num))

    group_num += 1
