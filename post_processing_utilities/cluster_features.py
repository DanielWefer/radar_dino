# Copyright (c) Northwestern Argonne Institute of Science and Engineering (NAISE)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import os
from pathlib import Path

import numpy as np
import torch

from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD as SVD
from sklearn.preprocessing import StandardScaler
try:
    from sklearn_som.som import SOM
except ImportError:
    SOM = None

from joblib import dump


def get_args_parser():
    parser = argparse.ArgumentParser(
        'Radar-DINO feature post processing using dimensionality reduction and clustering',
        add_help=False,
    )
    parser.add_argument('--features_path', default='', type=str,
        help='Path to feat.pth or to a directory containing feat.pth.')

    # For the dim red method
    parser.add_argument('--dimensions', default=2, type=int,
        help='Reduce to this number of dimensions for methods that use an explicit dimension count.')
    parser.add_argument('--dim_red_method', default='PCA', type=str,
        choices=['PCA', 'SVD'], help='Dimensionality reduction method.')
    parser.add_argument('--pca_components', default='mle', type=str,
        help='PCA components: use mle, variance fraction like 0.99, or an integer.')

    # For the clustering method
    parser.add_argument('--clustering_method', default='DBSCAN', type=str,
        choices=['DBSCAN', 'KMEANS', 'SOM'], help='Clustering method.')
    parser.add_argument('--dbscan_eps', type=float, default=0.5, help='DBSCAN eps.')
    parser.add_argument('--dbscan_min_samples', default=5, type=int, help='DBSCAN min_samples.')
    parser.add_argument('--kmeans_clusters', default=100, type=int, help='Number of KMEANS clusters.')
    parser.add_argument('--som_m', default=10, type=int, help='SOM grid rows.')
    parser.add_argument('--som_n', default=10, type=int, help='SOM grid columns.')

    # General arguments
    parser.add_argument('--output_dir', default=None, help='Path where to save clustering results.')
    parser.add_argument('--subsample_feats', default=None, type=int,
        help='Subsample the feature space to a reduced number of samples.')

    return parser


def process_features(args):
    scale_model, x = bring_features(args)
    dim_red_model, x = reduce_dim(x, args.dim_red_method, args.dimensions, args.pca_components)
    y = cluster_data(x, args)
    clusters = {'x': x, 'y': y}
    return clusters, dim_red_model, scale_model


def resolve_features_path(features_path):
    if os.path.isdir(features_path):
        features_path = os.path.join(features_path, 'feat.pth')
    if not os.path.isfile(features_path):
        raise FileNotFoundError(f'Could not find feature file: {features_path}')
    return features_path


def bring_features(args):
    features_path = resolve_features_path(args.features_path)
    feats = torch.load(features_path, map_location='cpu', weights_only=False)
    if isinstance(feats, torch.Tensor):
        feats = feats.detach().cpu().numpy()
    else:
        feats = np.asarray(feats)
    if feats.ndim != 2:
        raise ValueError(f'Expected a 2D feature matrix, got shape {feats.shape}')
    if args.subsample_feats:
        feats = choose_random_rows(feats, args.subsample_feats)

    scale = StandardScaler()
    feats = scale.fit_transform(feats)
    return scale, feats


def parse_pca_components(value):
    if value == 'mle':
        return 'mle'
    try:
        numeric = float(value)
    except ValueError as exc:
        raise ValueError('--pca_components must be mle, a variance fraction, or an integer') from exc
    if numeric.is_integer() and numeric >= 1:
        return int(numeric)
    return numeric


def reduce_dim(feats, method='PCA', dimensions=2, pca_components='mle'):
    if method == 'PCA':
        pca = PCA(n_components=parse_pca_components(pca_components), svd_solver='full')
        pca.fit(feats)
        return pca, pca.transform(feats)
    if method == 'SVD':
        svd = SVD(n_components=dimensions)
        svd.fit(feats)
        return svd, svd.transform(feats)
    raise NameError(f'Unknown dimensionality reduction method: {method}')


def cluster_data(data, args):
    if args.clustering_method == 'DBSCAN':
        labels = DBSCAN(eps=args.dbscan_eps, min_samples=args.dbscan_min_samples).fit_predict(data)
    elif args.clustering_method == 'KMEANS':
        labels = KMeans(init='k-means++', n_clusters=args.kmeans_clusters, n_init=100).fit_predict(data)
    elif args.clustering_method == 'SOM':
        if SOM is None:
            raise ImportError('SOM clustering requires sklearn_som. Install it or choose DBSCAN or KMEANS.')
        labels = SOM(m=args.som_m, n=args.som_n, dim=data.shape[1]).fit_predict(data)
    else:
        raise NameError(f'Unknown clustering algorithm method: {args.clustering_method}')
    return labels


def choose_random_rows(an_array, n_samples):
    number_of_rows = an_array.shape[0]
    if n_samples > number_of_rows:
        raise ValueError(f'Cannot subsample {n_samples} rows from only {number_of_rows} rows')
    random_indices = np.random.choice(number_of_rows, size=n_samples, replace=False)
    return an_array[random_indices, :]


def save_process(args, clusters, dim_red_model, scale_model):
    os.makedirs(args.output_dir, exist_ok=True)
    clusters_fname = os.path.join(args.output_dir, 'clusters.npy')
    np.save(clusters_fname, clusters)
    print(f'{clusters_fname} saved.')

    red_dim_model_fname = os.path.join(args.output_dir, 'dim_red_model.joblib')
    dump(dim_red_model, red_dim_model_fname)
    print(f'{red_dim_model_fname} saved.')

    scale_model_fname = os.path.join(args.output_dir, 'scale_model.joblib')
    dump(scale_model, scale_model_fname)
    print(f'{scale_model_fname} saved.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Radar-DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    clusters, dim_red_model, scale_model = process_features(args)
    if args.output_dir:
        save_process(args, clusters, dim_red_model, scale_model)
