#######################################
### Project: Network Medicine Framework for Identifying Drug Repurposing Opportunities for COVID-19.
### Description: Drug-disease scoring for A1-A4
### Author: Marinka Zitnik
### email: marinka at hms dot harvard dot edu
### date: 3rd March 2021
#######################################

import csv
from operator import itemgetter
from datetime import datetime
import os
from os.path import join as pjoin

import numpy as np
import umap
from sklearn import metrics

###
# NOTE:
# https://umap-learn.readthedocs.io/en/latest/reproducibility.html
#
# UMAP is a stochastic algorithm – it makes use of randomness both to speed up approximation steps,
# and to aid in solving hard optimization problems. This means that different runs of UMAP can produce
# different results. UMAP is relatively stable – thus the variance between runs should ideally be relatively
# small – but different runs may have variations none the less. To ensure that results can be reproduced exactly
# we use fixed random seed state.
#
# UMAP version 0.3 (available in March 2020) was used to generate drug rankings for AI1-AI4 pipeline. You need to have
# that version installed in order to reproduce results exactly.
#
# UMAP Release Notes (https://umap-learn.readthedocs.io/en/latest/release_notes.html):
# The current version of UMAP in March 2021 is 0.5.1.
#
# Since version 0.4 UMAP depends on stochastic algorithm of PyNNDescent and supports multi-threading for faster
# performance; when performing optimization this exploits the fact that race conditions between the threads are
# acceptable within certain optimization phases. Unfortunately this means that the randomness in UMAP outputs for the
# multi-threaded case depends not only on the random seed input, but also on race conditions between threads during
# optimization, over which no control can be had. Because of that, you can expect minor variation in drug rankings if
# you run the code on those UMAP versions.
#
###

def load_embedding_vectors(data_path):
    obj2idx, emb = {}, []
    current_directory = os.getcwd()
    with open(data_path) as fin:
        csv_reader = csv.reader(fin)
        for i, line in enumerate(csv_reader):
            obj2idx[line[0]] = i
            emb.append(list(map(float, line[1:])))
    return obj2idx, np.array(emb)

def write_prediction_list(drug_dist, out_path):
    with open(out_path, 'w') as fout:
        fout.write('# Disease: COVID-19\n')
        fout.write('# Date: %s\n' % datetime.now())
        fout.write('Drug ID\tDistance score\n')
        for drug, dist in drug_dist:
            fout.write('%s\t%8.5f\n' % (drug, dist))

def pipeline_A1(drug2idx, drug_emb, disease2idx, disease_emb, out_path):
    emb_together = np.concatenate((drug_emb, disease_emb))
    coords = umap.UMAP(
        n_components=2, n_neighbors=10, min_dist=0.25,
        metric='cosine', random_state=0).fit_transform(emb_together)

    y = np.asarray(coords[len(drug2idx) + disease2idx['COVID-19']])
    y_reshape = y.reshape(1, -1)
    distances = metrics.pairwise_distances(
        X=coords, Y=y_reshape, metric='euclidean')
    drug_dist = [(drug, distances[drug2idx[drug]][0]) for drug in drug2idx]
    drug_dist = sorted(drug_dist, reverse=False, key=itemgetter(1))
    write_prediction_list(drug_dist, out_path)

def pipeline_A2(drug2idx, drug_emb, disease2idx, disease_emb, out_path):
    emb_together = np.concatenate((drug_emb, disease_emb))
    coords = umap.UMAP(
        n_components=2, n_neighbors=10, min_dist=0.8,
        metric='cosine', random_state=0).fit_transform(emb_together)

    y = np.asarray(coords[len(drug2idx) + disease2idx['COVID-19']])
    y_reshape = y.reshape(1, -1)
    distances = metrics.pairwise_distances(
        X=coords, Y=y_reshape, metric='euclidean')
    drug_dist = [(drug, distances[drug2idx[drug]][0]) for drug in drug2idx]
    drug_dist = sorted(drug_dist, reverse=False, key=itemgetter(1))
    write_prediction_list(drug_dist, out_path)

def pipeline_A3(drug2idx, drug_emb, disease2idx, disease_emb, out_path):
    emb_together = np.concatenate((drug_emb, disease_emb))
    coords = umap.UMAP(
        n_components=2, n_neighbors=5, min_dist=0.5,
        metric='cosine', random_state=0).fit_transform(emb_together)

    y = np.asarray(coords[len(drug2idx) + disease2idx['COVID-19']])
    y_reshape = y.reshape(1, -1)
    distances = metrics.pairwise_distances(
        X=coords, Y=y_reshape, metric='euclidean')
    drug_dist = [(drug, distances[drug2idx[drug]][0]) for drug in drug2idx]
    drug_dist = sorted(drug_dist, reverse=False, key=itemgetter(1))
    write_prediction_list(drug_dist, out_path)

def pipeline_A4(drug2idx, drug_emb, disease2idx, disease_emb, out_path):
    emb_together = np.concatenate((drug_emb, disease_emb))
    coords = umap.UMAP(
        n_components=2, n_neighbors=10, min_dist=1, metric='cosine',
        random_state=0).fit_transform(emb_together)

    y = np.asarray(coords[len(drug2idx) + disease2idx['COVID-19']])
    y_reshape = y.reshape(1, -1)
    distances = metrics.pairwise_distances(
        X=coords, Y=y_reshape, metric='euclidean')
    drug_dist = [(drug, distances[drug2idx[drug]][0]) for drug in drug2idx]
    drug_dist = sorted(drug_dist, reverse=False, key=itemgetter(1))
    write_prediction_list(drug_dist, out_path)


data_path = 'submodules/baseline/data'
output_path = 'submodules/baseline/output'

disease2idx, disease_emb = load_embedding_vectors(pjoin(data_path, 'DatasetS6.csv'))
drug2idx, drug_emb = load_embedding_vectors(pjoin(data_path, 'DatasetS7.csv'))

pipeline_A1(drug2idx, drug_emb, disease2idx, disease_emb, pjoin(output_path, 'ai', '1__COVID-19.tsv'))
pipeline_A2(drug2idx, drug_emb, disease2idx, disease_emb, pjoin(output_path, 'ai', '2__COVID-19.tsv'))
pipeline_A3(drug2idx, drug_emb, disease2idx, disease_emb, pjoin(output_path, 'ai', '3__COVID-19.tsv'))
pipeline_A4(drug2idx, drug_emb, disease2idx, disease_emb, pjoin(output_path, 'ai', '4__COVID-19.tsv'))