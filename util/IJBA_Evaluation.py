import os
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
import scipy.io as sio
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_curve, auc

def Read_txt(TxtRoot):

    txt = open(TxtRoot)
    txt_tmp = txt.read().split('\n')

    return txt_tmp

def Test_Compare(Feature_Root, NumSplit):

    TAR = np.zeros([1, 2])
    fea_len = Feature_Root[list(Feature_Root.keys())[0]].shape[0]
    for sp in range(NumSplit):

        Intra = np.load('IJB-A_protocols/Compare_List/split{}/genuine_tmp_list.npy'.format(sp + 1), allow_pickle=True)
        Inter = np.load('IJB-A_protocols/Compare_List/split{}/imposter_tmp_list.npy'.format(sp + 1), allow_pickle=True)
        COMPARE = np.load('IJB-A_protocols/Compare_List/split{}/COMPARE.npy'.format(sp + 1), allow_pickle=True).item()

        COMPARE_Features = {}
        for tmp_compare_idx in COMPARE:
            if 'img/2055' in COMPARE[tmp_compare_idx]: COMPARE[tmp_compare_idx].remove('img/2055')
            tmp_fea = np.zeros((len(COMPARE[tmp_compare_idx]), fea_len))
            for ii, tmp_file in enumerate(COMPARE[tmp_compare_idx]):
                tmp_fea[ii, :] = Feature_Root[tmp_file].reshape(1, -1)
            COMPARE_Features.setdefault(tmp_compare_idx, np.mean(tmp_fea, 0))


        label = np.concatenate((np.ones((len(Intra), 1)), np.zeros((len(Inter), 1))))
        scores = [cosine_similarity(COMPARE_Features[in1].reshape(1, -1), COMPARE_Features[in2].reshape(1, -1))[0][0]
               for in1, in2 in np.concatenate((Intra, Inter))]

        fpr, tpr, thresh = roc_curve(label, scores)
        TAR[0, 0] += tpr[np.argmin(np.abs(fpr - 0.01))]
        TAR[0, 1] += tpr[np.argmin(np.abs(fpr - 0.001))]

    print('Already validating on verification set data')
    return TAR / NumSplit

def Test_Search(Feature_Root, NumSplit):

    fea_len = Feature_Root[list(Feature_Root.keys())[0]].shape[0]
    Rank = np.zeros((1, 2))
    for sp in range(NumSplit):

        root = 'IJB-A_protocols/IJB-A_1N_sets/split{}'.format(sp+1)
        gallery_meta = pd.read_csv('{}/search_gallery_{}.csv'.format(root, str(sp+1)))
        probe_meta = pd.read_csv('{}/search_probe_{}.csv'.format(root, str(sp+1)))

        Gallery = np.concatenate((np.array(list(gallery_meta.TEMPLATE_ID)).reshape(-1, 1),
                                  np.array(list(gallery_meta.SUBJECT_ID)).reshape(-1, 1)), 1)
        Probe = np.concatenate((np.array(list(probe_meta.TEMPLATE_ID)).reshape(-1, 1),
                                  np.array(list(probe_meta.SUBJECT_ID)).reshape(-1, 1)), 1)

        Gallery_Features = np.zeros((len(np.unique(Gallery, axis=0)), fea_len))
        for g_idx, (g1, g2) in enumerate(np.unique(Gallery, axis=0)):
            index = np.intersect1d(np.nonzero(np.in1d(Gallery[:, 0], g1))[0], np.nonzero(np.in1d(Gallery[:, 1], g2))[0])
            tmp_fea = np.zeros((len(list(gallery_meta.FILE[index])), fea_len))
            for ii, tmp_file_idx in enumerate(list(gallery_meta.FILE[index])):
                tmp_fea[ii, :] = Feature_Root[tmp_file_idx.split('.')[0]].reshape(1, -1)
            Gallery_Features[g_idx, :] = np.mean(tmp_fea, 0)
            # Gallery_Features.setdefault('{}_{}'.format(g1, g2), np.mean(tmp_fea, 0))

        Probe_Features = np.zeros((len(np.unique(Probe, axis=0)), fea_len))
        for p_idx, (p1, p2) in enumerate(np.unique(Probe, axis=0)):
            index = np.intersect1d(np.nonzero(np.in1d(Probe[:, 0], p1))[0], np.nonzero(np.in1d(Probe[:, 1], p2))[0])
            tmp_fea = np.zeros((len(list(probe_meta.FILE[index])), fea_len))
            for ii, tmp_file_idx in enumerate(list(probe_meta.FILE[index])):
                tmp_fea[ii, :] = Feature_Root[tmp_file_idx.split('.')[0]].reshape(1, -1)
            Probe_Features[p_idx, :] = np.mean(tmp_fea, 0)
            # Probe_Features.setdefault('{}_{}'.format(p1, p2), np.mean(tmp_fea, 0))


        scores = cosine_similarity(Probe_Features, Gallery_Features)  # [probe, gallery]
        candidate_rank = []
        for probe_idx, tmp_scores in zip(np.unique(Probe, axis=0)[:, 1], scores):  # 1st probe subject
            g_sort = np.unique(Gallery, axis=0)[:, 1][np.argsort(-tmp_scores)]
            if len(np.where(g_sort==probe_idx)[0])==0: continue
            else: candidate_rank.append(np.where(g_sort == probe_idx)[0][0])



        cmc = np.zeros([len(np.unique(Gallery, axis=0)[:, 1]), 1])
        for u in range(len(np.unique(Gallery, axis=0)[:, 1])):
            cmc[u] = len(np.where(np.array(candidate_rank)<=u)[0])

        Rank[0, 0] += cmc[0]/len(candidate_rank)
        Rank[0, 1] += cmc[4]/len(candidate_rank)

    print('Already validating on identification set data')
    return Rank / NumSplit
