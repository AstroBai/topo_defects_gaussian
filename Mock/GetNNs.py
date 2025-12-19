import numpy as np  
import os
import sys
from tqdm import tqdm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from DefectFinder import DefectFinder
realizations = 1000
DATA_DIR = '/cephfs/jbai/DefectMockGaussian'
for i in tqdm(range(realizations)):
    sigmas = [np.radians(15/60), np.radians(30/60), np.radians(90/60)]
    for sigma in sigmas:
        s = int(np.degrees(sigma)*60)
        df = DefectFinder(f"{DATA_DIR}/Maps/realisation_{i}.npz")
        df.load_defects(f"{DATA_DIR}/Defects/realisation_{i}_s{s}_defects.npz")

        nn_pos_same1 = df.nearest_distance(charge=0.5, k=1)
        nn_neg_same1 = df.nearest_distance(charge=-0.5, k=1)
        nn_pos_opp1 = df.nearest_opposite_distance(charge=0.5, k=1)
        nn_neg_opp1 = df.nearest_opposite_distance(charge=-0.5, k=1)

        nn_pos_same2 = df.nearest_distance(charge=0.5, k=2)
        nn_neg_same2 = df.nearest_distance(charge=-0.5, k=2)
        nn_pos_opp2 = df.nearest_opposite_distance(charge=0.5, k=2)
        nn_neg_opp2 = df.nearest_opposite_distance(charge=-0.5, k=2)

        nn_pos_same4 = df.nearest_distance(charge=0.5, k=4)
        nn_neg_same4 = df.nearest_distance(charge=-0.5, k=4)
        nn_pos_opp4 = df.nearest_opposite_distance(charge=0.5, k=4)
        nn_neg_opp4 = df.nearest_opposite_distance(charge=-0.5, k=4)

        nn_pos_same8 = df.nearest_distance(charge=0.5, k=8)
        nn_neg_same8 = df.nearest_distance(charge=-0.5, k=8)
        nn_pos_opp8 = df.nearest_opposite_distance(charge=0.5, k=8)
        nn_neg_opp8 = df.nearest_opposite_distance(charge=-0.5, k=8)

        if s == 15:
            bin_edges1 = np.linspace(30, 100, 5)
            bin_edges2 = np.linspace(30, 200, 5)
            bin_edges4 = np.linspace(30, 300, 5)
            bin_edges8 = np.linspace(30, 400, 5)
        elif s == 30:
            bin_edges1 = np.linspace(30, 300, 5)
            bin_edges2 = np.linspace(30, 450, 5)
            bin_edges4 = np.linspace(30, 600, 5)
            bin_edges8 = np.linspace(30, 750, 5)
        elif s == 90:
            bin_edges1 = np.linspace(30, 750, 5)
            bin_edges2 = np.linspace(30, 1125, 5)
            bin_edges4 = np.linspace(30, 1500, 5)
            bin_edges8 = np.linspace(30, 1875, 5)
        # cumulative number counts
        nn_pos_same1_hist, _ = np.histogram(nn_pos_same1, bins=bin_edges1, density=False)
        nn_neg_same1_hist, _ = np.histogram(nn_neg_same1, bins=bin_edges1, density=False)
        nn_pos_opp1_hist, _ = np.histogram(nn_pos_opp1, bins=bin_edges1, density=False)
        nn_neg_opp1_hist, _ = np.histogram(nn_neg_opp1, bins=bin_edges1, density=False)
        nn_pos_same2_hist, _ = np.histogram(nn_pos_same2, bins=bin_edges2, density=False)
        nn_neg_same2_hist, _ = np.histogram(nn_neg_same2, bins=bin_edges2, density=False)
        nn_pos_opp2_hist, _ = np.histogram(nn_pos_opp2, bins=bin_edges2, density=False)
        nn_neg_opp2_hist, _ = np.histogram(nn_neg_opp2, bins=bin_edges2, density=False)
        nn_pos_same4_hist, _ = np.histogram(nn_pos_same4, bins=bin_edges4, density=False)
        nn_neg_same4_hist, _ = np.histogram(nn_neg_same4, bins=bin_edges4, density=False)
        nn_pos_opp4_hist, _ = np.histogram(nn_pos_opp4, bins=bin_edges4, density=False)
        nn_neg_opp4_hist, _ = np.histogram(nn_neg_opp4, bins=bin_edges4, density=False)
        nn_pos_same8_hist, _ = np.histogram(nn_pos_same8, bins=bin_edges8, density=False)
        nn_neg_same8_hist, _ = np.histogram(nn_neg_same8, bins=bin_edges8, density=False)
        nn_pos_opp8_hist, _ = np.histogram(nn_pos_opp8, bins=bin_edges8, density=False)
        nn_neg_opp8_hist, _ = np.histogram(nn_neg_opp8, bins=bin_edges8, density=False)
        # make cumulative
        nn_pos_same1_cum = np.cumsum(nn_pos_same1_hist) / 4 / np.pi
        nn_neg_same1_cum = np.cumsum(nn_neg_same1_hist) / 4 / np.pi
        nn_pos_opp1_cum = np.cumsum(nn_pos_opp1_hist) / 4 / np.pi
        nn_neg_opp1_cum = np.cumsum(nn_neg_opp1_hist) / 4 / np.pi
        nn_pos_same2_cum = np.cumsum(nn_pos_same2_hist) / 4 / np.pi
        nn_neg_same2_cum = np.cumsum(nn_neg_same2_hist) / 4 / np.pi
        nn_pos_opp2_cum = np.cumsum(nn_pos_opp2_hist) / 4 / np.pi
        nn_neg_opp2_cum = np.cumsum(nn_neg_opp2_hist) / 4 / np.pi
        nn_pos_same4_cum = np.cumsum(nn_pos_same4_hist) / 4 / np.pi
        nn_neg_same4_cum = np.cumsum(nn_neg_same4_hist) / 4 / np.pi
        nn_pos_opp4_cum = np.cumsum(nn_pos_opp4_hist) / 4 / np.pi
        nn_neg_opp4_cum = np.cumsum(nn_neg_opp4_hist) / 4 / np.pi
        nn_pos_same8_cum = np.cumsum(nn_pos_same8_hist) / 4 / np.pi
        nn_neg_same8_cum = np.cumsum(nn_neg_same8_hist) / 4 / np.pi
        nn_pos_opp8_cum = np.cumsum(nn_pos_opp8_hist) / 4 / np.pi
        nn_neg_opp8_cum = np.cumsum(nn_neg_opp8_hist) / 4 / np.pi

        os.makedirs(f"{DATA_DIR}/NNs", exist_ok=True)
        np.savez(f"{DATA_DIR}/NNs/realisation_{i}_s{s}_NNs.npz",
                bin_edges1=bin_edges1,
                bin_edges2=bin_edges2,
                bin_edges4=bin_edges4,
                bin_edges8=bin_edges8,
                nn_pos_same1_cum=nn_pos_same1_cum,
                nn_neg_same1_cum=nn_neg_same1_cum,
                nn_pos_opp1_cum=nn_pos_opp1_cum,
                nn_neg_opp1_cum=nn_neg_opp1_cum,
                nn_pos_same2_cum=nn_pos_same2_cum,
                nn_neg_same2_cum=nn_neg_same2_cum,
                nn_pos_opp2_cum=nn_pos_opp2_cum,
                nn_neg_opp2_cum=nn_neg_opp2_cum,
                nn_pos_same4_cum=nn_pos_same4_cum,
                nn_neg_same4_cum=nn_neg_same4_cum,
                nn_pos_opp4_cum=nn_pos_opp4_cum,
                nn_neg_opp4_cum=nn_neg_opp4_cum,
                nn_pos_same8_cum=nn_pos_same8_cum,
                nn_neg_same8_cum=nn_neg_same8_cum,
                nn_pos_opp8_cum=nn_pos_opp8_cum,
                nn_neg_opp8_cum=nn_neg_opp8_cum)
        

