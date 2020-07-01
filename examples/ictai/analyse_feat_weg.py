import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from scipy.stats import norm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import scipy

from topo_quant import *

with open("feature_map.pkl", "rb") as fread:
    feat_map = pickle.load(fread)
    hist, bins, _ = plt.hist(feat_map.numpy().flatten(), 256, density=True, range=[-2, 2], color="#3498db")
    # plt.show()
    plt.savefig('feature_map.pdf')
    plt.clf()

    feat_map_quant = quantize(feat_map, num_bits=8, dequantize=True, signed=True).numpy().flatten()
    # weights = np.ones_like(feat_map_quant)/float(len(feat_map_quant))
    hist, bins, _ = plt.hist(feat_map_quant, 256, range=[-2, 2], density=True, color="#d35400")
    # plt.show()
    # mu, sigma = scipy.stats.norm.fit(feat_map_quant)
    # best_fit_line = scipy.stats.norm.pdf(bins, mu, sigma)
    # plt.plot(bins, best_fit_line, color="green")
    # plt.xlabel('(a)')
    # print((hist * np.diff(bins)).sum())
    # plt.ylabel('Precentage')
    # plt.grid(True)
    # plt.show()
    plt.savefig('feature_map_quant.pdf')
    plt.clf()

# with open("weight.pkl", "rb") as fread:
#     # feat_map = pickle.load(fread)

state_dict = torch.load('model.pth')
# print(state_dict['conv1.weight'])
feat_map = state_dict['conv1.weight'].cpu()
hist, bins, _ = plt.hist(feat_map.numpy().flatten(), 256, density=True, range=[-0.3, 0.3], color="#82e0aa")
# plt.show()
plt.savefig('weight.pdf')
plt.clf()

feat_map_quant = quantize(feat_map, num_bits=8, dequantize=True).numpy().flatten()
hist, bins, _ = plt.hist(feat_map_quant, 256, density=True, range=[-0.3, 0.3], color="#e74c3c")
# mu, sigma = scipy.stats.norm.fit(feat_map_quant)
# best_fit_line = scipy.stats.norm.pdf(bins, mu, sigma)
# plt.plot(bins, best_fit_line, color="red")
# plt.xlabel('(b)')
# print((hist * np.diff(bins)).sum())
# plt.ylabel('Precentage')
# plt.grid(True)
# plt.show()
plt.savefig('weight_quant.pdf')