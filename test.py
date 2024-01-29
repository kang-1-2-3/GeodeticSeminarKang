import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')
true = np.load('mask_true.npy')
pred= np.load('mask_pred.npy')

# plt.imshow(true, cmap='gray')

plt.imshow(pred, cmap='gray')
plt.show()