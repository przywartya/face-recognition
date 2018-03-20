from skimage import data, color, feature, io
from matplotlib import pyplot as plt


file_name = '../static/broda.jpg'
image = io.imread(file_name)
image = color.rgb2gray(image)
hog_vec, hog_vis = feature.hog(image, visualise=True, block_norm='L2-Hys')

fig, ax = plt.subplots(1, 2, figsize=(12, 6),
                       subplot_kw=dict(xticks=[], yticks=[]))
ax[0].imshow(image, cmap='gray')
ax[0].set_title('input image')

ax[1].imshow(hog_vis, cmap='gray')
ax[1].set_title('visualization of HOG features')

plt.show()
