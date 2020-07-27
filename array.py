from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

plt.subplot(221)
img_dir1= 'D:\\EdoublemA\\master\\Emma-Semantic-Segmentation\\chalkynet1\\dataset\\jpg\\0001TP_006690.jpg'
img1 = np.array(Image.open(img_dir1))
print(img1.max())
plt.imshow(img1)
print(type(img1))
plt.subplot(222)
# img_dir2= 'D:\\EdoublemA\\rice\\Emma-Semantic-Segmentation\\chalkynet1\\dataset2\\10000\\png\\0001TP_006690.png'
# img2 = np.array(Image.open(img_dir2))
# print(img2.max())
# plt.imshow(img2)
# print(type(img2))
# plt.subplot(223)
img_dir2= 'D:\\EdoublemA\\master\\Emma-Semantic-Segmentation\\chalkynet1\\dataset\\png\\0001TP_006840.png'
img2 = np.array(Image.open(img_dir2))
print(img2.max())
plt.imshow(img2)
print(type(img2))
plt.show()
# img = Image.fromarray(img)
# img.save(img_dir)