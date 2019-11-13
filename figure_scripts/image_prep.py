import imageio
import numpy as np
import matplotlib.pyplot as plt
import pickle

image = imageio.imread("/Users/clemens/Desktop/smile.jpg")

image = np.sum(image, axis=2)
#image = image/np.max(image)


#image[np.logical_and((image < 380),(image > 320))] = 1 
#image[image != 1] = 0

image[image < 500] = 1 
image[image != 1] = 0

plt.imshow(image)
plt.show()

#plt.imshow(image)
#plt.show()
with open('/Users/clemens/Desktop/smile100.pkl', 'wb') as f:
    pickle.dump(image, f)



noise = np.random.binomial(1, 0.95, image.shape[0]*image.shape[1]).reshape([image.shape[0], image.shape[1]])
image07 = image * noise

with open('/Users/clemens/Desktop/smile95.pkl', 'wb') as f:
    pickle.dump(image07, f)

plt.imshow(image07)
plt.show()

noise = np.random.binomial(1, 0.5, image.shape[0]*image.shape[1]).reshape([image.shape[0], image.shape[1]])
image07 = image * noise

with open('/Users/clemens/Desktop/smile50.pkl', 'wb') as f:
    pickle.dump(image07, f)

plt.imshow(image07)
plt.show()

noise = np.random.binomial(1, 0.3, image.shape[0]*image.shape[1]).reshape([image.shape[0], image.shape[1]])
image07 = image * noise

with open('/Users/clemens/Desktop/smile30.pkl', 'wb') as f:
    pickle.dump(image07, f)



plt.imshow(image07)
plt.show()
