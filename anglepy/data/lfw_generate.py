import numpy as np
from sklearn.datasets import fetch_lfw_people

# 'big' images (scale factor 0.5)
lfw_people = fetch_lfw_people(color=True)
np.save('lfw_62x47', lfw_people.images.astype('uint8'))
np.save('lfw_labels', lfw_people.target)

# 'small' images (scale factor 0.25)
lfw_people2 = fetch_lfw_people(color=True, resize=0.25)
np.save('lfw_31x23', lfw_people2.images.astype('uint8'))

