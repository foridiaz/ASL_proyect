"""

"""
import logging
import numpy as np
from glob import glob
from matplotlib import pyplot as plt
import skimage.io as sk
import os
from skimage import exposure
from skimage.filters import threshold_otsu
# Información para el logger
log = logging.getLogger('ASL')
log.setLevel(level=logging.DEBUG)
fh = logging.StreamHandler()
fh_formatter = logging.Formatter('%(asctime)s %(levelname)s Line %(lineno)d:%(filename)s - %(message)s')
fh.setFormatter(fh_formatter)
log.addHandler(fh)
#log.basicConfig(level=logging.DEBUG, format='%(asctime)s-%(levelname)s-%(message)s')

# Obtengo los nombres de todas las carpetas para clasificar las imágenes
directories = [x[0] for x in os.walk('asl_alphabet_train')][1:]
# Almaceno las direcciones de las imágenes en un diccionario
Images = {}
log.debug('Almacenando directorios')
for i in directories:
    if 'nothing' in i:
        Images['nothing'] = glob(os.path.join(i, '*.jpg'))  # [sk.imread(j) for j in glob(os.path.join(i, '*.jpg'))]
    elif 'space' in i:
        Images['space'] = glob(os.path.join(i, '*.jpg'))  # [sk.imread(j) for j in glob(os.path.join(i, '*.jpg'))]
    elif 'del' in i:
        Images['del'] = glob(os.path.join(i, '*.jpg'))  # [sk.imread(j) for j in glob(os.path.join(i, '*.jpg'))]
    else:
        Images[str(i[-1])] = glob(os.path.join(i, '*.jpg')) # [sk.imread(j) for j in glob(os.path.join(i, '*.jpg'))]
# Cargo las imágenes


def ycbcr2rgb(im):
    xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
    rgb = im.astype(np.float)
    rgb[:,:,[1,2]] -= 128
    rgb = rgb.dot(xform.T)
    np.putmask(rgb, rgb > 255, 255)
    np.putmask(rgb, rgb < 0, 0)
    return np.uint8(rgb)

log.debug('Iniciando test')
#Imagen a cargar
test = np.array(ycbcr2rgb(sk.imread(Images['A'][1])))
eq = exposure.equalize_hist(test[:,:,0])
eq1 = exposure.equalize_hist(test[:,:,1])
eq2 = exposure.equalize_hist(test[:,:,2])
ot0 = threshold_otsu(test[:,:,0])
ot1 = threshold_otsu(test[:,:,1])
ot2 = threshold_otsu(test[:,:,2])
#GAMMA
g = 0.5
gamma0 = exposure.adjust_gamma(test[:,:,0], g)
gamma1 = exposure.adjust_gamma(test[:,:,1], g)
gamma2 = exposure.adjust_gamma(test[:,:,2], g)
log.debug(str(eq.shape))
plt.figure()
plt.subplot(231)
plt.title('Canal 0')
plt.imshow(test[:,:,0], cmap='gray')
plt.subplot(232)
plt.title('Canal 1')
plt.imshow(test[:,:,1], cmap='gray')
plt.subplot(233)
plt.title('Canal 2')
plt.imshow(test[:,:,2], cmap='gray')
plt.subplot(234)
plt.title('Canal 0 Gamma: '+str(g))
eq[eq>0.4] = 1
plt.imshow(gamma0, cmap='gray')
plt.subplot(235)
plt.title('Canal 1 Gamma: '+str(g))
plt.imshow(gamma1, cmap='gray')
plt.subplot(236)
plt.title('Canal 2 Gamma: '+str(g))
plt.imshow(gamma2, cmap='gray')

plt.show()