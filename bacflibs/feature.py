import cv2
import numpy as np
# from .bacffeatures import fhog, TableFeature
from .bacffeatures import TableFeature
from .lookup_tables import fhog as pyfhog


# def extract_hog_feature(img, cell_size=4):
#     fhog_feature = fhog(img.astype(np.float32),cell_size,num_orients=9,clip=0.2)[:,:,:-1]
#     return fhog_feature

def extract_pyhog_feature(img, cell_size=4):
    h,w=img.shape[:2]
    img=cv2.resize(img,(w+2*cell_size,h+2*cell_size))
    mapp = {'sizeX': 0, 'sizeY': 0, 'numFeatures': 0, 'map': 0}
    mapp = pyfhog.getFeatureMaps(img, cell_size, mapp)
    mapp = pyfhog.normalizeAndTruncate(mapp, 0.2)
    mapp = pyfhog.PCAFeatureMaps(mapp)
    size_patch = list(map(int, [mapp['sizeY'], mapp['sizeX'], mapp['numFeatures']]))
    FeaturesMap = mapp['map'].reshape(
        (size_patch[0],size_patch[1], size_patch[2]))
    return FeaturesMap


def extract_cn_feature(img,cell_size=1):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255 - 0.5
    cn = TableFeature(fname='cn', cell_size=cell_size, compressed_dim=11, table_name="CNnorm",
                      use_for_color=True)

    if np.all(img[:, :, 0] == img[:, :, 1]):
        img = img[:, :, :1]
    else:
        # # pyECO using RGB format
        img = img[:, :, ::-1]
    h,w=img.shape[:2]
    cn_feature = \
    cn.get_features(img, np.array(np.array([h/2,w/2]), dtype=np.int16), np.array([h,w]), 1, normalization=False)[
        0][:, :, :, 0]
    # print('cn_feature.shape:', cn_feature.shape)
    # print('cnfeature:',cn_feature.shape,cn_feature.min(),cn_feature.max())
    gray = cv2.resize(gray, (cn_feature.shape[1], cn_feature.shape[0]))[:, :, np.newaxis]
    out = np.concatenate((gray, cn_feature), axis=2)
    return out
