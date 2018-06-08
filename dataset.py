from imgaug import augmenters as iaa
from keras.preprocessing.image import *
import cv2
from tqdm import tqdm
from config import *

def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]
    if center is None:
        center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def process(aug,model,width,fnames_test,n_test):
    X_test = np.zeros((n_test, width, width, 3), dtype=np.uint8)

    if (aug == 'default'):
        for i in tqdm(range(n_test)):
            img = cv2.resize(cv2.imread(TEST_IMG_DIR+'{0}'.format(fnames_test[i])), (width, width))
            X_test[i] = img[:, :, ::-1]
    elif (aug == 'flip'):
        for i in tqdm(range(n_test)):
            img = cv2.resize(cv2.imread(TEST_IMG_DIR+'{0}'.format(fnames_test[i])), (width, width))
            img = cv2.flip(img, 1)
            X_test[i] = img[:, :, ::-1]
    elif (aug == 'rotate1'):
        for i in tqdm(range(n_test)):
            img = cv2.resize(cv2.imread(TEST_IMG_DIR+'{0}'.format(fnames_test[i])), (width, width))
            img = rotate(img, 5)
            X_test[i] = img[:, :, ::-1]
    elif (aug == 'rotate2'):
        for i in tqdm(range(n_test)):
            img = cv2.resize(cv2.imread(TEST_IMG_DIR+'{0}'.format(fnames_test[i])), (width, width))
            img = rotate(img, -5)
            X_test[i] = img[:, :, ::-1]
    elif (aug == 'rotate3'):
        for i in tqdm(range(n_test)):
            img = cv2.resize(cv2.imread(TEST_IMG_DIR+'{0}'.format(fnames_test[i])), (width, width))
            img = cv2.flip(img, 1)
            img = rotate(img, 5)
            X_test[i] = img[:, :, ::-1]
    elif (aug == 'rotate4'):
        for i in tqdm(range(n_test)):
            img = cv2.resize(cv2.imread(TEST_IMG_DIR+'{0}'.format(fnames_test[i])), (width, width))
            img = cv2.flip(img, 1)
            img = rotate(img, -5)
            X_test[i] = img[:, :, ::-1]
    elif (aug == 'rotate5'):
        for i in tqdm(range(n_test)):
            img = cv2.resize(cv2.imread(TEST_IMG_DIR+'{0}'.format(fnames_test[i])), (width, width))
            img = rotate(img, 13)
            X_test[i] = img[:, :, ::-1]
    elif (aug == 'rotate6'):
        for i in tqdm(range(n_test)):
            img = cv2.resize(cv2.imread(TEST_IMG_DIR+'{0}'.format(fnames_test[i])), (width, width))
            img = rotate(img, -13)
            X_test[i] = img[:, :, ::-1]
    elif (aug == 'rotate7'):
        for i in tqdm(range(n_test)):
            img = cv2.resize(cv2.imread(TEST_IMG_DIR+'{0}'.format(fnames_test[i])), (width, width))
            img = cv2.flip(img, 1)
            img = rotate(img, 13)
            X_test[i] = img[:, :, ::-1]
    elif (aug == 'rotate8'):
        for i in tqdm(range(n_test)):
            img = cv2.resize(cv2.imread(TEST_IMG_DIR+'{0}'.format(fnames_test[i])), (width, width))
            img = cv2.flip(img, 1)
            img = rotate(img, -13)
            X_test[i] = img[:, :, ::-1]
    elif (aug == 'rotate9'):
        for i in tqdm(range(n_test)):
            img = cv2.resize(cv2.imread(TEST_IMG_DIR+'{0}'.format(fnames_test[i])), (width, width))
            img = rotate(img, 21)
            X_test[i] = img[:, :, ::-1]
    elif (aug == 'rotate10'):
        for i in tqdm(range(n_test)):
            img = cv2.resize(cv2.imread(TEST_IMG_DIR+'{0}'.format(fnames_test[i])), (width, width))
            img = rotate(img, -21)
            X_test[i] = img[:, :, ::-1]
    elif (aug == 'rotate11'):
        for i in tqdm(range(n_test)):
            img = cv2.resize(cv2.imread(TEST_IMG_DIR+'{0}'.format(fnames_test[i])), (width, width))
            img = cv2.flip(img, 1)
            img = rotate(img, 21)
            X_test[i] = img[:, :, ::-1]
    elif (aug == 'rotate12'):
        for i in tqdm(range(n_test)):
            img = cv2.resize(cv2.imread(TEST_IMG_DIR+'{0}'.format(fnames_test[i])), (width, width))
            img = cv2.flip(img, 1)
            img = rotate(img, -21)
            X_test[i] = img[:, :, ::-1]

    y_pred = model.predict(X_test, batch_size=32, verbose=1)
    del X_test
    return y_pred


def customizedImgAug(input_img):
    rarely = lambda aug: iaa.Sometimes(0.1, aug)
    sometimes = lambda aug: iaa.Sometimes(0.25, aug)
    often = lambda aug: iaa.Sometimes(0.5, aug)

    seq = iaa.Sequential([
        iaa.Fliplr(0.5),
        often(iaa.Affine(
            scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
            translate_percent={"x": (-0.1, 0.1), "y": (-0.12, 0)},
            rotate=(-10, 10),
            shear=(-8, 8),
            order=[0, 1],
            cval=(0, 255),
        )),
        iaa.SomeOf((0, 4), [
            rarely(
                iaa.Superpixels(
                    p_replace=(0, 0.3),
                    n_segments=(20, 200)
                )
            ),
            iaa.OneOf([
                iaa.GaussianBlur((0, 2.0)),
                iaa.AverageBlur(k=(2, 4)),
                iaa.MedianBlur(k=(3, 5)),
            ]),
            iaa.Sharpen(alpha=(0, 0.3), lightness=(0.75, 1.5)),
            iaa.Emboss(alpha=(0, 1.0), strength=(0, 0.5)),
            rarely(iaa.OneOf([
                iaa.EdgeDetect(alpha=(0, 0.3)),
                iaa.DirectedEdgeDetect(
                    alpha=(0, 0.7), direction=(0.0, 1.0)
                ),
            ])),
            iaa.AdditiveGaussianNoise(
                loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5
            ),
            iaa.OneOf([
                iaa.Dropout((0.0, 0.05), per_channel=0.5),
                iaa.CoarseDropout(
                    (0.03, 0.05), size_percent=(0.01, 0.05),
                    per_channel=0.2
                ),
            ]),
            rarely(iaa.Invert(0.05, per_channel=True)),
            often(iaa.Add((-40, 40), per_channel=0.5)),
            iaa.Multiply((0.7, 1.3), per_channel=0.5),
            iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),
            iaa.Grayscale(alpha=(0.0, 1.0)),
            sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.03))),
            sometimes(
                iaa.ElasticTransformation(alpha=(0.5, 1.5), sigma=0.25)
            ),

        ], random_order=True),
        iaa.Fliplr(0.5),
        iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True)
    ], random_order=True)  # apply augmenters in random order

    output_img = seq.augment_image(input_img)
    return output_img


class Generator():
    def __init__(self, X, y, batch_size=8, aug=False):
        def generator():
            while True:
                for i in range(0, len(X), batch_size):
                    X_batch = X[i:i + batch_size].copy()
                    y_barch = [x[i:i + batch_size] for x in y]
                    if aug:
                        for j in range(len(X_batch)):
                            X_batch[j] = customizedImgAug(X_batch[j])
                    yield X_batch, y_barch

        self.generator = generator()
        self.steps = len(X) // batch_size + 1
