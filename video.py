import cv2
from chainer import cuda, Variable, serializers
from net import *
import numpy as np

RUN_ON_GPU = True
CAMERA_ID = 1 # 0 for integrated cam, 1 for first external can ....
WIDTH=200
HEIGHT=200

model = FastStyleNet()

def _transform(in_image,loaded,m_path):
    if m_path == 'none':
        return in_image
    if not loaded:
        serializers.load_npz(m_path, model)
        if RUN_ON_GPU:
            cuda.get_device(0).use() #assuming only one core
            model.to_gpu()
        print "loaded"

    xp = np if not RUN_ON_GPU else cuda.cupy

    image = xp.asarray(in_image, dtype=xp.float32).transpose(2, 0, 1)
    image = image.reshape((1,) + image.shape)
    image -= 120

    x = Variable(image)
    y = model(x)

    result = cuda.to_cpu(y.data)
    result = result.transpose(0, 2, 3, 1)
    result = result.reshape((result.shape[1:]))
    result += 120
    result = np.uint8(result)

    return result


if __name__ == '__main__':
    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(CAMERA_ID)
    vc.set(3,WIDTH)
    vc.set(4,HEIGHT)

    if vc.isOpened():
        rval, frame = vc.read()
        loaded = False
        mpath = 'gafr-chainer-fast-neuralstyle-models-b3cf9b2/models/cubist.model'
    else:
        rval = False
    while rval:
        cv2.imshow("preview", frame)
        rval, frame = vc.read()
        frame = cv2.resize( _transform(frame,loaded,mpath), (0,0), fx=1.0, fy=1.00)

        loaded=True
        key = cv2.waitKey(20)
        if key == 49:
            mpath='gafr-chainer-fast-neuralstyle-models-b3cf9b2/models/cubist.model'
            loaded=False
        if key == 50:
            mpath='gafr-chainer-fast-neuralstyle-models-b3cf9b2/models/edtaonisl.model'
            loaded=False
        if key == 51:
            mpath='gafr-chainer-fast-neuralstyle-models-b3cf9b2/models/kandinsky_e2_crop512.model'
            loaded=False
        if key == 52:
            mpath='gafr-chainer-fast-neuralstyle-models-b3cf9b2/models/starrynight.model'
            loaded=False
        if key == 53:
            mpath='gafr-chainer-fast-neuralstyle-models-b3cf9b2/models/hokusai.model'
            loaded=False
        if key == 541:
            mpath='none'
            loaded=False
    cv2.destroyWindow("preview")