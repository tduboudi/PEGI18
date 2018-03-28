"""
Train our RNN on extracted features or images.
"""
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from models import ResearchModels
from data import DataSet
import time
import os.path
from images_loader import ImageLoader
import os

from parameters import param

def train(ImageLoader, data_type, seq_length, saved_model=None, class_limit=None, image_shape=None, load_to_memory=False, batch_size=32, nb_epoch=100):
    checkpointer = ModelCheckpoint( filepath=os.path.join('data', 'checkpoints', data_type + '.{epoch:03d}-{acc:.3f}.hdf5'), verbose=1, save_best_only=False)
    tb = TensorBoard(log_dir=os.path.join('data', 'logs'))
    early_stopper = EarlyStopping(patience=5)

    timestamp = time.time()

    X, y = ImageLoader.load()
    rm = ResearchModels(class_limit, seq_length, saved_model)

    rm.model.fit(X, y, batch_size=batch_size, verbose=1, callbacks=[tb, early_stopper, checkpointer], epochs=nb_epoch)

def main():
    seq_length  = param['seq_length']
    data_type   = param['data_type']
    saved_model = None
    class_limit = param['class_limit']
    image_shape = param['image_shape']

    load_to_memory = True
    batch_size = 32
    epoch = param['epoch']

    trainSequenceLoader = ImageLoader(param['trainSet'], param['trainLabels'], seq_length, image_shape)
    train(trainSequenceLoader, data_type, seq_length, saved_model=saved_model, class_limit=class_limit, image_shape=image_shape, load_to_memory=load_to_memory, batch_size=batch_size, nb_epoch=epoch)

if __name__ == '__main__':
    main()
