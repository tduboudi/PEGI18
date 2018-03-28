"""
Train our RNN on extracted features or images.
"""
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from models import ResearchModels
from data import DataSet
import time
import os.path
from images_loader import SequenceLoader
import os

from parameters import param

def train(SequenceLoader, data_type, seq_length, saved_model=None, class_limit=None, image_shape=None, load_to_memory=False, batch_size=32, nb_epoch=100):
    checkpointer = ModelCheckpoint( filepath=os.path.join('data', 'checkpoints', data_type + '.{epoch:03d}-{acc:.3f}.hdf5'), verbose=1, save_best_only=False)

    # Helper: TensorBoard
    tb = TensorBoard(log_dir=os.path.join('data', 'logs'))

    # Helper: Stop when we stop learning.
    early_stopper = EarlyStopping(patience=5)

    # Get data.
    X, y = SequenceLoader.loadSequencesToMemory()

    # Get the model.
    rm = ResearchModels(class_limit, seq_length, saved_model)

    rm.model.fit(X, y, batch_size=batch_size, verbose=1, callbacks=[tb, early_stopper, checkpointer], epochs=nb_epoch)


def main():
    saved_model = None  # None or weights file
    class_limit = param['class_limit']
    seq_length  = param['seq_length'] 
    load_to_memory = True  # pre-load the sequences into memory
    batch_size = 32
    nb_epoch = param['epoch']

    data_type = param['data_type']
    image_shape = param['image_shape']

    trainSequenceLoader = SequenceLoader(param['trainSet'], param['trainLabels'], seq_length)

    train(trainSequenceLoader, data_type, seq_length, saved_model=saved_model, class_limit=class_limit, image_shape=image_shape, load_to_memory=load_to_memory, batch_size=batch_size, nb_epoch=nb_epoch)

if __name__ == '__main__':
    main()
