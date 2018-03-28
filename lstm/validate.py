"""
Validate our RNN. Basically just runs a validation generator on
about the same number of videos as we have in our test set.
"""
from keras.callbacks import TensorBoard, ModelCheckpoint, CSVLogger
from models import ResearchModels
from data import DataSet
from images_loader import SequenceLoader

from parameters import param

def validate(data_type, seq_length=50, saved_model=None, class_limit=None, image_shape=None):

    sequenceLoader = SequenceLoader(param['testSet'], param['testLabels'], seq_length)
    X, y = sequenceLoader.loadSequencesToMemory()

    # Get the model.
    rm = ResearchModels(class_limit,  seq_length, saved_model)

    # Evaluate!
    results = rm.model.evaluate(X, y)

    print(results)
    print(rm.model.metrics_names)

def main():
    saved_model = param['saved_model']
    data_type   = param['data_type']
    image_shape = param['image_shape']
    class_limit = param['class_limit']

    validate(data_type, saved_model=saved_model, image_shape=image_shape, class_limit=class_limit)

if __name__ == '__main__':
    main()
