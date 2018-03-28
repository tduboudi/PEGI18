param = {
    'trainSet'    : '../data/FRAMES/Training',
    'trainLabels' : 'data/labels/train.json',
    'testSet'     : '../data/FRAMES/Testing',
    'testLabels'  : 'data/labels/test.json',

    'epoch' : 100,

    'saved_model' : 'data/checkpoints/images.current.hdf5',
    'seq_length'  : 40,
    'class_limit' : 2,
    'data_type'   : 'images',
    'image_shape' : (80, 80, 3)
}
