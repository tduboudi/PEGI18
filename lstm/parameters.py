param = {
    'inputTrainImages' : '/home/p3a3_1718/data/FRAMES/Training',
    'inputTrainLabels' : '/home/p3a3_1718/data/preprocessing/training_expanded.json',
    'inputTestImages'  : '/home/p3a3_1718/data/FRAMES/Testing',
    'inputTestLabels'  : '/home/p3a3_1718/data/preprocessing/testing_expanded.json',
    
    'outputTrainSequences' : 'data/sequences/train/features',
    'outputTrainLabels'    : 'data/sequences/train/labels',
    'outputTestSequences'  : 'data/sequences/test/features',
    'outputTestLabels'     : 'data/sequences/test/labels',

    'trainSet'    : 'data/sequences/train',
    'trainLabels' : 'data/labels/train.json',
    'testSet'     : 'data/sequences/test',
    'testLabels'  : 'data/labels/test.json',

    'epoch' : 100,

    'saved_model' : 'data/checkpoints/features.current.hdf5',
    'seq_length'  : 50,
    'class_limit' : 2,
    'data_type'   : 'features',
    'image_shape' : None
}
