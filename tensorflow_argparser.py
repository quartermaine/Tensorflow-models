import tensorflow as tf
import sys, os, pdb
import random
from argparse import ArgumentParser

STYLE_WEIGHT = 1e2

LEARNING_RATE = 1e-3
NUM_EPOCHS = 50
CHECKPOINT_ITERATIONS = 2000
DIR= 'C:\\Users\\quartermaine\\Documents\\Tensorflow\\Exercise-5'
TRAIN_PATH=DIR+"\\cats-v-dogs\\training"
TEST_PATH=DIR+"\\cats-v-dogs\\testing"
BATCH_SIZE = 10

def build_parser():
    parser = ArgumentParser()

    parser.add_argument('--style', type=str,
                        dest='style', help='style image path',
                        metavar='STYLE', required=False)

    parser.add_argument('--train-path', type=str,
                        dest='train_path', help='path to training images folder',
                        metavar='TRAIN_PATH', default=TRAIN_PATH)

    parser.add_argument('--test-path', type=str,
                        dest='test_path', help='test image path',
                        metavar='TEST', default=TEST_PATH)

    parser.add_argument('--slow', dest='slow', action='store_true',
                        help='gatys\' approach (for debugging, not supported)',
                        default=False)

    parser.add_argument('--epochs', type=int,
                        dest='epochs', help='num epochs',
                        metavar='EPOCHS', default=NUM_EPOCHS)

    parser.add_argument('--batch-size', type=int,
                        dest='batch_size', help='batch size',
                        metavar='BATCH_SIZE', default=BATCH_SIZE)

    parser.add_argument('--save-path',type=str,
                        dest='save_path', help='path to save model',
                        metavar='SAVE_PATH')

    parser.add_argument('--learning-rate', type=float,
                        dest='learning_rate',
                        help='learning rate (default %(default)s)',
                        metavar='LEARNING_RATE', default=LEARNING_RATE)

    parser.add_argument('--verbose',type=int,
                        dest='verbose',help='verbose for model',
                        metavar='VERBOSE',default=1)
    return parser

def check_opts(opts):
    if opts.style :
        pass
    else :
        print("style path not found!")
    if opts.train_path :
        pass
    else :
        print ("train path not found!")
    if opts.test_path:
        pass
    else :
        print( "test directory not found!")
    assert opts.epochs > 0
    assert opts.batch_size > 0
    assert opts.learning_rate >= 0
    assert opts.verbose >= 0


def make_model(options):
    model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
        ])

    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=options.learning_rate),
                  loss='binary_crossentropy', metrics=['acc'])
    return model


def main():
    parser = build_parser()
    options = parser.parse_args()
    check_opts(options)



    if options.train_path:
        TRAINING_DIR=options.train_path
    else :
        pass
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.,rotation_range=20)
    train_generator = train_datagen.flow_from_directory(TRAINING_DIR,
                                                    batch_size=options.batch_size,
                                                    class_mode='binary',
                                                    target_size=(150, 150))

    if options.test_path:
        VALIDATION_DIR=options.test_path
    else :
        pass

    validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.,rotation_range=20)
    validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,
                                                              batch_size=options.batch_size,
                                                              class_mode='binary',
                                                                  target_size=(150, 150))
    model=make_model(options)
    history = model.fit_generator(train_generator,
                                  epochs=options.epochs,
                                  verbose=options.verbose,
                                  validation_data=validation_generator)

    if options.save_path:
        model.save(options.save_path + '\model1.hdf5')



if __name__ == '__main__':
    main()

