import tensorflow as tf
import sys, os
from sklearn.model_selection import train_test_split
import numpy as np

nb_classes = 100

tf.app.flags.DEFINE_integer("epochs", 30, "number of training epochs")
tf.app.flags.DEFINE_integer("batch_size", 100, "number of batch size")
tf.app.flags.DEFINE_float("learning_rate", 1e-3, "learning rate")
tf.app.flags.DEFINE_integer("verbose", 1, "verbose")
tf.app.flags.DEFINE_string("save_path", None, "saving directory")
FLAGS = tf.app.flags.FLAGS

def check_args(FLAGS):
    print("...checking FLAGS")
    if len(sys.argv) < 2:
        print('Usage: CIFAR10_with_FLAGS.py [--epochs=int] '
              '[--batch_size=int] [--learning_rate=float] [--verbose=int] [--save_path]')
        sys.exit(-1)

    arg_list = ["epochs", "batch_size", "vebose"]
    filtered_dict = {k: v for k, v in tf.app.flags.FLAGS.flag_values_dict().items() if k in arg_list}
    listOfKeys = [attr for (attr, value) in filtered_dict.items() if value < 0]
    if len(listOfKeys) > 0:
        print('Please specify a positive value for ', listOfKeys, '.')
        sys.exit(-1)


def make_model(FLAGS, shape_input):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Convolution2D(32, (3, 3), padding='same', activation="relu",
                                      input_shape=shape_input),
        tf.keras.layers.Convolution2D(32, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Convolution2D(64, (3, 3), padding='same', activation="relu"),
        tf.keras.layers.Convolution2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(100, activation="softmax"),
    ])

    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=FLAGS.learning_rate),
                      loss='binary_crossentropy', metrics=['acc'])
    return model

def main(_):
    #FLAGS._parse_flags()

    check_args(FLAGS)

    (X, y), (X_test, y_test) = tf.keras.datasets.cifar100.load_data()
    X = X.astype("float32") / 255.
    X_test = X_test.astype("float32") / 255.
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20)
    y_train = tf.keras.utils.to_categorical(y_train, nb_classes)
    y_test = tf.keras.utils.to_categorical(y_test, nb_classes)
    y_val = tf.keras.utils.to_categorical(y_val, nb_classes)

    cif = make_model(FLAGS, X_train.shape[1:])

    if FLAGS.save_path :
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')
        tensorboard = tf.keras.callbacks.TensorBoard(log_dir=FLAGS.save_path + "\cifar100_graph", histogram_freq=0,
                                                     write_graph=True, write_images=True)
        model_saver = tf.keras.callbacks.ModelCheckpoint(filepath=FLAGS.save_path + '\cifar100_weights', verbose=0, period=2)

        history = cif.fit(X_train, y_train, batch_size=FLAGS.batch_size,
                          epochs=FLAGS.epochs, validation_data=(X_val, y_val),
                          verbose=FLAGS.verbose, callbacks=[early_stop, tensorboard, model_saver], shuffle=True)

        cif.save(FLAGS.save_path + '\cifar100_model.hdf5')

    else :
        history = cif.fit(X_train, y_train, batch_size=FLAGS.batch_size,
                          epochs=FLAGS.epochs, validation_data=(X_val, y_val),
                          verbose=FLAGS.verbose, shuffle=True)
    print('training is finished!')

    print('\n# Evaluate on test data')
    results = cif.evaluate(X_test, y_test, batch_size=128)
    if FLAGS.save_path :
        print("Test Loss : {0}, Test Accuracy : {1}".format(results[0] * 100, results[1] * 100))
        print("To see tensorboard use tensorboard --logdir==training:{}\cifar100_graph --host=127.0.0.1".format(FLAGS.save_path))
    else :
        print("Test Loss : {0}, Test Accuracy : {1}".format(results[0] * 100, results[1] * 100))


if __name__ == '__main__':
    tf.app.run()



