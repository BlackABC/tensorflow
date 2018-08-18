from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Activation
from keras.layers import Flatten, Dense, BatchNormalization, Dropout, PReLU
from keras.models import Model, Input
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard
from keras.optimizers import SGD
import numpy as np
import os


def bn_prelu(x):
    x = BatchNormalization()(x)
    x = PReLU()(x)
    return x


def bulid_model(out_dims, input_shape=(128, 128, 1)):
    input_dims = Input(input_shape)

    x = Conv2D(32, (3, 3), strides=(2, 2), padding='valid')(input_dims)
    x = bn_prelu(x)
    x = Conv2D(32, (3, 3), strides=(1, 1), padding='valid')(x)
    x = bn_prelu(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(64, (3, 3), strides=(1, 1), padding='valid')(x)
    x = bn_prelu(x)
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='valid')(x)
    x = bn_prelu(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(128, (3, 3), strides=(1, 1), padding='valid')(x)
    x = bn_prelu(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(128, (3, 3), strides=(1, 1), padding='valid')(x)
    x = bn_prelu(x)
    x = AveragePooling2D(pool_size=(2, 2))(x)

    x_flatten = Flatten()(x)

    fc1 = Dense(512)(x_flatten)
    fc1 = bn_prelu(fc1)
    dp_1 = Dropout(0.3)(fc1)

    fc2 = Dense(out_dims)(dp_1)
    fc2 = Activation('softmax')(fc2)

    model = Model(inputs=input_dims, outputs=fc2)
    return model


def lrschedule(epoch):
    if epoch < 40:
        return 0.1
    elif epoch < 80:
        return 0.01
    else:
        return 0.001


def model_train(model, loadweights, isCenterloss=False, lambda_center=False):
    lr = LearningRateScheduler(lrschedule)
    mdcheck = ModelCheckpoint(WEIGHTS_PATH, monitor='val_acc', save_best_only=True)
    td = TensorBoard(log_dir='tensorboard_log/')

    if loadweights:
        if os.path.isfile(WEIGHTS_PATH):
            assert model.load_weights(WEIGHTS_PATH)
            print("model have load pre weights of hanzi image !!")
        else:
            print("model not load weight!!")
    else:
        print("not load weight!!")

    sgd = SGD(lr=0.1, momentum=0.9, decay=5e-4, nesterov=True)
    # if not isCenterloss:
    print("model compile!")
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    print("model training!")
    history = model.fit_generator(train_generator,
                                  steps_per_epoch=32000,
                                  epochs=max_Epochs,
                                  validation_data=val_generator,
                                  validation_steps=8000,
                                  callbacks=[lr, mdcheck, td])
    return history


if __name__ == "__main__":
    train_path = 'image_data/train1/'
    val_path = 'image_data/val/'
    test_path = 'image_data/test/'
    num_classes = 100
    BATCH_SIZE = 128
    WEIGHTS_PATH = 'best_weights_hanzi.hdf5'
    max_Epochs = 100

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        horizontal_flip=True
    )

    val_datagen = ImageDataGenerator(
        rescale=1. / 255
    )

    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(128, 128),
        batch_size=BATCH_SIZE,
        color_mode='grayscale',
        class_mode='categorical'
    )

    val_generator = val_datagen.flow_from_directory(
        val_path,
        target_size=(128, 128),
        batch_size=BATCH_SIZE,
        color_mode='grayscale',
        class_mode='categorical'
    )

    simple_model = bulid_model(num_classes)
    print(simple_model.summary())

    print("=====start train image of epoch=====")

    model_history = model_train(simple_model, False)

    print("=====test label=====")
    simple_model.load_weights(WEIGHTS_PATH)
    model = simple_model
