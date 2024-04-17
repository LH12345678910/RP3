import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping
from Datasets import x_train, y_train, x_val, y_val






def conv2d_block(input_tensor, n_filters, mom, filter_size):
  x = input_tensor
  for i in range(2):
    x = tf.keras.layers.Conv2D(filters = n_filters, kernel_size = (filter_size, filter_size), padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization(axis=-1,momentum=mom)(x)
    x = tf.keras.layers.Activation('relu')(x)
  return x


def encoder_block(inputs, mom, n_filters, filter_size, pool_size):
  c = conv2d_block(inputs, n_filters, mom, filter_size)
  p = tf.keras.layers.MaxPooling2D(pool_size = (pool_size, pool_size))(c)

  return c, p


def encoder(inputs, mom, n_filters, filter_size, pool_size):
  c1, p1 = encoder_block(inputs, mom, n_filters, filter_size, pool_size)
  c2, p2 = encoder_block(p1, mom, n_filters*2, filter_size, pool_size)
  c3, p3 = encoder_block(p2, mom, n_filters*4, filter_size, pool_size)
  c4, p4 = encoder_block(p3, mom, n_filters*8, filter_size, pool_size)

  return p4, (c1, c2, c3, c4)

def bottleneck(inputs, n_filters, mom, filter_size):
  bottle_neck = conv2d_block(inputs, n_filters*16, mom, filter_size)

  return bottle_neck

def decoder_block(inputs, conv_output, mom, n_filters, filter_size, stride):
  u = tf.keras.layers.Conv2DTranspose(n_filters, kernel_size = (filter_size, filter_size), strides = (stride, stride), padding = 'same')(inputs)
  c = tf.keras.layers.concatenate([u, conv_output])
  c = conv2d_block(c, n_filters, mom, filter_size)

  return c


def decoder(inputs, convs, mom, n_filters, filter_size, stride, output_channels):
  c1, c2, c3, c4 = convs

  c6 = decoder_block(inputs, c4, mom, n_filters*8, filter_size, stride)
  c7 = decoder_block(c6, c3, mom, n_filters*4, filter_size, stride)
  c8 = decoder_block(c7, c2, mom, n_filters*2, filter_size, stride)
  c9 = decoder_block(c8, c1, mom, n_filters, filter_size, stride)

  outputs = tf.keras.layers.Conv2D(output_channels, (1, 1), activation='sigmoid')(c9)

  return outputs



def unet(mom, n_filters, filter_size, pool_size, stride, output_channels):
  inputs = tf.keras.layers.Input(shape=(128, 128,1,))

  encoder_output, convs = encoder(inputs, mom, n_filters, filter_size, pool_size)

  bottle_neck = bottleneck(encoder_output, n_filters, mom, filter_size)

  outputs = decoder(bottle_neck, convs, mom, n_filters, filter_size, stride, output_channels)
  model = tf.keras.Model(inputs=inputs, outputs=outputs)

  return model
tf.keras.backend.clear_session()

model = unet(mom=0.1981, n_filters=16, filter_size=3, pool_size=2, stride=2, output_channels=3)

model.summary()






def WBCE(y_true, y_pred):
    y_true_shape = tf.shape(y_true)
    bce_loss = 0.0
    for i in range(y_true_shape[0]):
        y_true_f = y_true[i, :, :, 0]
        y_true_m = y_true[i, :, :, 1]  
        y_true_v = y_true[i, :, :, 2]
        y_true1 = y_true[i, :, :, 0]
        y_pred1 = y_pred[i, :, :, 0]
        y_true2 = y_true[i, :, :, 1]
        y_pred2 = y_pred[i, :, :, 1]
        y_true3 = y_true[i, :, :, 2]
        y_pred3 = y_pred[i, :, :, 2]
        w1 = 1-(K.sum(y_true_f)/(128*128))
        w2 = 1-(K.sum(y_true_m)/(128*128))
        w3 = 1-(K.sum(y_true_v)/(128*128))
        bce1 = K.mean(K.mean(K.binary_crossentropy(y_true1, y_pred1), axis=-1))
        bce2 = K.mean(K.mean(K.binary_crossentropy(y_true2, y_pred2), axis=-1))
        bce3 = K.mean(K.mean(K.binary_crossentropy(y_true3, y_pred3), axis=-1))
        bce_loss = w1*bce1 + w2*bce2 + w3*bce3 + bce_loss
    y_true_shape = tf.cast(y_true_shape, tf.float32)
    wbce = (bce_loss/y_true_shape[0])/3
    return wbce


def GDL(y_true, y_pred):
    def dice_coef2(y_true, y_pred):
        dice = 0.0
        y_true_shape = tf.shape(y_true)

        for i in range(y_true_shape[0]):
            y_true_f1 = y_true[i, :, :, 0]
            y_pred_f1 = y_pred[i, :, :, 0]
            y_true_f2 = y_true[i, :, :, 1]
            y_pred_f2 = y_pred[i, :, :, 1]  
            y_true_f3 = y_true[i, :, :, 2]
            y_pred_f3 = y_pred[i, :, :, 2]

            intersection1 = K.sum(y_true_f1 * y_pred_f1)
            intersection2 = K.sum(y_true_f2 * y_pred_f2)
            intersection3 = K.sum(y_true_f3 * y_pred_f3)
            w1 = 1/(K.sum(y_true_f1)*K.sum(y_true_f1))
            w2 = 1/(K.sum(y_true_f2)*K.sum(y_true_f2))
            w3 = 1/(K.sum(y_true_f3)*K.sum(y_true_f3))
            num = (w1*intersection1) + (w2*intersection2) + (w3*intersection3)
            union1 = K.sum(y_true_f1) + K.sum(y_pred_f1)
            union2 = K.sum(y_true_f2) + K.sum(y_pred_f2)
            union3 = K.sum(y_true_f3) + K.sum(y_pred_f3)
            den = (w1*union1) + (w2*union2) + (w3*union3)
            dice = ((2. * num) / (den)) + dice
        y_true_shape = tf.cast(y_true_shape, tf.float32)
        m1 = dice/y_true_shape[0]
        return m1
    return 1 - dice_coef2(y_true, y_pred)

a = 0.0495
def combined_loss(y_true, y_pred):
    def calculate_combined_loss(y_true, y_pred, alpha=a):
        dice = GDL(y_true, y_pred)
        ce = WBCE(y_true, y_pred)
        cl = alpha*ce + (1-alpha)*dice
        return cl
    return calculate_combined_loss(y_true, y_pred)


def DL1(y_true, y_pred):
    y_true = y_true[:, :, :, 0]
    y_pred = y_pred[:, :, :, 0]
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.abs(y_true * y_pred)
    mask = K.abs(y_true) + K.abs(y_pred)
    union = K.sum(mask)
    return 2*K.sum(intersection) / union

def DL2(y_true, y_pred):
    y_true = y_true[:, :, :, 1]
    y_pred = y_pred[:, :, :, 1]
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.abs(y_true * y_pred)
    mask = K.abs(y_true) + K.abs(y_pred)
    union = K.sum(mask)
    return 2*K.sum(intersection) / union

def DL3(y_true, y_pred):
    y_true = y_true[:, :, :, 2]
    y_pred = y_pred[:, :, :, 2]
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.abs(y_true * y_pred)
    mask = K.abs(y_true) + K.abs(y_pred)
    union = K.sum(mask)
    return 2*K.sum(intersection) /union

model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0513,beta_1=0.7662,beta_2=0.9656), loss=combined_loss,
              metrics=[DL1,
                       DL2,
                       DL3,
                       tf.keras.metrics.BinaryAccuracy()])


EPOCHS = 1500

monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, 
                        patience=20, verbose=1, mode='min',
                                restore_best_weights=True)

model.fit(x_train,
          y_train,
          batch_size=26,
          epochs=EPOCHS,
          validation_data=(x_val, y_val),
          callbacks=[monitor],
          shuffle=True)

epochs = monitor.stopped_epoch


if 1 == 1:
    sample_images = x_val
    segmented_images = model.predict(sample_images)

    for i in range(segmented_images.shape[0]):
        z = segmented_images[i,:,:,:]
 
        max_values = np.max(z, axis=-1)

        output_image = np.zeros_like(z)

        output_image[np.where(z == max_values[..., None])] = 1
        segmented_images[i,:,:,:]=output_image

