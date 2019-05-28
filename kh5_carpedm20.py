from keras.models import Sequential, Model
from keras import layers
import keras
import numpy as np
import os
from PIL import Image
from keras.preprocessing import image
import glob
import cv2
import scipy
import scipy.misc



class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


## Generator
"""
carpedm20's generator is quasi-identical to soumith's
except at the 1st layer: carpedm20's was easier, just
a fully connected layer and a reshape.
"""

latent_dim = 100
#n_channels_large = 128
n_channels_large = 64
z = keras.Input(shape=(latent_dim,))

""" Replaced 
gen_img = layers.Reshape((1,1,latent_dim))(z)
# plane shape: (4,4,128*8)
gen_img = layers.Conv2DTranspose(filters=n_channels_large*8, kernel_size=4,
                                 strides=1, padding='valid',
                                 input_shape=(1,1,100))(gen_img)
gen_img = layers.BatchNormalization(axis=-1)(gen_img)
gen_img = layers.Activation('relu')(gen_img)
# Note that we follow Soumith and place the activation after BN.
"""

# new 1st layer
first_plane_shape = (4, 4, n_channels_large*8)
gen_img = layers.Dense(np.prod(first_plane_shape))(z)
gen_img = layers.Reshape(first_plane_shape)(gen_img)
gen_img = layers.BatchNormalization(axis=-1)(gen_img)
gen_img = layers.Activation('relu')(gen_img)



# plane shape: (8,8,128*4)
gen_img = layers.Conv2DTranspose(filters=n_channels_large*4, kernel_size=5,
                                 strides=2, padding='same')(gen_img)
gen_img = layers.BatchNormalization(axis=-1)(gen_img)
gen_img = layers.Activation('relu')(gen_img)


# plane shape: (16,16,128*2)
gen_img = layers.Conv2DTranspose(filters=n_channels_large*2, kernel_size=5,
                                 strides=2, padding='same')(gen_img)
gen_img = layers.BatchNormalization(axis=-1)(gen_img)
gen_img = layers.Activation('relu')(gen_img)


# plane shape: (32,32,128)
gen_img = layers.Conv2DTranspose(filters=n_channels_large, kernel_size=5,
                                 strides=2, padding='same')(gen_img)
gen_img = layers.BatchNormalization(axis=-1)(gen_img)
gen_img = layers.Activation('relu')(gen_img)


# plane shape: (64,64,3)
gen_img = layers.Conv2DTranspose(filters=3, kernel_size=5,
                                 strides=2, padding='same',
                                 activation='tanh')(gen_img)

generator = Model(z, gen_img)
generator.summary()


# Discriminator
"""
carpedm20's discriminator is quasi-identical to soumith's
except at the 1st layer: carpedm20's was easier, just
the final layer being diff.
"""


ndf = 64  # ndf or NDF: Number of Discriminator Filters
input_img = layers.Input(shape=(64,64,3))

# plane shape: (32,32,ndf)
d_output_proba = layers.Conv2D(filters=ndf, kernel_size=5,
                               strides=2, padding='same')(input_img)
d_output_proba = layers.LeakyReLU(alpha=0.2)(d_output_proba)

# plane shape: (16,16,ndf*2)
d_output_proba = layers.Conv2D(filters=ndf*2, kernel_size=5,
                               strides=2, padding='same')(d_output_proba)
d_output_proba = layers.BatchNormalization(axis=-1)(d_output_proba)
d_output_proba = layers.LeakyReLU(alpha=0.2)(d_output_proba)

# plane shape: (8,8,ndf*4)
d_output_proba = layers.Conv2D(filters=ndf*4, kernel_size=5,
                               strides=2, padding='same')(d_output_proba)
d_output_proba = layers.BatchNormalization(axis=-1)(d_output_proba)
d_output_proba = layers.LeakyReLU(alpha=0.2)(d_output_proba)

# plane shape: (4,4,ndf*8)
d_output_proba = layers.Conv2D(filters=ndf*8, kernel_size=5,
                               strides=2, padding='same')(d_output_proba)
d_output_proba = layers.BatchNormalization(axis=-1)(d_output_proba)
d_output_proba = layers.LeakyReLU(alpha=0.2)(d_output_proba)

""" Canceled and replaced by what follows it.
# plane shape: (1,1,1)
d_output_proba = layers.Conv2D(filters=1, kernel_size=5,
                               strides=1, padding='valid',
                               activation='sigmoid')(d_output_proba)
d_output_proba = layers.Reshape((1,))(d_output_proba)
"""

# Flatten and Dense to (None, 1) \in [0, 1].
d_output_proba = layers.Flatten()(d_output_proba)
d_output_proba = layers.Dense(1, activation='sigmoid')(d_output_proba)


discriminator = Model(input_img, d_output_proba)
discriminator.summary()


# ### Compile discriminator

d_optimizer = keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
#d_optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.5)
#d_optimizer = keras.optimizers.RMSprop(lr=0.0008, clipvalue=1.0, decay=1e-8)
# [clipvalue](https://github.com/keras-team/keras/issues/3414)
discriminator.compile(optimizer=d_optimizer,
                      loss='binary_crossentropy')


# ### Adversarial network

discriminator.trainable = False
gan_input = keras.Input(shape=(latent_dim,))
gan_output = discriminator(generator(gan_input))
gan = Model(gan_input, gan_output)
gan_optimizer = keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
#gan_optimizer = keras.optimizers.Adam(lr=0.002, beta_1=0.5)
#gan_optimizer = keras.optimizers.RMSprop(lr=0.0004, clipvalue=1.0, decay=1e-8)
gan.compile(optimizer=gan_optimizer, loss='binary_crossentropy')
gan.summary()


# ### Data: celebA

# #### To normalize or not to normalize (the input images)
"""
celebA_path = "ref/dcgan/soumith-chintala--torch/celebA/img_align_celeba"
X_train = []

for fname in os.listdir(celebA_path):
    img = Image.open(os.path.join(celebA_path, fname))
    #img = np.array(img)                            # dtype('uint8')
    img = np.array(img).astype(np.float32) / 255   # dtype('float32')
    X_train.append(img)

X_train = np.array(X_train)
"""


#celebA_path = "ref/dcgan/soumith-chintala--torch/celebA/img_align_celeba"
celebA_path = "/home/wucf20/Documents/home/wucf20/Desktop/TruongHV/2019/age-progression/orange-lab/ref/dcgan/carpedm20--tf/data/celebA"

X_train_filepaths = glob.glob(os.path.join(celebA_path, '*.jpg'))
# so far X_train_filepaths is a list of jpg filepaths

#for fname in os.listdir(celebA_path):
#    img = Image.open(os.path.join(celebA_path, fname))
#    #img = np.array(img)                            # dtype('uint8')
#    img = np.array(img).astype(np.float32) / 255   # dtype('float32')
#    X_train.append(img)
#    n_pics += 1
#    if n_pics >= enough:
#        break

#X_train = np.array(X_train)


n_epochs = 3
batch_size = 64
save_dir = 'gan-celebA'
step = 0  # This is the number of mini-batch steps crossed so far
#m = X_train.shape[0]
m = len(X_train_filepaths)
shuffled_indices = np.arange(m)
n_steps_in_one_epoch = m // batch_size
print(bcolors.OKGREEN + bcolors.BOLD, end='')
print("We run {} epochs, each epoch containing {} steps.".format(n_epochs, n_steps_in_one_epoch))
print(bcolors.ENDC)

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

ckpt_freq = 200

def get_images(path_real_images):
    """
    This function copies the spirit of the one w/ the same name of carpedm20.
    """
    real_images = []
    for fpath in path_real_images:
        img_bgr = cv2.imread(fpath)
        img_rgb = img_bgr[..., ::-1]
        img_rgb = img_rgb.astype(np.float)
        h, w = img_rgb.shape[:2]
        input_h = 108
        j = int(round((h - input_h)/2.))
        i = int(round((w - input_h)/2.))
        cropped_img = scipy.misc.imresize(img_rgb[j:j+input_h, i:i+input_h], [64, 64])
        cropped_img = np.array(cropped_img)/127.5 - 1.
        real_images.append(cropped_img)
    real_images = np.array(real_images).astype(np.float32)
    return real_images


fixed_z = np.random.uniform(-1, 1, size=(batch_size, latent_dim))
"""
print(bcolors.OKGREEN + bcolors.BOLD, end='')
#print("sample_z = {}".format(sample_z))
for j in range(0, batch_size, 10):
    dist_to_0 = np.linalg.norm(sample_z[j])
    print("sample_z[{}] to 0: {:.4f}".format(j, dist_to_0))
print(bcolors.ENDC)
"""

for epoch in range(n_epochs):
    start = 0
    np.random.shuffle(X_train_filepaths)
    while True:
        #print("At step: {}".format(step))
        # First: Train discriminator only
        #z = np.random.normal(size=(batch_size, latent_dim))
        z = np.random.uniform(-1, 1, size=(batch_size, latent_dim))
        gen_images = generator.predict(z)
        stop = start + batch_size
        #batch_indices = shuffled_indices[start: stop]
        #real_images = X_train[batch_indices]
        path_real_images = X_train_filepaths[start: stop]
        #print("path_real_images.shape = {}".format(path_real_images.shape))
        real_images = get_images(path_real_images)
        combined_images = np.concatenate([gen_images, real_images])
        labels = np.concatenate([np.zeros((batch_size, 1)),
                                 np.ones((batch_size, 1))])
        # The next line is a trick suggested by fchollet: Add some noise.
        #labels += 0.05 * np.random.uniform(size=(labels.shape))
        # Note that these noise are all positive numbers.
        # So no worry as to the case, "some labels are negative."
        #labels = np.clip(labels, 0, 1)
        d_loss = discriminator.train_on_batch(combined_images, labels)


        # Second: Train gan/generator only
        #z = np.random.normal(size=(batch_size, latent_dim))
        white_lies = np.ones((batch_size, 1))
        a_loss = gan.train_on_batch(z, white_lies)
        # Run gan twice like carpedm20
        a_loss = gan.train_on_batch(z, white_lies)
        #a_loss = gan.train_on_batch(z, white_lies)

        start += batch_size
        if m - start < batch_size:
            #shuffled_indices = np.random.permutation(m)
            break
        step += 1
        if step % 200 == 0:
            #gan.save_weights(os.path.join(save_dir, "gan.h5"))
            print(bcolors.OKGREEN + bcolors.BOLD, end='')
            print("step: %s" % (step), end='')
            print(bcolors.ENDC)
            print("discriminator loss:", d_loss)
            print("adversarial loss:", a_loss)

            fixed_gen = generator.predict(fixed_z)
            fooled = discriminator.predict(fixed_gen) > 0.5
            # shape (so far): (batch_size, 1)
            print('fooled rate: %s/%s' % (fooled.sum(), batch_size))
            mistaken = discriminator.predict(real_images) <= 0.5
            print('mistaken rate: %s/%s' % (mistaken.sum(), batch_size))

            # Let's save some images so that we can view our results.
            #im = image.array_to_img(gen_images[0]*255, scale=False)
            #im = image.array_to_img(gen_images[0], scale=False)
            #im = image.array_to_img((gen_images[0] + 1)/2*255, scale=False)
            im = image.array_to_img(fixed_gen[0], scale=True)
            im.save(os.path.join(save_dir, 'gen-' + str(step).zfill(4) + '.png'))
            print("{} succeeds in fooling? {}".format('gen-' + str(step).zfill(4), bool(fooled[0])))
            #im = image.array_to_img(real_images[0]*255, scale=False)
            #im = image.array_to_img(real_images[0], scale=False)
            #im = image.array_to_img((real_images[0] + 1)/2*255, scale=False)
            im = image.array_to_img(real_images[0], scale=True)
            im.save(os.path.join(save_dir, 'real-' + str(step).zfill(4) + '.png'))
            print("{} mistaken? {}".format('real-' + str(step).zfill(4), bool(mistaken[0])))
            print()  # Create a line of separation.



