from keras.models import Sequential, Model
from keras import layers
import keras
import numpy as np
import os
from PIL import Image
from keras.preprocessing import image



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
height = 64
width = 64
channels = 3

generator_input = keras.Input(shape=(latent_dim,))

# First, transform the input into a 16x16 128-channels feature map
x = layers.Dense(128 * 16 * 16)(generator_input)
x = layers.LeakyReLU()(x)
x = layers.Reshape((16, 16, 128))(x)

# Then, add a convolution layer
x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)

# Upsample to 32x32
x = layers.Conv2DTranspose(256, 4, strides=2, padding='same')(x)
x = layers.LeakyReLU()(x)

# Upsample to 64x64
x = layers.Conv2DTranspose(256, 4, strides=2, padding='same')(x)
x = layers.LeakyReLU()(x)


# Few more conv layers
x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)

# Produce a 32x32 1-channel feature map
x = layers.Conv2D(channels, 7, activation='tanh', padding='same')(x)
generator = keras.models.Model(generator_input, x)
generator.summary()



# Discriminator
"""
carpedm20's discriminator is quasi-identical to soumith's
except at the 1st layer: carpedm20's was easier, just
the final layer being diff.
"""

discriminator_input = layers.Input(shape=(height, width, channels))
x = layers.Conv2D(128, 3)(discriminator_input)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Flatten()(x)

# One dropout layer - important trick!
x = layers.Dropout(0.4)(x)

# Classification layer
x = layers.Dense(1, activation='sigmoid')(x)

discriminator = keras.models.Model(discriminator_input, x)
discriminator.summary()



# ### Compile discriminator

d_optimizer = keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
#d_optimizer = keras.optimizers.RMSprop(lr=0.0008, clipvalue=1.0, decay=1e-8)
#d_optimizer = keras.optimizers.RMSprop(lr=0.0004, clipvalue=1.0, decay=1e-8)
discriminator.compile(optimizer=d_optimizer,
                      loss='binary_crossentropy')


# ### Adversarial network

discriminator.trainable = False
gan_input = keras.Input(shape=(latent_dim,))
gan_output = discriminator(generator(gan_input))
gan = Model(gan_input, gan_output)
gan_optimizer = keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
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


celebA_path = "ref/dcgan/soumith-chintala--torch/celebA/img_align_celeba"
X_train = []
n_pics = 0
enough = 2e3

for fname in os.listdir(celebA_path):
    img = Image.open(os.path.join(celebA_path, fname))
    #img = np.array(img)                            # dtype('uint8')
    img = np.array(img).astype(np.float32) / 255   # dtype('float32')
    X_train.append(img)
    n_pics += 1
    if n_pics >= enough:
        break

X_train = np.array(X_train)


n_epochs = 100
batch_size = 64
save_dir = 'gan-celebA'
step = 0  # This is the number of mini-batch steps crossed so far
m = X_train.shape[0]
shuffled_indices = np.arange(m)
n_steps_in_one_epoch = m // batch_size
print(bcolors.OKGREEN + bcolors.BOLD, end='')
print("We run {} epochs, each epoch containing {} steps.".format(n_epochs, n_steps_in_one_epoch))
print(bcolors.ENDC)

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

for epoch in range(n_epochs):
    start = 0
    while True:
        #print("At step: {}".format(step))
        # First: Train discriminator only
        z = np.random.normal(size=(batch_size, latent_dim))
        #z = np.random.uniform(-1, 1, size=(batch_size, latent_dim))
        gen_images = generator.predict(z)
        stop = start + batch_size
        #batch_indices = shuffled_indices[start: stop]
        #real_images = X_train[batch_indices]
        real_images = X_train[start: stop]
        combined_images = np.concatenate([gen_images, real_images])
        #labels = np.concatenate([np.zeros((batch_size, 1)),
        #                         np.ones((batch_size, 1))])
        labels = np.concatenate([np.ones((batch_size, 1)),
                                 np.zeros((batch_size, 1))])
        # The next line is a trick suggested by fchollet: Add some noise.
        labels += 0.05 * np.random.uniform(size=(labels.shape))
        # Note that these noise are all positive numbers.
        # So no worry as to the case, "some labels are negative."
        #labels = np.clip(labels, 0, 1)
        d_loss = discriminator.train_on_batch(combined_images, labels)


        # Second: Train gan/generator only
        #z = np.random.normal(size=(batch_size, latent_dim))
        #white_lies = np.ones((batch_size, 1))
        white_lies = np.zeros((batch_size, 1))
        a_loss = gan.train_on_batch(z, white_lies)
        # Run gan twice like carpedm20
        #a_loss = gan.train_on_batch(z, white_lies)

        start += batch_size
        if m - start < batch_size:
            #shuffled_indices = np.random.permutation(m)
            break
        step += 1
        if step % 50 == 0:
            gan.save_weights(os.path.join(save_dir, "gan.h5"))
            print(bcolors.OKGREEN + bcolors.BOLD, end='')
            print("step: %s" % (step), end='')
            print(bcolors.ENDC)
            print("discriminator loss:", d_loss)
            print("adversarial loss:", a_loss)

            fooled = discriminator.predict(gen_images) < 0.5
            # shape (so far): (batch_size, 1)
            print('fooled rate: %s/%s' % (fooled.sum(), batch_size))
            mistaken = discriminator.predict(real_images) >= 0.5
            print('mistaken rate: %s/%s' % (mistaken.sum(), batch_size))

            # Let's save some images so that we can view our results.
            im = image.array_to_img(gen_images[0]*255, scale=False)
            #im = image.array_to_img((gen_images[0] + 1)/2*255, scale=False)
            #im = image.array_to_img(gen_images[0], scale=False)
            im.save(os.path.join(save_dir, 'gen-' + str(step).zfill(4) + '.png'))
            print("{} succeeds in fooling? {}".format('gen-' + str(step).zfill(4), bool(fooled[0])))
            #im = image.array_to_img(real_images[0]*255, scale=False)
            #im = image.array_to_img(real_images[0], scale=False)
            #im = image.array_to_img(gen_images[0]*255, scale=False)
            #im.save(os.path.join(save_dir, 'by255-' + str(step).zfill(4) + '.png'))
            im = image.array_to_img(gen_images[0], scale=True)
            im.save(os.path.join(save_dir, 'scale-True-' + str(step).zfill(4) + '.png'))
            #im = image.array_to_img(gen_images[0], scale=False)
            #im.save(os.path.join(save_dir, 'unchanged-' + str(step).zfill(4) + '.png'))
            print("{} mistaken? {}".format('real-' + str(step).zfill(4), bool(mistaken[0])))
            print()  # Create a line of separation.



