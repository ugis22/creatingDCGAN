epochs = 1000
batch_size = 6
lr = 0.0002
random_dim = 100
buffer_size = 20

with tf.variable_scope("input"):
    # placeholders for input
    real_image = tf.placeholder(tf.float32, shape = [None, 64, 64, 3])
    random_input = tf.placeholder(tf.float32, shape=[None, random_dim])

    # Generate fake image
    fake_image = generator(random_input, reuse=False)

    # Now apply discriminator to both fake and real images
    real_logits = discriminator(real_image, reuse=False)
    fake_logits = discriminator(fake_image)

    #Calculate the loss for discriminator and generator. Take Binary Cross Entropy with logits (Logits should be a tensor of float32 or float64)
    #Discriminator Loss with real images, because the intended target is 1(because it's the discriminator and the aim is to clasify real images
    #as 1, a vector of one is pass to the cross entropy function.
    discriminator_loss_real = tf.losses.sigmoid_cross_entropy(tf.ones_like(real_logits), real_logits)
    #Now the discriminator loss with fake images should be calculated and the target should be 0.
    #discriminator_loss_real = tf.reduce_mean(fake_logits) - tf.reduce_mean(real_logits)

    discriminator_loss_fake = tf.losses.sigmoid_cross_entropy(tf.zeros_like(fake_logits), fake_logits)
    #Now the total loss of the discriminator is calculated as the sum of both losses
    discriminator_loss_total = (discriminator_loss_fake + discriminator_loss_real) / 2.0

    #Generator loss only with fake images
    generator_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(fake_logits), fake_logits)

    #Create the optimizer for both the discriminator and the generator
    training_vars = tf.trainable_variables()
    discrim_vars = [var for var in training_vars if "discriminator" in var.name]
    generator_vars = [var for var in training_vars if "generator" in var.name]

    discriminator_optimizer = AdamOptimizer(learning_rate=lr, beta1=0.3).minimize(discriminator_loss_total, var_list=discrim_vars)
    generator_optimizer = AdamOptimizer(learning_rate=lr, beta1=0.3).minimize(generator_loss, var_list=generator_vars)

    fake_sample = generator(random_input, training=False)
#Run TF session and initialize variables
sess = tf.Session()


sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

#Save and restore variables and Checkpoints with Saver
saver = Saver(max_to_keep=5)

path_saved = saver.save(sess, "/tmp/model.checkpnt")
checkpnt = latest_checkpoint(".model/newimages")

saver.restore(sess, path_saved)
#coordinator_threads = Coordinator()
#threads = tf.train.QueueRunners(sess=sess, coord=coordinator_threads)
#threads = [threading.Thread(target=MyLoop, args=(oordinator_threads,)) for i in xrange(10)]

#Define some parameters
image_batch, sample_number = reading_images(buffer_size, batch_size)

#image_batch = reading_images(buffer_size, batch_size)
batch_number = int(sample_number/batch_size)
total_batch = 0


#start training
#Run the training loop for determined epochs

for epoch in range(epochs):
    #Show progress
    print("Running epoch {}/{}".format(epoch, epochs))
    #Run the samples in minibatches (number of batch was already predefined)
    for minibatch in range(batch_number):
        noise_training = np.random.uniform(-1.0, 1.0, size=[batch_size, random_dim]).astype(np.float32)

        #We established the number of iterations for the discriminator and generator. Because if the discriminator is not good
        #enough, the generator will get away with no so good quality images, we will iterate more over the discriminator than the
        #generator
        iterations_discriminator = 4
        iterations_generator = 1

        #We iterate n times with the discriminator
        for n_iter in range(iterations_discriminator):
            image_training = sess.run(image_batch)
            #GAN clip weights
            #sess.run(discrim_clip)

            #Now the discriminator has to be updated. Inside the tf session, the graph is initialized, and the trainer and
            #loss function has to be feeded. In this case, trainer will be feeded with noise, real image and training
            _, discriminator_loss_total = sess.run([discriminator_optimizer, discriminator_loss_total],
                                                  feed_dict={random_input: noise_training,
                                                            real_image: image_training,
                                                            training: True})
            print("Generator loss is {}".format(discriminator_loss_total))

        #Iterate m times with generator
        for m_iter in range(iterations_generator):

            #Now the generator has to be updated. In this case, the trainer and loss function will be feeded with noise and
            #training only because generator will not see the real images
            _, generator_loss_total = sess.run([generator_optimizer, generator_loss],
                                                  feed_dict={random_input: noise_training,training: True})

            print("Generator loss is {}".format(generator_loss_total))

    #Now the checkpoints and the images has to be saved:
    #Save every 500 the checkpoints
    if epoch%500 == 0:
        #If the directory where saving checkpoints does not exist, create it
        if not os.path.exists('./model'):
            os.makedirs('./model')
        #Save checkpoints
        saver.save(sess, './model' + '/' + str(epoch))

    if epoch%50 == 0:
        #If the directory where saving images does not exist, create it
        if not os.path.exists('./newimages'):
            os.makedirs('./newimages')

        #Create a random vector to feed the generator
        noise_sample = np.random.normal(size=[100, random_dim])
        #Pass those parameters to generate an image but this time in not training mode
        test_image = sess.run(fake_sample, feed_dict={random_input: noise_sample})
        #Save the image created in the chosen directory
        save_images(test_image, [16,16], './newimages'+ '/epoch' + str(epoch) + '.jpg')

#coordinator_threads.request_stop()
#coordinator_threads.join(threads)
sess.close()                   
