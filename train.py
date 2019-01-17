def train_network(batch_size, epochs):
    """
    This function train the two neural network at the same time. Because the real data is used to train the discrimator to distingish
    real from fake data, indirectly is also training the generator to create better and better images.
    """
    # Placeholders are created. They are gateways to input the data into the computational graph.
    with tf.variable_scope('input'):
        real_image = tf.placeholder(tf.float32, shape = [None, 128, 128, 3], name='real_image')
        random_data = tf.placeholder(tf.float32, shape=[None, initial_dimension], name='random_data')
        training = tf.placeholder(tf.bool, name='training')
      
    #We set to 100 the initial dimension because the noise is a 100*1 vector 
    initial_dimension = 100
    
    #We generate the image with the generator
    fake_image = generator(random_data, initial_dimension, training)
    
    #Now we obtain the result for the fake and real image
    real_logits, real_results = discriminator(real_image, training)
    fake_logits, fake_results = discriminator(fake_image, training, reuse=True)
    
    #Calculate the loss for discriminator and generator. Take Binary Cross Entropy with logits (Logits should be a tensor of float32 or float64)

    #Discriminator Loss with real images, because the intended target is 1(because it's the discriminator and the aim is to clasify real images 
    #as 1, a vector of one is pass to the cross entropy function.
    discriminator_loss_real = tf.reduce_mean(
        sigmoid_cross_entropy_with_logits(logits=real_logits, lables=tf.ones_like(real_logits)))
    #Now the discriminator loss with fake images should be calculated and the target should be 0.
    discriminator_loss_fake = tf.reduce_mean(
        sigmoid_cross_entropy_with_logits(logits=fake_logits, lables=tf.zeros_like(fake_logits)))
    #Now the total loss of the discriminator is calculated as the sum of both losses
    discriminator_loss_total = discriminator_loss_real + discriminator_loss_fake
    
    #Generator loss only with fake images
    generator_loss = tf.reduce_mean(
        sigmoid_cross_entropy_with_logits(logits=fake_logits, labels=tf.ones_like(fake_logits)))
      
    #Create the optimizer for both the discriminator and the generator
    discriminator_optimizer = AdamOptimizer(learning_rate=0.0002, beta1=0.5, beta2=0.999).minimize(discriminator_loss_total)
    generator_optimizer = AdamOptimizer(learning_rate=0.0002, beta1=0.5, beta2=0.999).minimize(generator_loss)
    
    #Run TF session and initialize variables
    sess = tf.Session()
    sess.run(tf.global_variables_initializer)
    sess.run(tf.local_variables_initializer)

    #Save and restore variables and Checkpoints with Saver
    saver = Saver()
    path_saved = saver.save(sess, "/tmp/model.checkpnt")
    checkpnt = latest_checkpoint(".model/newimages")
    saver.restore(sess, path_saved)
    coordinator_threads = Coordinator()
    #threads = ???
    
    #Define some parameters
    image_batch, sample_number = processing_data()
    batch_number = int(sample_number/batch_size)
    total_batch = 0

    #start training
    #Run the training loop for determined epochs
    for epoch in range(epochs):
        #Run the samples in minibatches (number of batch was already predefined)
        for minibatch in range(batch_number):
            noise_training = np.random.uniform(-1.0, 1.0, size=[batch_size, initial_dimension]).astype(np.float32)
            
            #We established the number of iterations for the discriminator and generator. Because if the discriminator is not good
            #enough, the generator will get away with no so good quality images, we will iterate more over the discriminator than the
            #generator
            iterations_discriminator = 4
            iterations_generator = 1
            
            #We iterate n times with the discriminator
            for n_iter in range(iterations_discriminator):
                image_training = sess.run(image_batch)
                
                #Now the discriminator has to be updated. Inside the tf session, the graph is initialized, and the trainer and
                #loss function has to be feeded. In this case, trainer will be feeded with noise, real image and training
                _, discriminator_loss_total = sess.run([discriminator_optimizer, discriminator_loss_total], 
                                                      feed_dict={random_data: noise_training,
                                                                real_image: image_training,
                                                                training: True})
            #Iterate m times with generator
            for m_iter in range(iterations_generator):
                image_training = sess.run(image_batch)
                
                #Now the generator has to be updated. In this case, the trainer and loss function will be feeded with noise and 
                #training only because generator will not see the real images
                _, generator_loss_total = sess.run([generator_optimizer, generator_loss], 
                                                      feed_dict={random_data: noise_training,training: True})
            
        #Now the checkpoints and the images has to be saved:
        #Save every 500 the checkpoints
        if epoch%500 == 0:
            #If the directory where saving checkpoints does not exist, create it
            if not os.path.exists('./model/newimages'):
                os.makedirs('./model/newimages')
            #Save checkpoints
            saver.save(sess, './model/newimages' + '/' + str(epoch))
        
        if epoch%50 == 0:
            #If the directory where saving images does not exist, create it
            if not os.path.exists('./newimages'):
                os.makedirs('./newimages')
                    
            #Create a random vector to feed the generator
            noise_sample = np.random.uniform(-1.0, 1.0, size=[batch_size, initial_dimension]).astype(np.float32)
            #Pass those parameters to generate an image but this time in not training mode
            test_image = sess.run(fake_image, feed_dict={random_data: noise_sample, training: False})
            #Save the image created in the chosen directory
            save_images(test_image, [16,16], './newimages'+ '/epoch' + str(epoch) + '.jpg')

    coordinator.request_stop()
    coordinator.join(threads)                     