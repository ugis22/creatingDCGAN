def train_network(batch_size, epoch):
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
    save = Saver()
    

    
    
    
    
    
    
    
    
    #start training
    for i in range(epoch):
        