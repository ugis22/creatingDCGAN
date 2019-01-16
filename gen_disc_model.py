def generator(input_data, initial_dim, training, reuse=False):
    """
    The function takes a vector and generates an images through a deconvolution neural net.
    The parameters that need to be passed are: 
    * input_data: a vector 
    * initial_dim: the shape of the input data
    * Training: False or True, corresponding to whether the net is still on training or not
    * Reuse: Set to default to False, If True the function will reuse the variables already created
    """
    #Specifing the size of the different layers
    layer_size_1 = 512
    layer_size_2 = 256
    layer_size_3 = 128
    layer_size_4 = 64
    #layer_size_5 = 32
    
    #Final dimension output should be 3, because RGB images are wanted (one for each channel)
    final_layer = 3
    
    with tf.variable_scope("generator", reuse=reuse):
        #Project and reshape layer that will map the noise vector of shape 100x1 to a layer of 4*4*512
        #First a variable for the initial weights should be created
        #Shape should be 4*4*512
        #Random dimension is the dimension of the noise vector: 100
        #Standard deviation is selected according to literature
        weights_1 = tf.variable('weights_1', shape=[initial_dimension, 4*4*512], dtype=tf.float32, initizialiter=tf.truncated_normal_initializer(stddev=0.02))
        #Create a variable for the initial bias
        bias_1 = tf.variable('bias_1', shape=[layer_size_1*4*4], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
        #Add the first layer
        act_layer_1 = tf.add(tf.matmul(input_data, weights_1), bias_1, name='act_layer_1')
        

        #Each layer should contain three steps:
        #Deconvolutional layer2
        #convolution
        conv_layer_2 = conv2d_transpose(act_layer_1, layer_size_2, kernel_size=[4, 4], strides=[2, 2], padding='SAME', kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name='conv_layer_2')
        #bias
        bias_transf_2 = batch_norm(conv_layer_2, decay=0.9, epsilon=1e-5, is_training=training, scope='bias_transf_2')    
        #activation
        act_layer_2 = relu(bias_transf_2, name='act_layer_2')

        #Deconvolutional layer3
        #convolution
        conv_layer_3 = conv2d_transpose(act_layer_2, layer_size_3, kernel_size=[4, 4], strides=[2, 2], padding='SAME', kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name='conv_layer_3')
        #bias
        bias_transf_3 = batch_norm(conv_layer_3, decay=0.9, epsilon=1e-5, is_training=training, scope='bias_transf_3')    
        #activation
        act_layer_3 = relu(bias_transf_3, name='act_layer_3')

        #Deconvolutional layer4
        #convolution
        conv_layer_4 = conv2d_transpose(act_layer_3, layer_size_4, kernel_size=[4, 4], strides=[2, 2], padding='SAME', kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name='conv_layer_4')
        #bias
        bias_transf_4 = batch_norm(conv_layer_4, decay=0.9, epsilon=1e-5, is_training=training, scope='bias_transf_4')    
        #activation
        act_layer_4 = relu(bias_transf_4, name='act_layer_4')

        #Deconvolutional layer5
        #convolution
        #conv_layer_5 = tf.layers.conv2d_transpose(act_layer_4, layer_size_5, kernel_size=[4, 4], strides=[2, 2], padding='SAME', kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name='conv_layer_5')
        #bias
        #bias_transf_5 = tf.contrib.layers.batch_norm(conv_layer_5, decay=0.9, epsilon=1e-5, is_training=, scope='bias_transf_5')    
        #activation
        #act_layer_5 = tf.nn.relu(bias_transf_5, name='act_layer_5')

        #Final deconvolutional layer squash everything together to get a RGB image
        final_conv = conv2d_transpose(act_layer_4, final_layer, kernel_size=[4, 4], strides=[2, 2], padding='SAME', kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name='final_conv')
        #For the last layer always should be used tahn act function
        final_act = tahn(final_conv, name='final_act')
        return final_act


def discriminator(input_data, training, reuse=False):
    """
    The function takes images and outputs a vector of probabilities (btw 0 and 1) that the images are real using a convolution neural net.
    The parameters that need to be passed are: 
    * input_data: set of images 
    * Training: False or True, corresponding to whether the net is still on training or not
    * Reuse: Set to default to False, If True the function will reuse the variables already created
    """ 

    with tf.variable_scope("discriminator", reuse=reuse):
        #Size of the different layers   
        #As it takes the image generated by the generator, the input layer has the size of the output layer from generator
        input_layer_size = 3
        layer_size_1 = 64
        layer_size_2 = 128
        layer_size_3 = 256
        layer_size_4 = 512
        #The last layer should have size 1 as it is a vector
        output_size = 1

        #Convolutional Layer 1
        #Convolution
        conv_layer1 = conv2d(input_data, layer_size1, kernel_size=[4, 4], strides=[2, 2], padding='SAME', kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name='conv_layer1')
        #Batch Normalization
        bias_layer1 = batch_norm(conv_layer1, is_training=training, decay=0.9, epsilon=1e-5, scope='bias_layer1')
        #Activation function
        #In this case, leaky relu instead of relu will be used to avoid gradient fading
        act_layer1 = leaky_relu(conv_layer1, name="act_layer1")

        #Convolutional Layer 2
        conv_layer2 = conv2d(act_layer1, layer_size2, kernel_size=[4, 4], strides=[2, 2], padding='SAME', kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name='conv_layer2')
        bias_layer2 = batch_norm(conv_layer2, is_training=training, decay=0.9, epsilon=1e-5, scope='bias_layer2')
        act_layer2 = leaky_relu(conv_layer1, name="act_layer2")

        #Convolutional Layer 3
        conv_layer3 = conv2d(act_layer2, layer_size3, kernel_size=[4, 4], strides=[2, 2], padding='SAME', kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name='conv_layer3')
        bias_layer3 = batch_norm(conv_layer2, is_training=training, decay=0.9, epsilon=1e-5, scope='bias_layer3')
        act_layer3 = leaky_relu(conv_layer1, name="act_layer3")

        #Convolutional Layer 4
        conv_layer4 = conv2d(act_layer3, layer_size4, kernel_size=[4, 4], strides=[2, 2], padding='SAME', kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name='conv_layer4')
        bias_layer4 = batch_norm(conv_layer4, is_training=training, decay=0.9, epsilon=1e-5, scope='bias_layer4')
        act_layer4 = leaky_relu(conv_layer1, name="act_layer4")
        
        #Get the dimension of the final layer by getting the dynamic shape of the activation layer 4 and performing the product of the array.
        dimension = int(np.prod(act_layer4.get_shape()[1:]))
        #Now the steps are reverse from the generator layer. First, a reshape has to be performed.
        final_layer = tf.reshape(act_layer4, shape=[-1, dimension], name='final_layer')
        #Final weights are obtained. Shape is the shape of the final layer, and 1 because we are going to output a vector
        weights_2 = tf.get_variable('weights_1', shape=[final_layer.shape[-1], 1], dtype=tf.float32, initizialiter=tf.truncated_normal_initializer(stddev=0.02))
        #Create a variable for the initial bias. Shape is 1, vector is output
        bias_2 = tf.get_variable('bias_1', shape=[1], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
        #Add the last layer
        act_layer_outs = tf.add(tf.matmul(final_layer, weigths_2), bias_2, name='act_layer_outs')
        #Last layer for classification should be sigmoid 
        output_layer = sigmoid(act_layer_outs)

        return output_layer, act_layer_outs