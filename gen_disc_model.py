
def generator(input, training=True, reuse=True):
    """
    The function takes a vector and generates an images through a deconvolution neural net.
    The parameters that need to be passed are:
    * input: a vector
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
    output_layer = 3

    with tf.variable_scope("generator") as scope:
        if reuse:
            scope.reuse_variables()
        #Project and reshape layer that will map the noise vector of shape 100x1 to a layer of 4*4*512
        #First a variable for the initial weights should be created
        #Shape should be 4*4*512
        #Random dimension is the dimension of the noise vector: 100
        #Standard deviation is selected according to literature
        weights_1 = tf.get_variable('weights_1', shape=[random_dim, 4 * 4 * layer_size_1], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
        #Create a variable for the initial bias
        bias_1 = tf.get_variable('bias_1', shape=[layer_size_1 * 4 * 4], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
        #Add the first layer
        initial_layer = tf.add(tf.matmul(input, weights_1), bias_1, name='initial_layer')
        #Reshape the layer from 100*1 to 4*4*512
        conv_layer_1 = tf.reshape(initial_layer, shape=[-1, 4, 4, layer_size_1], name='conv_layer_1')
        #Normalization
        bias_transf_1 = batch_norm(conv_layer_1, is_training=training, decay = 0.9,  epsilon=1e-5, updates_collections=None, scope='bias_transf_1')
        #Activation
        act_layer_1 = relu(bias_transf_1, name='act_layer_1')

        #Now each layer should contain three steps: convolution, bias, activation.
        #Deconvolutional layer2
        #convolution
        conv_layer_2 = conv2d_transpose(act_layer_1, layer_size_2, kernel_size=[5, 5], strides=[2, 2], padding='SAME', kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name='conv_layer_2')
        #bias
        bias_transf_2 = batch_norm(conv_layer_2, decay=0.9, epsilon=1e-5, is_training=training, updates_collections=None, scope='bias_transf_2')
        #activation
        act_layer_2 = relu(bias_transf_2, name='act_layer_2')

        #Deconvolutional layer3
        #convolution
        conv_layer_3 = conv2d_transpose(act_layer_2, layer_size_3, kernel_size=[5, 5], strides=[2, 2], padding='SAME', kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name='conv_layer_3')
        #bias
        bias_transf_3 = batch_norm(conv_layer_3, decay=0.9, epsilon=1e-5, is_training=training, updates_collections=None, scope='bias_transf_3')
        #activation
        act_layer_3 = relu(bias_transf_3, name='act_layer_3')

        #Deconvolutional layer4
        #convolution
        conv_layer_4 = conv2d_transpose(act_layer_3, layer_size_4, kernel_size=[5, 5], strides=[2, 2], padding='SAME', kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name='conv_layer_4')
        #bias
        bias_transf_4 = batch_norm(conv_layer_4, decay=0.9, epsilon=1e-5, is_training=training, updates_collections=None, scope='bias_transf_4')
        #activation
        act_layer_4 = relu(bias_transf_4, name='act_layer_4')

        #Deconvolutional layer5
        #convolution
        #conv_layer_5 = conv2d_transpose(act_layer_4, layer_size_5, kernel_size=[5, 5], strides=[2, 2], padding='SAME', kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name='conv_layer_5')
        #bias
        #bias_transf_5 = batch_norm(conv_layer_5, decay=0.9, epsilon=1e-5, is_training=training, updates_collections=None, scope='bias_transf_5')
        #activation
        #act_layer_5 = relu(bias_transf_5, name='act_layer_5')

        #Final deconvolutional layer squash everything together to get a RGB image
        final_conv = conv2d_transpose(act_layer_4, output_layer, kernel_size=[5, 5], strides=[2, 2], padding='SAME', kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name='final_conv')

        #bias
        bias_final = batch_norm(final_conv, decay=0.9, epsilon=1e-5, is_training=training, updates_collections=None, scope='bias_final')


        #For the last layer always should be used tahn act function
        final_act = tanh(bias_final, name='final_act')
        return final_act


def discriminator(input, training=True, reuse=True):
    """
    The function takes images and outputs a vector of probabilities (btw 0 and 1) that the images are real using a convolution neural net.
    The parameters that need to be passed are:
    * input: set of images
    * Training: False or True, corresponding to whether the net is still on training or not
    * Reuse: Set to default to False, If True the function will reuse the variables already created
    """
    #Size of the different layers
    #As it takes the image generated by the generator, the input layer has the size of the output layer from generator
    layer_size1 = 64
    layer_size2 = 128
    layer_size3 = 256
    layer_size4 = 512

    with tf.variable_scope("discriminator") as scope:
        if reuse:
            scope.reuse_variables()

            #Convolution, activation, bias, repeat!

        #Convolutional Layer 1
        #Convolution
        conv_layer1 = conv2d(input, layer_size1, kernel_size=[5, 5], strides=[2, 2], padding='SAME', kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name='conv_layer1')
        #Batch Normalization
        bias_layer1 = batch_norm(conv_layer1, is_training=training, decay=0.9, epsilon=1e-5, scope='bias_layer1')
        #Activation function
        #In this case, leaky relu instead of relu will be used to avoid gradient fading
        act_layer1 = lrelu(conv_layer1, n='act_layer1')

        #Convolutional Layer 2
        conv_layer2 = conv2d(act_layer1, layer_size2, kernel_size=[5, 5], strides=[2, 2], padding='SAME', kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name='conv_layer2')
        bias_layer2 = batch_norm(conv_layer2, is_training=training, decay=0.9, epsilon=1e-5, scope='bias_layer2')
        act_layer2 = lrelu(conv_layer1, n='act_layer2')

        #Convolutional Layer 3
        conv_layer3 = conv2d(act_layer2, layer_size3, kernel_size=[5, 5], strides=[2, 2], padding='SAME', kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name='conv_layer3')
        bias_layer3 = batch_norm(conv_layer2, is_training=training, decay=0.9, epsilon=1e-5, scope='bias_layer3')
        act_layer3 = lrelu(conv_layer1, n='act_layer3')

        #Convolutional Layer 4
        conv_layer4 = conv2d(act_layer3, layer_size4, kernel_size=[5, 5], strides=[2, 2], padding='SAME', kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name='conv_layer4')
        bias_layer4 = batch_norm(conv_layer4, is_training=training, decay=0.9, epsilon=1e-5, scope='bias_layer4')
        act_layer4 = lrelu(bias_layer4, n='act_layer4')

        #Get the dimension of the final layer by getting the dynamic shape of the activation layer 4 and performing the product of the array.
        #dimension = int(np.prod(act_layer4.get_shape()[1:]))
        dimension = int(np.prod(act_layer4.get_shape()[1:]))

        #Now the steps are reverse from the generator layer. First, a reshape has to be performed.
        final_layer = tf.reshape(act_layer4, shape=[-1, dimension], name='final_layer')
        #Final weights are obtained. Shape is the shape of the final layer, and 1 because we are going to output a vector
        weights_2 = tf.get_variable('weights_2', shape=[final_layer.shape[-1], 1], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
        #Create a variable for the initial bias. Shape is 1, vector is output
        bias_2 = tf.get_variable('bias_1', shape=[1], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
        #Add the last layer
        act_layer_outs = tf.add(tf.matmul(final_layer, weights_2), bias_2, name='act_layer_outs')
        #Last layer for classification should be sigmoid
        output_layer = sigmoid(act_layer_outs)

        #return act_layer_outs, output_layer
        return act_layer_outs
