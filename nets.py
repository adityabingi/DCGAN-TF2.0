import tensorflow as tf
from config import Config

b_init = tf.constant_initializer(0.0) #bias initializer
w_init = tf.initializers.TruncatedNormal(stddev=0.02) #kernel_initializer

def ReLU():	

	return tf.keras.layers.ReLU()

def leaky_ReLU(leak=0.2):

	return tf.keras.layers.LeakyReLU(alpha =leak)

def batch_norm(axis=-1):

	return tf.keras.layers.BatchNormalization(momentum=Config.momentum,\
											epsilon=Config.epsilon,
											axis=axis)
def flatten():

	return tf.keras.layers.Flatten()

def linear(units, use_bias=False):

	return tf.keras.layers.Dense(units = units,\
								kernel_initializer = w_init,
								use_bias=use_bias,
								bias_initializer=b_init)

def conv_layer(filters, kernel_size, strides, use_bias=False):

	return tf.keras.layers.Conv2D(filters = filters,\
								kernel_size = kernel_size,
								strides = strides,
								padding='same',
								data_format='channels_last',
								kernel_initializer = w_init,
								use_bias=use_bias,
								bias_initializer=b_init )

def transpose_conv_layer(filters, kernel_size, strides,use_bias=False):
	
	return tf.keras.layers.Conv2DTranspose(filters= filters,\
										kernel_size =kernel_size,
										strides = strides,
										padding='same',
										data_format='channels_last',
										kernel_initializer = w_init,
										use_bias=use_bias,
										bias_initializer=b_init)

class ConvBlock(tf.keras.Model):

	def __init__(self, filters):
		super(ConvBlock, self).__init__(name='conv_block')
		self.kernel_size = (5,5)
		self.strides = (2,2)
		self.filters = filters
		self.conv = conv_layer(self.filters, self.kernel_size, self.strides)
		self.batch_norm = batch_norm()
		self.leaky_ReLU = leaky_ReLU()

	def call(self, inputs, training=False):

		inputs = self.conv(inputs)
		inputs = self.batch_norm(inputs, training=training)
		output = self.leaky_ReLU(inputs)
		return output

class TransposeConvBlock(tf.keras.Model):

	def __init__(self, filters):
		super(TransposeConvBlock, self).__init__(name='transpose_conv_block')
		self.kernel_size = (5,5)
		self.strides = (2,2)
		self.filters = filters
		self.transpose_conv = transpose_conv_layer(self.filters, self.kernel_size, self.strides)
		self.batch_norm = batch_norm()
		self.ReLU = ReLU()

	def call(self, inputs, training=False):

		inputs = self.transpose_conv(inputs)
		inputs = self.batch_norm(inputs, training=training)
		output = self.ReLU(inputs)
		return output

class Discriminator(tf.keras.Model):
	# class for discriminator
	def __init__(self):
		super(Discriminator, self).__init__(name='Discriminator')
		self.disc_filters = Config.disc_filters
		self.conv = conv_layer(filters=self.disc_filters[0], kernel_size=(5,5), strides=(2,2), use_bias=True)
		self.leaky_ReLU = leaky_ReLU()
		self.conv_blocks = [ConvBlock(self.disc_filters[i]) for i in range(1, len(self.disc_filters), 1)]
		self.flatten = flatten()
		self.linear = linear(units=1, use_bias=True)

	def call(self, inputs, training=False):
		inputs = self.conv(inputs)
		inputs = self.leaky_ReLU(inputs)
		for conv_block in self.conv_blocks:
			inputs = conv_block(inputs, training=training)
		inputs = self.flatten(inputs)
		output = self.linear(inputs)
		return output

class Generator(tf.keras.Model):
	# class for generator
	def __init__(self):
		super(Generator, self).__init__(name='Generator')
		self.gen_filters = Config.gen_filters
		self.img_shape = Config.img_shape
		self.linear = linear(units=4*4*self.gen_filters[0])
		self.batch_norm = batch_norm()
		self.ReLU = ReLU()
		self.transpose_conv_blocks = [TransposeConvBlock(self.gen_filters[i]) for i in range(1, len(self.gen_filters), 1)]
		self.transpose_conv = transpose_conv_layer(filters=self.img_shape[-1], kernel_size=(5,5), strides=(2,2), use_bias=True)


	def call(self, inputs, training=False):
		inputs = self.linear(inputs)
		inputs = tf.reshape(inputs, [-1, 4, 4, self.gen_filters[0]])
		inputs = self.batch_norm(inputs, training=training)
		inputs = self.ReLU(inputs)
		for transpose_conv_block in self.transpose_conv_blocks:
			inputs = transpose_conv_block(inputs, training=training)
		inputs = self.transpose_conv(inputs)
		output = tf.nn.tanh(inputs)
		return output
