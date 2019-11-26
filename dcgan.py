import sys
import numpy
import argparse
import tensorflow as tf

import nets
from utils import *
from config import Config
from dataset import prepare_dataset

class DCGAN:

	def __init__(self, strategy, restore):

		self.strategy = strategy
		self.z_dim = Config.latent_dim
		self.global_batchsize = Config.global_batchsize
		self.batchsize_per_replica = int(self.global_batchsize/self.strategy.num_replicas_in_sync)

		self.gen_model = nets.Generator()
		self.disc_model = nets.Discriminator()
		self.gen_optimizer = tf.keras.optimizers.Adam(learning_rate=Config.gen_lr, beta_1=Config.beta1, beta_2=Config.beta2)
		self.disc_optimizer = tf.keras.optimizers.Adam(learning_rate=Config.disc_lr, beta_1=Config.beta1, beta_2=Config.beta2)
		self.train_writer = tf.summary.create_file_writer(Config.summaryDir+'train')

		self.ckpt = tf.train.Checkpoint(step=tf.Variable(0),
										generator_optimizer=self.gen_optimizer,
										generator_model = self.gen_model,
										discriminator_optimizer=self.disc_optimizer,
										discriminator_model=self.disc_model)

		self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, Config.modelDir, max_to_keep=3)

		self.global_step = 0

		if(restore):
			latest_ckpt= tf.train.latest_checkpoint(Config.modelDir)
			if not latest_ckpt:
				raise Exception('No saved model found in: ' + Config.modelDir)
			self.ckpt.restore(latest_ckpt)
			self.global_step = int(latest_ckpt.split('-')[-1])   # .../ckpt-300 returns 300 previously trained totalbatches
			print("Restored saved model from latest checkpoint")

	def compute_loss(self, labels, predictions):

		cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True,\
											reduction=tf.keras.losses.Reduction.NONE)
		return cross_entropy(labels, predictions)

	def disc_loss(self, real_output, fake_output):

		real_loss = self.compute_loss(tf.ones_like(real_output), real_output)
		fake_loss = self.compute_loss(tf.zeros_like(fake_output), fake_output)
		total_loss = real_loss + fake_loss
		total_loss = total_loss/self.global_batchsize
		return total_loss

	def gen_loss(self, fake_output):

		gen_loss = self.compute_loss(tf.ones_like(fake_output), fake_output)
		gen_loss = gen_loss / self.global_batchsize
		return gen_loss

	#@tf.function
	def train_step(self, real_imgs):

		noise = tf.random.normal(shape=[tf.shape(real_imgs)[0], self.z_dim])

		with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
			generated_imgs = self.gen_model(noise, training=True)
			real_output = self.disc_model(real_imgs, training=True)
			fake_output = self.disc_model(generated_imgs, training=True)
			d_loss = self.disc_loss(real_output, fake_output)
			g_loss = self.gen_loss(fake_output)

		G_grads = gen_tape.gradient(g_loss, self.gen_model.trainable_variables)
		D_grads = disc_tape.gradient(d_loss, self.disc_model.trainable_variables)

		self.gen_optimizer.apply_gradients(zip(G_grads, self.gen_model.trainable_variables))
		self.disc_optimizer.apply_gradients(zip(D_grads, self.disc_model.trainable_variables))

		#run g_optim twice to make sure d_loss doesn't go to zero
		with tf.GradientTape() as gen_tape:
			generated_imgs = self.gen_model(noise, training=True)
			fake_output = self.disc_model(generated_imgs, training=True)
			g_loss = self.gen_loss(fake_output)

		G_grads = gen_tape.gradient(g_loss, self.gen_model.trainable_variables)
		self.gen_optimizer.apply_gradients(zip(G_grads, self.gen_model.trainable_variables))

		return g_loss, d_loss

	#@tf.function
	def gen_step(self, random_latents):
		gen_imgs = self.gen_model(random_latents, training=False)
		return gen_imgs

	@tf.function
	def distribute_trainstep(self, dist_dataset):
		per_replica_g_losses, per_replica_d_losses = self.strategy.experimental_run_v2(self.train_step,\
																					args=(dist_dataset,))
		total_g_loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_g_losses,axis=0)
		total_d_loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_d_losses, axis=0)

		return total_g_loss, total_d_loss

	@tf.function
	def distribute_genstep(self, dist_gen_noise):

		per_replica_genimgs = self.strategy.experimental_run_v2(self.gen_step, args=(dist_gen_noise,))
		gen_imgs = self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_genimgs,axis=None)
		return gen_imgs

	#@tf.function
	def train_loop(self, num_epochs, dist_dataset, dist_noise_dataset):

		num_batches = self.global_step
		for i in range(num_epochs):
			print('At Epoch {}'.format(i+1))
			print('.........................................')
			for one_batch in dist_dataset:
				total_g_loss, total_d_loss = self.distribute_trainstep(one_batch)

				with self.train_writer.as_default():
					tf.summary.scalar('generator_loss',total_g_loss, step=num_batches)
					tf.summary.scalar('discriminator_loss',total_d_loss, step=num_batches)

				if(num_batches % Config.image_snapshot_freq == 0):
					for dist_gen_noise in dist_noise_dataset:
						gen_imgs = self.distribute_genstep(dist_gen_noise)

					filename = Config.results_dir + 'fakes_epoch{:02d}_batch{:05d}.jpg'.format(i+1, num_batches)
					save_image_grid(gen_imgs.numpy(), filename, drange=[-1,1], grid_size=Config.grid_size)

				num_batches+=1
				print('Gen_loss at batch {}: {:0.3f}'.format(num_batches, total_g_loss))
				print('Disc_loss at batch {}: {:0.3f}'.format(num_batches, total_d_loss))

			self.ckpt.step.assign(i+1)
			self.ckpt_manager.save()

def train(strategy, restore):

	train_dataset = prepare_dataset(Config.tfrecord_dir+'train.tfrecord')
	dist_dataset = strategy.experimental_distribute_dataset(train_dataset)

	#periodic generation of fake images
	gen_noise = tf.random.normal(shape=[Config.num_gen_imgs, Config.latent_dim], seed=Config.random_seed)
	gen_dataset = tf.data.Dataset.from_tensor_slices(gen_noise).repeat(Config.num_gpu).batch(Config.num_gen_imgs*Config.num_gpu)
	dist_noise_dataset = strategy.experimental_distribute_dataset(gen_dataset)

	with strategy.scope():

		dcgan = DCGAN(strategy, restore)
		#num_epochs = tf.constant(Config.num_epochs, dtype=tf.int32)
		dcgan.train_loop(Config.num_epochs, dist_dataset, dist_noise_dataset)

	make_training_gif()

def generate(strategy, restore):

	gen_noise = tf.random.normal(shape=[Config.num_gen_imgs, Config.latent_dim])
	gen_dataset = tf.data.Dataset.from_tensor_slices(gen_noise).batch(Config.num_gen_imgs)
	dist_noise_dataset = strategy.experimental_distribute_dataset(gen_dataset)
	with strategy.scope():

		dcgan = DCGAN(strategy, restore)
		for dist_gen_noise in dist_noise_dataset:
			per_replica_genimgs = dcgan.distribute_genstep(dist_gen_noise)

	filename = Config.results_dir + 'randomFakeGrid'
	save_image_grid(per_replica_genimgs.numpy(), filename, drange=[-1,1], grid_size=Config.grid_size)

def main():

	parser = argparse.ArgumentParser()
	parser.add_argument('--train', help='train DCGAN', action='store_true')
	parser.add_argument('--generate',help='generate images', action='store_true')
	args = parser.parse_args()

	devices = ['/device:GPU:{}'.format(i) for i in range(Config.num_gpu)]

	if(args.train):
		#prepare_tfrecords()
		strategy = tf.distribute.MirroredStrategy(devices)
		train(strategy, restore=False)
	if(args.generate):
		strategy = tf.distribute.OneDeviceStrategy('/device:GPU:0')
		generate(strategy, restore=True)

if __name__=='__main__':
	main()
