import tensorflow as tf
import os
import numpy as np
import scipy.io as sio

flags = tf.app.flags
flags.DEFINE_string("dataset", "./data/data4train.mat", "The path to dataset")
flags.DEFINE_integer("epoch", 2000001, "how much time to train [2000001]")
flags.DEFINE_float("learning_rate", 0.0000005, "Learning rate of for RMS [0.0000005]")
flags.DEFINE_integer("batch_size", 11, "The size of batch images [64]")
flags.DEFINE_integer("data_dim", 1000, "data dimension")
flags.DEFINE_integer("train_times", 3, "Discriminator train times per epoch [3,4,5]")
flags.DEFINE_integer("sample_rate", 50000, "how many epoch you want to sample once[50000]")
flags.DEFINE_string("checkpoint_dir", "./checkpoint/", "Directory name to save the checkpoints")
flags.DEFINE_string("extractor_dir", "./save/CNN.ckpt", "Directory name to the extractor")
flags.DEFINE_string("train_data", "x1", "number of signal you want to train[x1,x2,x3,x4,x5,x6,x7,x8,x9]")
FLAGS = flags.FLAGS

if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)

matfn=FLAGS.dataset
f = sio.loadmat(matfn)
fake_data = f['x0']
fake_data = np.array(fake_data)
real_data = f[FLAGS.train_data]
real_data = np.array(real_data)

if FLAGS.train_data == 'x3' or FLAGS.train_data == 'x7':
    tr = 3.
elif FLAGS.train_data =='x4':
    tr = 2.
elif FLAGS.train_data =='x9':
    tr= 3.5
elif FLAGS.train_data =='x1':
    tr= 1.5 
else:
    tr = 1.
real_data = real_data/tr
def LeakyReLu(x, alpha=0.1):
    x = tf.maximum(alpha*x,x)
    return x

batch_size = FLAGS.batch_size
data_dim = FLAGS.data_dim
epochs = FLAGS.epoch
LR = FLAGS.learning_rate
sess = tf.Session()

L = list(range(1,len(real_data)+1))
def real_batch(inputD, size, e):
    X = np.zeros([size,1000])
    if e%batch_size == 0:
        np.random.shuffle(L)
    e = e%batch_size * size
    for j in range(size):
        X[j] = inputD[L[e+j]-1]
    return X

L2 = list(range(1,len(fake_data)+1))
def fake_batch(inputD, size, e):
    X = np.zeros([size,1000])
    if e%batch_size == 0:
        np.random.shuffle(L2)
    e = e%batch_size * size
    for j in range(size):
        X[j] = inputD[L2[e+j]-1]
    return X

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
def my_init(size):
    return tf.random_uniform(size, -0.05, 0.05)
def conv1d(x, W, s):
    return tf.nn.conv1d(x, W, stride=s, padding='SAME')

with tf.name_scope('Generator') as scope:
    GW_conv1 = weight_variable([5, 1, 32])
    Gb_conv1 = bias_variable([32])
    GW_conv2 = weight_variable([5, 32, 32])
    Gb_conv2 = bias_variable([32])
    GW_conv3 = weight_variable([5, 32, 32])
    Gb_conv3 = bias_variable([32])
    GW_conv4 = weight_variable([5, 32, 32])
    Gb_conv4 = bias_variable([32])
    GW_conv5 = weight_variable([5, 32, 1])
    Gb_conv5 = bias_variable([1])
    # GW_conv6 = weight_variable([5, 250, 500])
    # Gb_conv6 = bias_variable([500])
    # GW_conv7 = weight_variable([5, 500, 1000])
    # Gb_conv7 = bias_variable([1000])
    GW = weight_variable([1 * 1000, 1000])
    Gb = bias_variable([1000])
    G_variables = [GW_conv1, Gb_conv1, GW_conv2, Gb_conv2,
                   GW_conv3, Gb_conv3, GW_conv4, Gb_conv4,
                   GW_conv5, Gb_conv5, GW, Gb]
    def G(X):
        X = tf.nn.relu(conv1d(X, GW_conv1, 1) + Gb_conv1) #-->1000*32
        X = tf.nn.relu(conv1d(X, GW_conv2, 1) + Gb_conv2) #-->1000*32
        X = tf.nn.relu(conv1d(X, GW_conv3, 1) + Gb_conv3) #-->1000*32
        X = tf.nn.relu(conv1d(X, GW_conv4, 1) + Gb_conv4) #-->1000*32
        X = tf.nn.relu(conv1d(X, GW_conv5, 1) + Gb_conv5) #-->1000*1
        # X = tf.nn.relu(conv1d(X, GW_conv6, 1) + Gb_conv6) #-->2*500
        # X = tf.nn.relu(conv1d(X, GW_conv7, 2) + Gb_conv7) #-->1*1000
        X = tf.reshape(X, [-1, 1 * 1000])
        X = tf.nn.tanh(tf.matmul(X, GW) + Gb)
        return X

with tf.name_scope('Discriminator') as scope:
    DW_conv1 = weight_variable([5, 1, 16])
    Db_conv1 = bias_variable([16])
    DW_conv2 = weight_variable([5, 16, 32])
    Db_conv2 = bias_variable([32])
    DW_conv3 = weight_variable([5, 32, 64])
    Db_conv3 = bias_variable([64])
    DW_conv4 = weight_variable([5, 64, 128])
    Db_conv4 = bias_variable([128])
    DW_conv5 = weight_variable([5, 128, 256])
    Db_conv5 = bias_variable([256])
    # DW_conv6 = weight_variable([5, 256, 512])
    # Db_conv6 = bias_variable([512])
    # DW_conv7 = weight_variable([5, 512, 1024])
    # Db_conv7 = bias_variable([1024])
    DW = weight_variable([5 * 256, 1])
    Db = bias_variable([1])
    D_variables = [DW_conv1, Db_conv1, DW_conv2, Db_conv2,
                   DW_conv3, Db_conv3, DW_conv4, Db_conv4,
                   DW_conv5, Db_conv5, DW, Db]
    def D(X):
        X = LeakyReLu(conv1d(X, DW_conv1, 2) + Db_conv1) #-->500*16
        X = LeakyReLu(conv1d(X, DW_conv2, 5) + Db_conv2) #-->100*32
        X = LeakyReLu(conv1d(X, DW_conv3, 2) + Db_conv3) #-->50*64
        X = LeakyReLu(conv1d(X, DW_conv4, 5) + Db_conv4) #-->10*128
        X = LeakyReLu(conv1d(X, DW_conv5, 2) + Db_conv5) #-->5*256
        # X = LeakyReLu(conv1d(X, DW_conv6, 5) + Db_conv6) #-->4*256
        # X = LeakyReLu(conv1d(X, DW_conv7, 2) + Db_conv7) #-->2*512
        X = tf.reshape(X, [-1, 5 * 256])
        X = X = tf.nn.tanh(tf.matmul(X, DW) + Db)
        return X

W_conv5 = tf.Variable(tf.constant(0.1, shape=[5, 128, 256]), name="W_conv5")
b_conv5 = tf.Variable(tf.constant(0.1, shape=[256]), name="b_conv5")
saver = tf.train.Saver({'W_conv5': W_conv5, 'b_conv5': b_conv5})
saver.restore(sess, FLAGS.extractor_dir)
W_conv5 = tf.reshape(W_conv5[0:5,0,0], [5,1,1])
b_conv5 = tf.reshape(b_conv5[0], [1])
W_conv5 = tf.constant(W_conv5.eval(session=sess))
b_conv5 = tf.constant(b_conv5.eval(session=sess))
def C(X):
    Con = tf.nn.conv1d(X, W_conv5, stride=1, padding='SAME') + b_conv5
    return Con

real_X = tf.placeholder(tf.float32, shape=[None, data_dim], name='real_input')
fake_X = tf.placeholder(tf.float32, shape=[None, data_dim], name='fake_input')
real_X_shaped = tf.reshape(real_X, [-1, data_dim, 1])
fake_X_shaped = tf.reshape(fake_X, [-1, data_dim, 1])
fake_Y = G(fake_X_shaped)
fake_Y = tf.reshape(fake_Y, [-1, data_dim])
fake_Y_shaped = tf.reshape(fake_Y, [-1, data_dim, 1])

output = fake_Y*tr

eps = tf.random_uniform([batch_size, 1, 1], minval=0., maxval=1.)
X_inter = eps * real_X_shaped + (1. - eps) * fake_Y_shaped
grad = tf.gradients(D(X_inter), [X_inter])[0]
grad_norm = tf.sqrt(tf.reduce_sum((grad)**2, axis=1))
grad_pen = 10 * tf.reduce_mean(tf.nn.relu(grad_norm - 1.))

res = tf.square(C(real_X_shaped) - C(fake_Y_shaped))
res = tf.reshape(res, [11, data_dim])
ploss = tf.reduce_sum(res, 1)*0.00001

D_loss = tf.reduce_mean(D(fake_Y_shaped)) - tf.reduce_mean(D(real_X_shaped)) + grad_pen
tf.summary.scalar('D_loss', D_loss)
G_loss = tf.reduce_mean(ploss - D(fake_Y_shaped))
tf.summary.scalar('G_loss', G_loss)
D_solver = tf.train.RMSPropOptimizer(LR).minimize(D_loss, var_list=D_variables)
G_solver = tf.train.RMSPropOptimizer(LR).minimize(G_loss, var_list=G_variables)

if not os.path.exists('./samples/'):
    os.makedirs('./samples/')
    
saver = tf.train.Saver()
model = FLAGS.checkpoint_dir+'/GAN1D'
sess.run(tf.global_variables_initializer())

for e in range(epochs):
    for i in range(Flags.train_times):
        real_batch_X = real_batch(real_data, 11, (e+1)*(i+1))
        fake_batch_X = fake_batch(fake_data, 11, (e+1)*(i+1))
        _,D_loss_ = sess.run([D_solver,D_loss],
                             feed_dict={real_X:real_batch_X, fake_X:fake_batch_X})
    fake_batch_X = fake_batch(fake_data,11, (e+1)*(i+1))
    _,G_loss_ = sess.run([G_solver,G_loss], feed_dict={real_X:real_batch_X, fake_X: fake_batch_X})

    if e % 1000 == 0:
        print ('epoch %s, D_loss: %s, G_loss: %s'%(e//1000, D_loss_, G_loss_))

    if e % 100000 == 0:
        saver.save(sess, model, global_step=e)

    if e % FLAGS.sample_rate == 0:
        output_csv = sess.run(output, feed_dict={fake_X: fake_batch_X})
        output_csv = np.array(output_csv)
        np.savetxt("./samples/"+"sample_"+str(e)+'.csv', output_csv, fmt="%f", delimiter=",")
