import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from set_path import read_path, save_path

file = pd.read_csv('path.txt', header=0, sep='\n')


def train_data():
    """
    prepare train data for AutoEncoder (remove irrelevant data & normalization)
    :return: None
    """
    print("Preparing training data")
    name = pd.read_csv(read_path + 'title.csv', delimiter=',', header=0).columns.values  # 读入表格抬头
    for i in range(len(file)):
        ele = file.iat[i, 0]
        print(ele)
        d_in = np.array(
            pd.read_csv(read_path + 'preprocessing/outlier_processed/' + ele, delimiter=',', encoding='unicode_escape'))
        d_out = normalize(d_in[:, 1:], axis=0)  # 去除时间信息, 归一化
        save_d = np.reshape(d_in[:, 0], (-1, 1))  # 时间信息
        save_d = np.hstack((save_d, d_out))
        save_d = np.vstack((name, save_d))
        pd.DataFrame(save_d).to_csv(save_path + 'feature/train_data/' + ele, header=0, index=0)


class AutoEncoder(object):
    def __init__(self, w_size, step, f_size):
        """
        init parameters for autoencoder
        :param w_size: size of moving window
        :param step: size of step
        :param f_size: size of feature for each sample
        """
        # parameters for NN structure
        self.w_size = w_size  # moving window to select sub-data set
        self.step = step  # step of moving window. We compress step-min-data per process
        self.n_input = 131 * self.w_size  # dimension of input data (Flatter data as a vector)
        self.n_hidden_1 = 64 * self.w_size  # number of nodes in the first hidden layer
        self.n_hidden_2 = 32 * self.w_size  # number of nodes in the second hidden layer
        self.f_size = f_size  # number of compressed feature

        # parameters for training process
        self.lr = 0.01  # learn rate for AutoEncoder
        self.epochs = 15  # number of epoch
        self.d_step = 1  # commander print training status every d_steps

        # parameters for input data
        self.file = file
        self.f_count = 0  # count which file to read
        self.b_count = 0  # count which batch in the file to read
        self.subpath = 'w' + str(w_size) + 's' + str(step) + '/'

        # The structures of encoder and decoder are mutually reversed, W.size=n_input*n_output,bias.size=n_output*1
        # The structure of AutoEncoder: n_input→n_hidden_1→n_hidden_2→f_size→n_hidden_2→n_hidden_1→n_input
        self.weights = {  # define weights for different layers
            'encoder_h1': tf.Variable(tf.random_normal([self.n_input, self.n_hidden_1])),
            'encoder_h2': tf.Variable(tf.random_normal([self.n_hidden_1, self.n_hidden_2])),
            'encoder_h3': tf.Variable(tf.random_normal([self.n_hidden_2, self.f_size])),
            'decoder_h1': tf.Variable(tf.random_normal([self.f_size, self.n_hidden_2])),
            'decoder_h2': tf.Variable(tf.random_normal([self.n_hidden_2, self.n_hidden_1])),
            'decoder_h3': tf.Variable(tf.random_normal([self.n_hidden_1, self.n_input]))
        }
        self.biases = {  # define bias for different layers
            'encoder_b1': tf.Variable(tf.random_normal([self.n_hidden_1])),
            'encoder_b2': tf.Variable(tf.random_normal([self.n_hidden_2])),
            'encoder_b3': tf.Variable(tf.random_normal([self.f_size])),
            'decoder_b1': tf.Variable(tf.random_normal([self.n_hidden_2])),
            'decoder_b2': tf.Variable(tf.random_normal([self.n_hidden_1])),
            'decoder_b3': tf.Variable(tf.random_normal([self.n_input])),
        }

    def get_single_data(self, d_in, time):
        """
        get next batch of data from the dataset
        :return: input data [batch, self.n_input] (each row is a sample)
        """
        start = self.b_count
        if self.b_count + self.step >= d_in.shape[0]:
            end = d_in.shape[0]  # end只能读到file末尾
            self.b_count = 0  # count置零，用于下一个file读取
            self.f_count = self.f_count + 1  # 读取下一个文件
        elif self.b_count + self.w_size >= d_in.shape[0]:
            end = d_in.shape[0]
            self.b_count = self.b_count + self.step
        else:
            end = self.b_count + self.w_size
            self.b_count = self.b_count + self.step
        t = time[start]
        x = np.zeros((self.w_size, 131))
        x[0:end - start, :] = d_in[start:end, 1:]
        x = x.reshape((1, -1))
        return t, x

    def get_batch_data(self, judge):
        ele = self.file.iat[judge, 0]
        time = np.array(pd.read_csv(read_path + 'feature/train_data/' + ele, delimiter=','))[:, 0].reshape(-1, 1)
        d_in = np.array(pd.read_csv(read_path + 'feature/train_data/' + ele, delimiter=','))
        print(ele)
        t, x = self.get_single_data(d_in, time)
        while judge == self.f_count:
            t_temp, x_temp = self.get_single_data(d_in, time)
            t = np.vstack((t, t_temp))
            x = np.vstack((x, x_temp))
        return t, x

    def encoder(self, x):
        """
        Fully-connected NN, activation function: sigmoid
        :param x: input data for encoder
        :return: output value for encoder, which is also the compressed data
        """
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights['encoder_h1']), self.biases['encoder_b1']))
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, self.weights['encoder_h2']), self.biases['encoder_b2']))
        layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, self.weights['encoder_h3']), self.biases['encoder_b3']))
        return layer_3  # return value of encoder part [self.n_input,1]

    def decoder(self, x):
        """
        Fully-connected NN, activation function: sigmoid
        :param x: input data for decoder, which is the output of encoder
        :return: output value for decoder, which is the approximation of the input of encoder
        """
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights['decoder_h1']), self.biases['decoder_b1']))
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, self.weights['decoder_h2']), self.biases['decoder_b2']))
        layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, self.weights['decoder_h3']), self.biases['decoder_b3']))
        return layer_3  # return value of encoder part [self.f_size,1]

    def SSE(self, a, b):
        """
        Sum of Square Error
        :param a: value 1
        :param b: value 2
        :return: SSE of a and b
        """
        return np.sum((a - b) * (a - b))

    def train(self):
        """
        Train AutoEncoder based on train data, and save model
        :return: None
        """
        # tf Graph input
        print("Training Autoencoder")
        X = tf.placeholder("float", [None, self.n_input])  # input data, None means undefined number, here equals b_size
        # Set up model
        encoder_op = self.encoder(X)  # calculate the value of layer_3 through encoder based on input data
        decoder_op = self.decoder(encoder_op)  # calculate the value of output layer through decoder based on encoder_op

        # Prediction
        y_pred = decoder_op  # y_pred is the approximation of input data through AutoEncoder
        y_true = X  # ground_truth value of input X (None*n_input)

        # Define cost and Optimizer. Optimization goal is to make y_pred=y_ture as closely as possible
        cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))  # use SSE as cost function.
        optimizer = tf.train.AdamOptimizer().minimize(cost)  # adapt AdamOptimizer to optimize parameters
        self.f_count = 0
        time_temp, batch_temp = self.get_batch_data(self.f_count)  # load batch data(batch_x:[batch,n_input])
        batch_x = np.zeros((batch_temp.shape[0], batch_temp.shape[1], len(file)))
        batch_x[:, :, 0] = batch_temp
        i = 1
        while self.f_count < len(file):
            time_temp, batch_temp = self.get_batch_data(self.f_count)  # load batch data(batch_x:[batch,n_input])
            batch_x[:, :, i] = batch_temp
            i = i + 1

        saver = tf.train.Saver()
        with tf.Session() as sess:  # Start tensorflow session
            init = tf.global_variables_initializer()  # Initialize weights and biases
            sess.run(init)  # run session based on initial parameter
            for epoch in range(self.epochs):  # Iterate each epoch
                for i in range(batch_x.shape[2]):
                    train_x = batch_x[:, :, i].reshape((batch_x.shape[0], batch_x.shape[1]))
                    _, c = sess.run([optimizer, cost], feed_dict={X: train_x})  # run sess, gain loss and optimize
                if epoch % self.d_step == 0:
                    print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c))  # print loss
            print("Optimization Finished!")
            saver.save(sess, save_path + 'feature/AutoEncoder/model/' + self.subpath + 'model')

    def compress(self):
        """
        Restore trained model, and fetch y_pred in sess, which is also the Encoded data
        :return: None
        """
        print("Compressing data")
        print("Using model in " + read_path + 'feature/AutoEncoder/model/' + self.subpath)
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        model_file = tf.train.latest_checkpoint(read_path + 'feature/AutoEncoder/model/' + self.subpath)
        saver.restore(sess, model_file)  # restore trained model
        X = tf.placeholder("float", [None, self.n_input])  # input data, None means undefined
        encoder_op = self.encoder(X)
        decoder_op = self.decoder(encoder_op)
        self.f_count = 0
        while self.f_count < len(self.file):
            ele = self.file.iat[self.f_count, 0]
            # time = np.array(pd.read_csv(read_path + 'train_data/' + ele, delimiter=','))[:, 0].reshape(-1, 1)
            time, batch_x = self.get_batch_data(self.f_count)  # load batch data(batch_x:[batch,n_input])
            encode_decode = sess.run(encoder_op, feed_dict={X: batch_x})  # compressed data

            out = sess.run(decoder_op, feed_dict={encoder_op: encode_decode})
            np.savetxt(save_path + 'feature/AutoEncoder/data/Example_in.csv', batch_x, fmt='%s', delimiter=',')
            np.savetxt(save_path + 'feature/AutoEncoder/data/Example_out.csv', out, fmt='%s', delimiter=',')

            record = encode_decode.reshape(-1, self.f_size)
            record = np.hstack((time, record))
            np.savetxt(save_path + 'feature/AutoEncoder/data/' + self.subpath + ele, record, fmt='%s', delimiter=',')

    def assemble(self):
        print("assembling data")
        record = np.array(
            pd.read_csv(read_path + 'feature/AutoEncoder/data/' + self.subpath + self.file.iat[0, 0], delimiter=',',
                        header=None))
        for i in range(1, len(file)):
            record = np.vstack((record, np.array(
                pd.read_csv(read_path + 'feature/AutoEncoder/data/' + self.subpath + self.file.iat[i, 0], delimiter=',',
                            header=None))))
        np.savetxt(save_path + 'feature/AutoEncoder/data/' + self.subpath + 'assemble.csv', record, fmt='%s',
                   delimiter=',')
        print("Assembled data saved in " + save_path + 'feature/AutoEncoder/data/' + self.subpath + 'assemble.csv')
