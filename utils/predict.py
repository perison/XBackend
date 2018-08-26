import tensorflow as tf
from tensorflow.contrib import seq2seq
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import os
import pickle

def main():
    data_dir = './data/cp.txt'
    text = load_data(data_dir)
    view_sentence_range = (0, 10)


    print('数据情况：')
    print('不重复单词(彩票开奖记录)的个数: {}'.format(len({word: None for word in text.split()})))
    scenes = text.split('\n\n')
    sentence_count_scene = [scene.count('\n') for scene in scenes]
    print('开奖期数: {}期'.format(int(np.average(sentence_count_scene))))

    sentences = [sentence for scene in scenes for sentence in scene.split('\n')]
    print('行数: {}'.format(len(sentences)))
    word_count_sentence = [len(sentence.split()) for sentence in sentences]
    print('平均每行单词数: {}'.format(np.ceil(np.average(word_count_sentence))))

    print()
    print('开奖记录从 {} 到 {}:'.format(*view_sentence_range))
    print('\n'.join(text.split('\n')[view_sentence_range[0]:view_sentence_range[1]]))

    # Preprocess Training, Validation, and Testing Data
    preprocess_and_save_data(data_dir, create_lookup_tables)

    int_text, vocab_to_int, int_to_vocab = load_preprocess()

    '''
    num_epochs 设置训练几代。
    batch_size 是批次大小。
    rnn_size 是RNN的大小（隐藏节点的维度）。
    embed_dim 是嵌入层的维度。
    seq_length 是序列的长度，始终为1。
    learning_rate 是学习率。
    show_every_n_batches 是过多少batch以后打印训练信息。
    '''

    # Number of Epochs
    num_epochs = 25
    # Batch Size
    batch_size = 32
    # RNN Size
    rnn_size = 1000
    # Embedding Dimension Size
    embed_dim = 1000
    # Sequence Length
    seq_length = 1
    # Learning Rate
    learning_rate = 0.01
    # Show stats for every n number of batches
    show_every_n_batches = 10

    save_dir = './save'

    tf.reset_default_graph()
    train_graph = tf.Graph()
    with train_graph.as_default():
        vocab_size = len(int_to_vocab)
        input_text, targets, lr = get_inputs()
        input_data_shape = tf.shape(input_text)
        cell, initial_state = get_init_cell(input_data_shape[0], rnn_size)
        logits, final_state, embed_matrix = build_nn(cell, rnn_size, input_text, vocab_size, embed_dim)

        # Probabilities for generating words
        probs = tf.nn.softmax(logits, name='probs')

        # Loss function
        cost = seq2seq.sequence_loss(
            logits,
            targets,
            tf.ones([input_data_shape[0], input_data_shape[1]]))
        #     cost = build_loss(logits, targets, vocab_size)

        # We use the cosine distance:
        norm = tf.sqrt(tf.reduce_sum(tf.square(embed_matrix), 1, keep_dims=True))
        normalized_embedding = embed_matrix / norm

        probs_embeddings = tf.nn.embedding_lookup(normalized_embedding,
                                                  tf.squeeze(tf.argmax(probs, 2)))  # np.squeeze(probs.argmax(2))
        probs_similarity = tf.matmul(probs_embeddings, tf.transpose(normalized_embedding))

        y_embeddings = tf.nn.embedding_lookup(normalized_embedding, tf.squeeze(targets))
        y_similarity = tf.matmul(y_embeddings, tf.transpose(normalized_embedding))

        #     data_moments = tf.reduce_mean(y_similarity, axis=0)
        #     sample_moments = tf.reduce_mean(probs_similarity, axis=0)
        similar_loss = tf.reduce_mean(tf.abs(y_similarity - probs_similarity))
        total_loss = cost + similar_loss

        # Optimizer
        optimizer = tf.train.AdamOptimizer(lr)

        # Gradient Clipping
        gradients = optimizer.compute_gradients(total_loss)  # cost
        capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if
                            grad is not None]  # clip_by_norm
        train_op = optimizer.apply_gradients(capped_gradients)

        # Accuracy
        correct_pred = tf.equal(tf.argmax(probs, 2),
                                tf.cast(targets, tf.int64))  # logits <--> probs  tf.argmax(targets, 1) <--> targets
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

    batches = get_batches(int_text[:-(batch_size + 1)], batch_size, seq_length)
    test_batches = get_batches(int_text[-(batch_size + 1):], batch_size, seq_length)
    top_k = 10
    topk_acc_list = []
    topk_acc = 0
    sim_topk_acc_list = []
    sim_topk_acc = 0

    range_k = 5
    floating_median_idx = 0
    floating_median_acc_range_k = 0
    floating_median_acc_range_k_list = []

    floating_median_sim_idx = 0
    floating_median_sim_acc_range_k = 0
    floating_median_sim_acc_range_k_list = []

    losses = {'train': [], 'test': []}
    accuracies = {'accuracy': [], 'topk': [], 'sim_topk': [], 'floating_median_acc_range_k': [],
                  'floating_median_sim_acc_range_k': []}

    with tf.Session(graph=train_graph) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        for epoch_i in range(num_epochs):
            state = sess.run(initial_state, {input_text: batches[0][0]})

            # 训练的迭代，保存训练损失
            for batch_i, (x, y) in enumerate(batches):
                feed = {
                    input_text: x,
                    targets: y,
                    initial_state: state,
                    lr: learning_rate}
                train_loss, state, _ = sess.run([total_loss, final_state, train_op], feed)  # cost
                losses['train'].append(train_loss)

                # Show every <show_every_n_batches> batches
                if (epoch_i * len(batches) + batch_i) % show_every_n_batches == 0:
                    print('Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(
                        epoch_i,
                        batch_i,
                        len(batches),
                        train_loss))

            # 使用测试数据的迭代
            acc_list = []
            prev_state = sess.run(initial_state, {input_text: np.array([[1]])})  # test_batches[0][0]
            for batch_i, (x, y) in enumerate(test_batches):
                # Get Prediction
                test_loss, acc, probabilities, prev_state = sess.run(
                    [total_loss, accuracy, probs, final_state],
                    {input_text: x,
                     targets: y,
                     initial_state: prev_state})  # cost

                # 保存测试损失和准确率
                acc_list.append(acc)
                losses['test'].append(test_loss)
                accuracies['accuracy'].append(acc)

                print('Epoch {:>3} Batch {:>4}/{}   test_loss = {:.3f}'.format(
                    epoch_i,
                    batch_i,
                    len(test_batches),
                    test_loss))

                # 利用嵌入矩阵和生成的预测计算得到相似度矩阵sim
                valid_embedding = tf.nn.embedding_lookup(normalized_embedding, np.squeeze(probabilities.argmax(2)))
                similarity = tf.matmul(valid_embedding, tf.transpose(normalized_embedding))
                sim = similarity.eval()

                # 保存预测结果的Top K准确率和与预测结果距离最近的Top K准确率
                topk_acc = 0
                sim_topk_acc = 0
                for ii in range(len(probabilities)):

                    nearest = (-sim[ii, :]).argsort()[0:top_k]
                    if y[ii] in nearest:
                        sim_topk_acc += 1

                    if y[ii] in (-probabilities[ii]).argsort()[0][0:top_k]:
                        topk_acc += 1

                topk_acc = topk_acc / len(y)
                topk_acc_list.append(topk_acc)
                accuracies['topk'].append(topk_acc)

                sim_topk_acc = sim_topk_acc / len(y)
                sim_topk_acc_list.append(sim_topk_acc)
                accuracies['sim_topk'].append(sim_topk_acc)

                # 计算真实值在预测值中的距离数据
                realInSim_distance_list = []
                realInPredict_distance_list = []
                for ii in range(len(probabilities)):
                    sim_nearest = (-sim[ii, :]).argsort()
                    idx = list(sim_nearest).index(y[ii])
                    realInSim_distance_list.append(idx)

                    nearest = (-probabilities[ii]).argsort()[0]
                    idx = list(nearest).index(y[ii])
                    realInPredict_distance_list.append(idx)

                print('真实值在预测值中的距离数据：')
                print('max distance : {}'.format(max(realInPredict_distance_list)))
                print('min distance : {}'.format(min(realInPredict_distance_list)))
                print('平均距离 : {}'.format(np.mean(realInPredict_distance_list)))
                print('距离中位数 : {}'.format(np.median(realInPredict_distance_list)))
                print('距离标准差 : {}'.format(np.std(realInPredict_distance_list)))

                print('真实值在预测值相似向量中的距离数据：')
                print('max distance : {}'.format(max(realInSim_distance_list)))
                print('min distance : {}'.format(min(realInSim_distance_list)))
                print('平均距离 : {}'.format(np.mean(realInSim_distance_list)))
                print('距离中位数 : {}'.format(np.median(realInSim_distance_list)))
                print('距离标准差 : {}'.format(np.std(realInSim_distance_list)))
                #             sns.distplot(realInPredict_distance_list, rug=True)  #, hist=False
                # plt.hist(np.log(realInPredict_distance_list), bins=50, color='steelblue', normed=True )

                # 计算以距离中位数为中心，范围K为半径的准确率
                floating_median_sim_idx = int(np.median(realInSim_distance_list))
                floating_median_sim_acc_range_k = 0

                floating_median_idx = int(np.median(realInPredict_distance_list))
                floating_median_acc_range_k = 0
                for ii in range(len(probabilities)):
                    nearest_floating_median = (-probabilities[ii]).argsort()[0][
                                              floating_median_idx - range_k:floating_median_idx + range_k]
                    if y[ii] in nearest_floating_median:
                        floating_median_acc_range_k += 1

                    nearest_floating_median_sim = (-sim[ii, :]).argsort()[
                                                  floating_median_sim_idx - range_k:floating_median_sim_idx + range_k]
                    if y[ii] in nearest_floating_median_sim:
                        floating_median_sim_acc_range_k += 1

                floating_median_acc_range_k = floating_median_acc_range_k / len(y)
                floating_median_acc_range_k_list.append(floating_median_acc_range_k)
                accuracies['floating_median_acc_range_k'].append(floating_median_acc_range_k)

                floating_median_sim_acc_range_k = floating_median_sim_acc_range_k / len(y)
                floating_median_sim_acc_range_k_list.append(floating_median_sim_acc_range_k)
                accuracies['floating_median_sim_acc_range_k'].append(floating_median_sim_acc_range_k)

            print('Epoch {:>3} floating median sim range k accuracy {} '.format(epoch_i, np.mean(
                floating_median_sim_acc_range_k_list)))  #:.3f
            print('Epoch {:>3} floating median range k accuracy {} '.format(epoch_i, np.mean(
                floating_median_acc_range_k_list)))  #:.3f
            print('Epoch {:>3} similar top k accuracy {} '.format(epoch_i, np.mean(sim_topk_acc_list)))  #:.3f
            print('Epoch {:>3} top k accuracy {} '.format(epoch_i, np.mean(topk_acc_list)))  #:.3f
            print('Epoch {:>3} accuracy {} '.format(epoch_i, np.mean(acc_list)))  #:.3f

        # Save Model
        saver.save(sess, save_dir)  # , global_step=epoch_i
        print('Model Trained and Saved')
        embed_mat = sess.run(normalized_embedding)


    sns.distplot(realInSim_distance_list, rug=True)
    sns.distplot(realInPredict_distance_list, rug=True)

    plt.plot(losses['train'], label='Training loss')
    plt.legend()
    _ = plt.ylim()

    plt.plot(losses['test'], label='Test loss')
    plt.legend()
    _ = plt.ylim()

    plt.plot(accuracies['accuracy'], label='Accuracy')
    plt.plot(accuracies['topk'], label='Top K')
    plt.plot(accuracies['sim_topk'], label='Similar Top K')
    plt.plot(accuracies['floating_median_acc_range_k'], label='Floating Median Range K Acc')
    plt.plot(accuracies['floating_median_sim_acc_range_k'], label='Floating Median Sim Range K Acc')
    plt.legend()
    _ = plt.ylim()

    for batch_i, (x, y) in enumerate(test_batches):
        plt.plot(y, label='Targets')
        plt.plot(np.squeeze(probabilities.argmax(2)), label='Prediction')
        plt.legend()
        _ = plt.ylim()

    # Save parameters for checkpoint
    save_params((seq_length, save_dir))

    _, vocab_to_int, int_to_vocab = load_preprocess()
    seq_length, load_dir = load_params()

    with train_graph.as_default():
        saver = tf.train.Saver()

    with tf.Session(graph=train_graph) as sess:
        # Load saved model
        loader = tf.train.import_meta_graph(load_dir + '.meta')
        loader.restore(sess, load_dir)

        #     saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
        embed_mat = sess.run(embed_matrix)

    viz_words = 1000
    tsne = TSNE()
    with train_graph.as_default():
        embed_tsne = tsne.fit_transform(embed_mat[:viz_words, :])

    fig, ax = plt.subplots(figsize=(24, 24))
    for idx in range(viz_words):
        plt.scatter(*embed_tsne[idx, :], color='steelblue')
        plt.annotate(int_to_vocab[idx], (embed_tsne[idx, 0], embed_tsne[idx, 1]), alpha=0.7)

    gen_length = 17
    prime_word = '202'

    loaded_graph = tf.Graph()  # loaded_graph
    with tf.Session(graph=train_graph) as sess:
        # Load saved model
        loader = tf.train.import_meta_graph(load_dir + '.meta')
        loader.restore(sess, load_dir)

        # Get Tensors from loaded model
        input_text, initial_state, final_state, probs = get_tensors(train_graph)  # loaded_graph

        # Sentences generation setup
        gen_sentences = [prime_word]
        prev_state = sess.run(initial_state, {input_text: np.array([[1]])})

        # Generate sentences
        for n in range(gen_length):
            # Dynamic Input
            dyn_input = [[vocab_to_int[word] for word in gen_sentences[-seq_length:]]]
            dyn_seq_length = len(dyn_input[0])

            # Get Prediction
            probabilities, prev_state = sess.run(
                [probs, final_state],
                {input_text: dyn_input, initial_state: prev_state})

            valid_embedding = tf.nn.embedding_lookup(normalized_embedding, probabilities.argmax())
            valid_embedding = tf.reshape(valid_embedding, (1, len(int_to_vocab)))
            similarity = tf.matmul(valid_embedding, tf.transpose(normalized_embedding))
            sim = similarity.eval()

            pred_word = pick_word(probabilities[dyn_seq_length - 1], sim, int_to_vocab, 5, 'median')

            gen_sentences.append(pred_word)

        cp_script = ' '.join(gen_sentences)
        cp_script = cp_script.replace('\n ', '\n')
        cp_script = cp_script.replace('( ', '(')

        print(cp_script)

    int_sentences = [int(words) for words in gen_sentences]
    int_sentences = int_sentences[1:]

    val_data = [[103], [883], [939], [36], [435], [173], [572], [828], [509], [723], [145], [621], [535], [385],
                [98], [321], [427]]

    plt.plot(int_sentences, label='History')
    plt.plot(val_data, label='val_data')
    plt.legend()
    _ = plt.ylim()



# def build_loss(logits, targets, num_classes):
#     ''' Calculate the loss from the logits and the targets.

#         Arguments
#         ---------
#         logits: Logits from final fully connected layer
#         targets: Targets for supervised learning
#         num_classes: Number of classes in targets

#     '''
#     y_one_hot = tf.one_hot(tf.squeeze(targets), num_classes)
#     y_reshaped = tf.reshape(y_one_hot, (batch_size, num_classes))

#     # Softmax cross entropy loss
#     loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped)
#     loss = tf.reduce_mean(loss)
#     return loss

def get_tensors(loaded_graph):
    """
    Get input, initial state, final state, and probabilities tensor from <loaded_graph>
    :param loaded_graph: TensorFlow graph loaded from file
    :return: Tuple (InputTensor, InitialStateTensor, FinalStateTensor, ProbsTensor)
    """
    inputs = loaded_graph.get_tensor_by_name("input:0")
    initial_state = loaded_graph.get_tensor_by_name("initial_state:0")
    final_state = loaded_graph.get_tensor_by_name("final_state:0")
    probs = loaded_graph.get_tensor_by_name("probs:0")
    return inputs, initial_state, final_state, probs

def pick_word(probabilities, sim, int_to_vocab, top_n = 5, pred_mode = 'sim'):
    """
    Pick the next word in the generated text
    :param probabilities: Probabilites of the next word
    :param int_to_vocab: Dictionary of word ids as the keys and words as the values
    :param use_max: use max probabilities number
    :param top_n: Top number
    :return: String of the predicted word
    """
    vocab_size = len(int_to_vocab)

    if pred_mode == 'sim':
        p = np.squeeze(sim)
        p[np.argsort(p)[:-top_n]] = 0
        p = p / np.sum(p)
        c = np.random.choice(vocab_size, 1, p=p)[0]
        return int_to_vocab[c]
    elif pred_mode == 'median':
        p = np.squeeze(sim)
        p[np.argsort(p)[:floating_median_sim_idx - top_n]] = 0
        p[np.argsort(p)[floating_median_sim_idx + top_n:]] = 0
        p = np.abs(p) / np.sum(np.abs(p))
        c = np.random.choice(vocab_size, 1, p=p)[0]
        return int_to_vocab[c]
    elif pred_mode == 'topk':
        p = np.squeeze(probabilities)
        p[np.argsort(p)[:-top_n]] = 0
        p = p / np.sum(p)
        c = np.random.choice(vocab_size, 1, p=p)[0]
        return int_to_vocab[c]
    elif pred_mode == 'max':
        return int_to_vocab[probabilities.argmax()]

def get_inputs():
    """
    Create TF Placeholders for input, targets, and learning rate.
    :return: Tuple (input, targets, learning rate)
    """
    inputs = tf.placeholder(tf.int32, [None, None], name="input")
    targets = tf.placeholder(tf.int32, [None, None], name="targets")
    LearningRate = tf.placeholder(tf.float32)
    return inputs, targets, LearningRate

def get_init_cell(batch_size, rnn_size):
    """
    Create an RNN Cell and initialize it.
    :param batch_size: Size of batches
    :param rnn_size: Size of RNNs
    :return: Tuple (cell, initialize state)
    """
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(rnn_size)#num_units=embed_dim
    cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * 2)
    InitialState = cell.zero_state(batch_size, tf.float32)
    InitialState = tf.identity(InitialState, name="initial_state")
    return cell, InitialState

def get_embed(input_data, vocab_size, embed_dim):
    """
    Create embedding for <input_data>.
    :param input_data: TF placeholder for text input.
    :param vocab_size: Number of words in vocabulary.
    :param embed_dim: Number of embedding dimensions
    :return: Tuple (Embedded input, embed_matrix)
    """
    embed_matrix = tf.Variable(tf.random_uniform([vocab_size, embed_dim], -1, 1))
    embed_layer = tf.nn.embedding_lookup(embed_matrix, input_data)
    return embed_layer, embed_matrix

def build_rnn(cell, inputs):
    """
    Create a RNN using a RNN Cell
    :param cell: RNN Cell
    :param inputs: Input text data
    :return: Tuple (Outputs, Final State)
    """
    Outputs, final_State = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
    final_State = tf.identity(final_State, name="final_state")
    return Outputs, final_State

def build_nn(cell, rnn_size, input_data, vocab_size, embed_dim):
    """
    Build part of the neural network
    :param cell: RNN cell
    :param rnn_size: Size of rnns
    :param input_data: Input data
    :param vocab_size: Vocabulary size
    :param embed_dim: Number of embedding dimensions
    :return: Tuple (Logits, FinalState, embed_matrix)
    """
    embed_layer, embed_matrix = get_embed(input_data, vocab_size, embed_dim)
    Outputs, final_State = build_rnn(cell, embed_layer)
    logits = tf.layers.dense(Outputs, vocab_size)
    return logits, final_State, embed_matrix

def get_batches(int_text, batch_size, seq_length):
    """
    Return batches of input and target
    :param int_text: Text with the words replaced by their ids
    :param batch_size: The size of batch
    :param seq_length: The length of sequence
    :return: Batches as a Numpy array
    """
    batchCnt = len(int_text) // (batch_size * seq_length)
    int_text_inputs = int_text[:batchCnt * (batch_size * seq_length)]
    int_text_targets = int_text[1:batchCnt * (batch_size * seq_length)+1]

    result_list = []
    x = np.array(int_text_inputs).reshape(1, batch_size, -1)
    y = np.array(int_text_targets).reshape(1, batch_size, -1)

    x_new = np.dsplit(x, batchCnt)
    y_new = np.dsplit(y, batchCnt)

    for ii in range(batchCnt):
        x_list = []
        x_list.append(x_new[ii][0])
        x_list.append(y_new[ii][0])
        result_list.append(x_list)

    return np.array(result_list)


# get_batches([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], 2, 3)

def create_lookup_tables():
    """
    Create lookup tables for vocabulary
    :param text: The text of tv scripts split into words
    :return: A tuple of dicts (vocab_to_int, int_to_vocab)
    """
    vocab_to_int = {str(ii).zfill(3) : ii for ii in range(1000)}
    int_to_vocab = {ii : str(ii).zfill(3) for ii in range(1000)}
    return vocab_to_int, int_to_vocab

def load_data(path):
    """
    Load Dataset from File
    """
    input_file = os.path.join(path)
    with open(input_file, "r") as f:
        data = f.read()

    return data


def preprocess_and_save_data(dataset_path, create_lookup_tables):
    """
    Preprocess Text Data
    """
    text = load_data(dataset_path)

    text = text.lower()
    # text = text.split()

    words = [word for word in text.split()]

    reverse_words = [text.split()[idx] for idx in (range(len(words) - 1, 0, -1))]
    vocab_to_int, int_to_vocab = create_lookup_tables()  # text
    # int_text = [vocab_to_int[word] for word in text]
    int_text = [vocab_to_int[word] for word in reverse_words]
    pickle.dump((int_text, vocab_to_int, int_to_vocab), open('preprocess.p', 'wb'))


def load_preprocess():
    """
    Load the Preprocessed Training data and return them in batches of <batch_size> or less
    """
    return pickle.load(open('preprocess.p', mode='rb'))


def save_params(params):
    """
    Save parameters to file
    """
    pickle.dump(params, open('params.p', 'wb'))


def load_params():
    """
    Load parameters from file
    """
    return pickle.load(open('params.p', mode='rb'))