
import numpy as np
import matplotlib.pyplot as plt
from vclab.classifiers.neural_net import TwoLayerNet
from vclab.data_utils import load_CIFAR10


def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the two-layer neural net classifier. These are the same steps as
    we used for the SVM, but condensed to a single function.
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = 'vclab/datasets/cifar-10-batches-py'

    # Cleaning up variables to prevent loading data multiple times (which may cause memory issue)
    try:
        del X_train, y_train
        del X_test, y_test
        print('Clear previously loaded data.')
    except:
        pass

    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # Subsample the data
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    # Reshape data to rows
    X_train = X_train.reshape(num_training, -1)
    X_val = X_val.reshape(num_validation, -1)
    X_test = X_test.reshape(num_test, -1)

    return X_train, y_train, X_val, y_val, X_test, y_test

def train_and_validate(num_iters=1000, batch_size=200,
                learning_rate=1e-4, learning_rate_decay=0.95,
                reg=0.25):
    X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()
    input_size = 32 * 32 * 3
    num_classes = 10
    hidden_size = 50
    best_net = None
    best_acc = 0.0
    best_parameter = None

    # 构造待测试的参数表
    initial_parameter = {'hidden_size': hidden_size, 'num_iters': num_iters, 'batch_size': batch_size,
                        'learning_rate': learning_rate, 'learning_rate_decay': learning_rate_decay, 'reg': reg,
                        'val_acc': 0.0}
    initial_parameter['learning_rate'] = 1e-3
    initial_parameter['hidden_size'] = 100
    initial_parameter['num_iters'] = 1500
    parameter_list = [initial_parameter.copy() for i in range(12)]
    parameter_list[1]['hidden_size'] = 30
    parameter_list[2]['hidden_size'] = 70
    parameter_list[3]['learning_rate'] = 5e-4
    parameter_list[4]['learning_rate'] = 1.1e-3
    parameter_list[5]['num_iters'] = 500
    parameter_list[6]['num_iters'] = 1500
    parameter_list[7]['reg'] = 0.20
    parameter_list[8]['reg'] = 0.30
    parameter_list[9]['batch_size'] = 500
    parameter_list[10]['learning_rate'] = 1e-3
    parameter_list[10]['hidden_size'] = 70
    parameter_list[11]['learning_rate'] = 1e-3
    parameter_list[11]['hidden_size'] = 100

    # training and validation
    for (i, parameter) in enumerate(parameter_list):
        # build net
        net = TwoLayerNet(input_size, parameter['hidden_size'], num_classes)

        # Train the network
        print('\nstart training : (%d/%d)\n\t' % (i, len(parameter_list)), parameter)
        stats = net.train(X_train, y_train, X_val, y_val,
                    num_iters=parameter['num_iters'], batch_size=parameter['batch_size'],
                    learning_rate=parameter['learning_rate'], learning_rate_decay=parameter['learning_rate_decay'],
                    reg=parameter['reg'], verbose=True)
        # plt.figure()
        plt.plot(stats['loss_history'])
        plt.title('#[%02d]  Loss history' % i)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.show()

        # Predict on the validation set
        val_acc = (net.predict(X_val) == y_val).mean()
        parameter['val_acc'] = val_acc
        print('end training : (%d/%d)\n\t' % (i, len(parameter_list)), parameter, '\n')

        # save best net
        if val_acc > best_acc:
            best_acc = val_acc
            best_parameter = parameter
            best_net = net
    return best_net, best_parameter, parameter_list

# train and validate
best_net, best_parameter, parameter_list = train_and_validate()
print('best_parameter:\n ', best_parameter)