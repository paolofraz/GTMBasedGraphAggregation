from textwrap import wrap

import numpy as np
import os
import matplotlib.pyplot as plt


def load_file(result_folder, test_name, n_split, suffix=""):
    split_result = []
    # Load file
    for i in range(n_split):
        test_res_from_file = np.loadtxt(os.path.join(result_folder, suffix + test_name + "--split-" + str(i) + "_test"),
                                        skiprows=2)
        train_res_from_file = np.loadtxt(
            os.path.join(result_folder, suffix + test_name + "--split-" + str(i) + "_train"), skiprows=2)
        valid_res_from_file = np.loadtxt(
            os.path.join(result_folder, suffix + test_name + "--split-" + str(i) + "_valid"), skiprows=2)
        split_result.append((train_res_from_file, test_res_from_file, valid_res_from_file))
    return split_result


def perform_validation_loss(log_file, split_result, n_epoch, test_epoch):
    log_file.write("Validation based on loss value\n")
    print("Validation based on loss value")
    # find the best epoch in validation
    split_valid_epoch = []
    acc_split = []
    # validation on loss

    for i, split in enumerate(split_result):
        valid_res = split[2]

        # get loss column
        loss_list = valid_res[0:n_epoch,
                    2]  # NOTE:validation based on loss value, if it has to be computed on acc change index with 3, and use argmax 2 rows below

        # get the index (epoch) of min loss
        val_epoch = np.argmin(loss_list)

        split_valid_epoch.append(val_epoch)
        # get accuracy in the test ste of da validated epoch
        test_split = split[1]
        # acc col is the row 3 of the table
        acc = test_split[val_epoch, 3]
        print("split: ", str(i), " epoch: ", str(val_epoch * test_epoch), " Acc: ", acc)
        log_file.write("split: " + str(i) + " epoch: " + str(val_epoch * test_epoch) + " Acc: " + str(acc) + "\n")
        acc_split.append(acc)

    print(np.mean(np.asarray(acc_split)))
    log_file.write("\nAVG Acc: " + str(np.mean(np.asarray(acc_split))) + "\t StrDev: " + str(
        np.sqrt(np.var(np.asarray(acc_split)))) + "\n")


def perform_validation_acc(log_file, split_result, n_epoch, test_epoch):
    log_file.write("\n\nValidation based on ACC value\n")
    print("Validation based on ACC value")
    split_valid_epoch = []
    acc_split = []
    # validation on loss

    for i, split in enumerate(split_result):
        valid_res = split[2]
        # get loss column
        try:
            acc_list = valid_res[0:n_epoch, 3]
        except IndexError:
            acc_list = valid_res[3]

        # get the index (epoch) of min loss
        val_epoch_list = np.argwhere(
            acc_list == np.amax(acc_list))  # get the list of the min (useful if there are more epoch with the same acc

        # if there is many valid epoch with the same value it will take the best one based on loss
        try:
            loss_list = valid_res[0:n_epoch, 2]
        except IndexError:
            loss_list = valid_res[2]
        try:
            val_epoch = val_epoch_list[np.argmin(loss_list[val_epoch_list])]
        except IndexError:
            val_epoch = 0

        # if there is many valid epoch with the same value it will take the first one
        # val_epoch = val_epoch_list[0]

        # get accuracy in the test ste of da validated epoch
        test_split = split[1]
        # acc col i the row 3 of the table
        try:
            acc = test_split[val_epoch, 3]
        except IndexError:
            acc = test_split[3]

        split_valid_epoch.append(val_epoch)

        print("split: ", str(i), " epoch: ", str(val_epoch * test_epoch), " Acc: ", acc)
        log_file.write("split: " + str(i) + " epoch: " + str(val_epoch * test_epoch) + " Acc: " + str(acc) + "\n")
        acc_split.append(acc)

    print(np.mean(np.asarray(acc_split)))

    log_file.write("\nAVG Acc: " + str(np.mean(np.asarray(acc_split))) + "\t StrDev: " + str(
        np.sqrt(np.var(np.asarray(acc_split)))) + "\n")


def plot_graph(test_name, result_folder, n_epoch, test_epoch, split_result, n_split, plot_loss=True, plot_acc=True):
    # evenly sampled time at 200ms intervals
    n_step = int(n_epoch / test_epoch)
    train_acc_avg = np.zeros(n_step)
    test_acc_avg = np.zeros(n_step)
    valid_acc_avg = np.zeros(n_step)

    train_loss_avg = np.zeros(n_step)
    test_loss_avg = np.zeros(n_step)
    valid_loss_avg = np.zeros(n_step)

    for i, (split_train, split_test, split_valid) in enumerate(split_result):
        # print(i)
        try:
            actual_n_epoch = split_train[:, :].shape[0]
        except IndexError:
            actual_n_epoch = 0

        if actual_n_epoch < n_epoch:
            added_epoch_indexs = np.asarray(range(actual_n_epoch, n_epoch)).reshape(-1, 1)

            added_stat_value_train = np.stack([split_train[-1, 1:6]] * (n_epoch - actual_n_epoch), axis=0)
            padding_vec_train = np.concatenate((added_epoch_indexs, added_stat_value_train), axis=1)

            split_train = np.concatenate((split_train, padding_vec_train), axis=0)

            added_stat_value_test = np.stack([split_test[-1, 1:6]] * (n_epoch - actual_n_epoch), axis=0)
            padding_vec_test = np.concatenate((added_epoch_indexs, added_stat_value_test), axis=1)

            split_test = np.concatenate((split_test, padding_vec_test), axis=0)

            added_stat_value_valid = np.stack([split_valid[-1, 1:6]] * (n_epoch - actual_n_epoch), axis=0)
            padding_vec_valid = np.concatenate((added_epoch_indexs, added_stat_value_valid), axis=1)

            split_valid = np.concatenate((split_valid, padding_vec_valid), axis=0)

        train_acc_avg += split_train[0:n_epoch, 3]
        train_loss_avg += split_train[0:n_epoch, 2]

        test_acc_avg += split_test[0:n_epoch:, 3]
        test_loss_avg += split_test[0:n_epoch, 2]

        valid_acc_avg += split_valid[0:n_epoch, 3]
        valid_loss_avg += split_valid[0:n_epoch, 2]

    if plot_acc:
        plt.plot(split_train[0:n_epoch, 0], train_acc_avg / n_split, 'r--', label='Train')
        plt.plot(split_test[0:n_epoch, 0], test_acc_avg / n_split, 'b--', label="Test")
        plt.plot(split_valid[0:n_epoch, 0], valid_acc_avg / n_split, 'g--', label="Valid")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("\n".join(wrap(test_name.replace("_", " "))))
        plt.legend()
        plt.savefig(os.path.join(result_folder, test_name + "_ACC.png"))
        # plt.show()
        plt.clf()

    if plot_loss:
        # plt.ylim(bottom=0, top=2)
        plt.plot(split_train[0:n_epoch, 0], train_loss_avg / n_split, 'r--', label='Train')
        plt.plot(split_test[0:n_epoch, 0], test_loss_avg / n_split, 'b--', label='Test')
        plt.plot(split_valid[0:n_epoch, 0], valid_loss_avg / n_split, 'g--', label='Valid')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("\n".join(wrap(test_name.replace("_", " "))))
        plt.legend()
        plt.savefig(os.path.join(result_folder, test_name + "_LOSS.png"))
        plt.clf()
        # plt.show()
        # pass


def plot_SOM_graph(test_name, result_folder, n_epoch, test_epoch, split_result, n_split):
    # evenly sampled time at 200ms intervals
    n_step = int(n_epoch / test_epoch)
    train_acc_avg = np.zeros(n_step)

    train_loss_avg = np.zeros(n_step)

    for i, (split_train, split_test, split_valid) in enumerate(split_result):
        # print(i)
        actual_n_epoch = split_train[:, :].shape[0]

        if actual_n_epoch < n_epoch:
            added_epoch_indexs = np.asarray(range(actual_n_epoch, n_epoch)).reshape(-1, 1)

            added_stat_value_train = np.stack([split_train[-1, 1:6]] * (n_epoch - actual_n_epoch), axis=0)
            padding_vec_train = np.concatenate((added_epoch_indexs, added_stat_value_train), axis=1)

            split_train = np.concatenate((split_train, padding_vec_train), axis=0)

        train_loss_avg += split_train[0:n_epoch, 5]  # /3#TODO: togliere il /3 per i cas in cui non si usa la shared som

    # plt.ylim(bottom=0, top=2)
    plt.plot(split_train[0:n_epoch, 0], train_loss_avg / n_split, 'r--', label='Train')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("\n".join(wrap(test_name.replace("_", " "))))
    plt.legend()
    plt.savefig(os.path.join(result_folder, test_name + "_LOSS.pdf"))
    plt.show()
    plt.clf()

    # pass
