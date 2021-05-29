from textwrap import wrap

import numpy as np
from pathlib import Path
import os
import matplotlib.pyplot as plt
from results_parser_utils import load_file, perform_validation_acc, plot_graph
from utils import longname

def compute_best_validation_acc(log_file, split_result, n_epoch, test_epoch):
    log_file.write("\n\nValidation based on ACC value\n")
    print("Validation based on ACC value")
    split_valid_epoch = []
    acc_split = []
    # validation on loss

    for i, split in enumerate(split_result):
        valid_res = split[2]
        # get loss column
        acc_list = valid_res[0:n_epoch,3]

        # get the index (epoch) of min loss
        val_epoch_list = np.argwhere(acc_list == np.amax(acc_list)) #get the list of the min (usefull if there are more epoch with the same acc


        # if there is many valid epoch with thesame value'ill take the best one base on loss
        loss_list = valid_res[0:n_epoch, 2]
        val_epoch = val_epoch_list[np.argmin(loss_list[val_epoch_list])]

        # if there is many valid epoch with thesame value'ill take the first one

        # acc col i the row 3 of the table
        acc = valid_res[val_epoch, 3]

        split_valid_epoch.append(val_epoch)

        print("split: ", str(i), " epoch: ", str(val_epoch * test_epoch), " Acc: ", acc)
        log_file.write("split: " + str(i) + " epoch: " + str(val_epoch * test_epoch) + " Acc: " + str(acc) + "\n")
        acc_split.append(acc)

    print(np.mean(np.asarray(acc_split)))

    log_file.write("\nAVG Acc: "+str(np.mean(np.asarray(acc_split)))+"\t StrDev: "+str(np.sqrt(np.var(np.asarray(acc_split))))+"\n")
    return np.mean(np.asarray(acc_split))

if __name__ == '__main__':

    grid_folder = "C:/Users/pfrazzetto00/PycharmProjects/SOMBasedGraphAggregation/experiments/MUTAG/Analysis"
    grid_folder = longname(Path(grid_folder))

    n_split = 3#10

    n_epoch = 500
    test_epoch = 1
    validation_test_results=[]
    test_list=os.listdir(grid_folder)
    if ".directory" in test_list:
        test_list.remove(".directory")
    if "map" in test_list:
        test_list.remove("map")

    for test_name in test_list:
        print(test_name)

        test_folder = grid_folder.joinpath(test_name)#os.path.join(grid_folder,test_name)

        validation_folder = test_folder.joinpath('validation_result')#os.path.join(test_folder, "validation_result")
        validation_folder.mkdir(parents=True, exist_ok=True)
        #if not os.path.exists(validation_folder):
        #    os.makedirs(validation_folder)

        log_file = open(os.path.join(validation_folder, "GRID_Search_" + test_name + ".log"), 'a')
        log_file.write("\n")

        split_result = load_file(test_folder, test_name, n_split)
        res_training = compute_best_validation_acc(log_file,split_result,n_epoch,test_epoch)

        split_result = load_file(test_folder, test_name, n_split, "_fine_tuning_")
        res_fine_tuning = compute_best_validation_acc(log_file, split_result, n_epoch, test_epoch)


        if res_training >= res_fine_tuning:
            validation_test_results.append(res_training)
        else:
            validation_test_results.append(res_fine_tuning)





    best_result_index =np.argmax(np.asarray(validation_test_results))

    best_test= test_list[best_result_index]
    print("best test: ", best_test)


    #do the same process but by considering the output after fine tuning


    #validation of the best test

    best_test_folder = os.path.join(grid_folder, best_test)

    validation_folder = os.path.join(best_test_folder, "validation_result")
    if not os.path.exists(validation_folder):
        os.makedirs(validation_folder)

    split_result = load_file(best_test_folder, best_test, n_split)

    log_file = open(os.path.join(validation_folder, "LOG_" + best_test + ".log"), 'a')
    log_file.write("\n")

    perform_validation_acc(log_file, split_result, n_epoch, test_epoch)
    plot_graph(best_test, validation_folder, n_epoch, test_epoch, split_result, n_split)

    # valid result of the first GNN

    log_file_pre_gnn = open(os.path.join(validation_folder, "LOG_PRE_GNN_" + best_test + ".log"), 'a')
    split_result = load_file(best_test_folder, best_test, n_split, "_conv_part_")

    perform_validation_acc(log_file_pre_gnn, split_result, n_epoch, test_epoch)

    plot_graph("PRE_GNN_" + best_test, validation_folder, n_epoch, test_epoch, split_result, n_split)

    # valid fine tuning results

    log_file_pre_gnn = open(os.path.join(validation_folder, "LOG_FINE_TUNE_GNN_" + best_test + ".log"), 'a')
    split_result = load_file(best_test_folder, best_test, n_split, "_fine_tuning_")

    perform_validation_acc(log_file_pre_gnn, split_result, n_epoch, test_epoch)
    plot_graph("FINE_TUNE_GNN_" + best_test, validation_folder, n_epoch, test_epoch, split_result, n_split)








