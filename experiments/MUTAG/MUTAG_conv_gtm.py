import os
import sys
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

import torch

from model.GNN_Conv_GTM import GNN_Conv_GTM
from impl.binGraphClassifier_GTM_Layer import modelImplementation_GraphBinClassifier
from utils.utils import printParOnFile, longname
from data_reader.cross_validation_reader import getcross_validation_split

if __name__ == '__main__':

    n_epochs_conv = 500
    n_epochs_readout = 500
    n_epochs_fine_tuning = 500
    n_classes = 2
    dataset_path = '~/Dataset/'
    dataset_name = 'MUTAG'
    n_folds = 3 #10
    test_epoch = 1

    n_units = 30
    lr_conv = 0.0005
    lr_readout = 0.0005
    lr_fine_tuning = 0.0001
    weight_decay = 5e-4
    drop_prob = 0.5
    batch_size = 16 #32

    gtm_epoch = 500
    gtm_grids_dim = (15, 15)
    gtm_lr = 0.005

    # early stopping par
    max_n_epochs_without_improvements = 25
    early_stopping_threshold = 0.075
    early_stopping_threshold_gtm = 0.02

    test_name = "test_GNN_Conv_GTM"

    test_name = test_name + \
                "_data-" + dataset_name + \
                "_nFold-" + str(n_folds) + \
                "_lr_conv-" + str(lr_conv) + \
                "_lr_gtm-" + str(gtm_lr) + \
                "_lr_readout-" + str(lr_readout) + \
                "_lr_fine_tuning-" + str(lr_fine_tuning) + \
                "_drop_prob-" + str(drop_prob) + \
                "_weight-decay-" + str(weight_decay) + \
                "_batchSize-" + str(batch_size) + \
                "_nHidden-" + str(n_units) + \
                "_gtm_grid-" + str(gtm_grids_dim[0]) + "_" + str(gtm_grids_dim[1]) + \
                "_gtm_lr-" + str(gtm_lr)

    training_log_dir = Path.cwd() / "test_log" / test_name
    training_log_dir = longname(training_log_dir)
    training_log_dir.mkdir(parents=True, exist_ok=True)

    printParOnFile(test_name=test_name, log_dir=training_log_dir,
                   par_list={"dataset_name": dataset_name,
                             "n_fold": n_folds,
                             "learning_rate_conv": lr_conv,
                             "learning_rate_gtm": gtm_lr,
                             "learning_rate_read_out": lr_readout,
                             "learning_rate_fine_tuning": lr_fine_tuning,
                             "drop_prob": drop_prob,
                             "weight_decay": weight_decay,
                             "batch_size": batch_size,
                             "n_hidden": n_units,
                             "gtm_grid_dims": gtm_grids_dim,
                             "gtm_lr": gtm_lr,
                             "test_epoch": test_epoch})

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    criterion = torch.nn.NLLLoss()

    dataset_cv_splits = getcross_validation_split(dataset_path, dataset_name, n_folds,
                                                  batch_size)
    for split_id, split in enumerate(dataset_cv_splits):
        loader_train = split[0]
        loader_test = split[1]
        loader_valid = split[2]

        model = GNN_Conv_GTM(loader_train.dataset.num_features, n_units, n_classes, gtm_grids_dim, drop_prob).to(
            device)

        model_impl = modelImplementation_GraphBinClassifier(model=model,
                                                            criterion=criterion,
                                                            device=device).to(device)

        model_impl.set_optimizer(lr_conv=lr_conv,
                                 lr_gtm=gtm_lr,
                                 lr_reaout=lr_readout,
                                 lr_fine_tuning=lr_fine_tuning,
                                 weight_decay=weight_decay)

        model_impl.train_test_model(split_id=split_id,
                                    loader_train=loader_train,
                                    loader_test=loader_test,
                                    loader_valid=loader_valid,
                                    n_epochs_conv=n_epochs_conv,
                                    n_epochs_readout=n_epochs_readout,
                                    n_epochs_fine_tuning=n_epochs_fine_tuning,
                                    n_epochs_gtm=gtm_epoch,
                                    test_epoch=test_epoch,
                                    early_stopping_threshold=early_stopping_threshold,
                                    early_stopping_threshold_gtm=early_stopping_threshold_gtm,
                                    max_n_epochs_without_improvements=max_n_epochs_without_improvements,
                                    test_name=test_name,
                                    log_path=training_log_dir)
