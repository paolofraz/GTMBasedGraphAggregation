import torch
import os
import sys
import datetime
import time
from pathlib import Path
from utils.utils import longname
from numpy.linalg import svd
import matplotlib.pyplot as plt
import numpy as np
# plt.rcParams["figure.figsize"] = (20,14)
# plt.rcParams["image.cmap"] = 'bwr'
import gc

predict_fn = lambda output: output.max(1, keepdim=True)[1].detach().cpu()  # Takes the MAX of each row

_GTM_LAYERS = 3

def prepare_log_files(test_name, log_dir):
    '''
    create a log file where test information and results will be saved
    :param test_name: name of the test
    :param log_dir: directory where the log files will be created
    :return: return a log file for each sub set (training, test, validation)
    '''
    train_log = open(os.path.join(log_dir, (test_name + "_train")), 'w+')
    test_log = open(os.path.join(log_dir, (test_name + "_test")), 'w+')
    valid_log = open(os.path.join(log_dir, (test_name + "_valid")), 'w+')

    for f in (train_log, test_log, valid_log):
        f.write("test_name: %s \n" % test_name)
        f.write(str(datetime.datetime.now()) + '\n')
        f.write("#epoch \t split \t loss \t acc \t avg_epoch_time \t avg_epoch_cost \n")

    return train_log, test_log, valid_log


class modelImplementation_GraphBinClassifier(torch.nn.Module):
    """
    General implementation of training routine for a GNN that perform graph classification
    """

    def __init__(self, model, criterion, device='cpu'):
        super(modelImplementation_GraphBinClassifier, self).__init__()

        self.model = model
        self.criterion = criterion
        self.device = device

    def stop_grad(self, phase):
        for name, param in self.model.named_parameters():
            # TODO is it here so that it never keeps grads for SOM/GTMs?
            if phase == "conv":  # train only first conv part
                if "gtm" in name or "out" in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            elif phase == "readout":  # train only readout layer
                if "out" not in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            elif phase == "fine_tuning":  # retrain everything but GTMs
                if "gtm" in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True

    def set_optimizer(self, lr_conv, lr_gtm, lr_readout, lr_fine_tuning, weight_decay=0):
        """
        Set the optimizer for the training phase as AdamW (w/ weight decay) and different learning rates for each phase
        :param weight_decay: amount of weight decay to apply during training
        """
        # ------------------#
        train_out_stage_params = []
        train_conv_stage_params = []
        fine_tune_stage_par = []
        for name, param in self.model.named_parameters():
            if not "gtm" in name:  # exclude GTMs
                if "out" in name:
                    train_out_stage_params.append(param)
                else:
                    train_conv_stage_params.append(param)

                fine_tune_stage_par.append(param)  # keep out and readout for fine tuning

        self.conv_optimizer = torch.optim.AdamW(train_conv_stage_params, lr=lr_conv, weight_decay=weight_decay)
        self.out_optimizer = torch.optim.AdamW(train_out_stage_params, lr=lr_readout, weight_decay=weight_decay)
        self.fine_tune_optimizer = torch.optim.AdamW(fine_tune_stage_par, lr=lr_fine_tuning, weight_decay=weight_decay)
        self.lr_gtm = lr_gtm

    def train_test_model(self, split_id, loader_train, loader_test, loader_valid, n_epochs_conv, n_epochs_readout,
                         n_epochs_fine_tuning, n_epochs_gtm, test_epoch, early_stopping_threshold,
                         early_stopping_threshold_gtm, max_n_epochs_without_improvements, test_name="", log_path=".",
                         verbose=0):
        """
        Method that performs the training of a given model, and tests it after a given number of epochs.
        :param split_id: numeric id of the considered split (use to identify the current split in a cross-validation setting)
        :param loader_train: loader of the training set
        :param loader_test: loader of the test set
        :param loader_valid: load of the validation set
        :param n_epochs: number of training epochs
        :param test_epoch: the test phase is performed every test_epoch epochs # TODO can I increase it to speed it up? Si, però test a più epoche abbassa i risultati della validazione
        :param test_name: name of the test
        :param log_path: past where the logs file will be saved
        """

        print(" # CONV_PART TRAIN # ")
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()
        self.stop_grad("conv")
        self.training_phase(n_epochs=n_epochs_conv,
                            optimizer=self.conv_optimizer,
                            loader_train=loader_train,
                            loader_test=loader_test,
                            loader_valid=loader_train,#loader_valid, # TODO SISTEMA!
                            test_epoch=test_epoch,
                            log_file_name="_conv_part_" + test_name,
                            split_id=split_id,
                            log_path=log_path,
                            use_conv_out=True,
                            test_name=test_name,
                            early_stopping_threshold=early_stopping_threshold,
                            max_n_epochs_without_improvements=max_n_epochs_without_improvements)

        print(" # GTM TRAIN # ")
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()
        # load best model from previous step
        # TODO stai attento a ES per GTM e variazione della loss con validazione, peggioramento a livello di decimi su migliia
        self.load_model(test_name)

        # --- GTM PCA Initialization ---
        self.model.eval()
        with torch.no_grad():
            h_dataset = torch.empty((0, self.model.out_channels * 3), device=self.device)
            for batch in loader_train:
                data = batch.to(self.device)
                _, h_conv, _ = self.model(data, gtm_train=True)
                h_dataset = torch.cat((h_conv, h_dataset), 0)

            self.model.gtm1.initialize(h_dataset[:, 0:self.model.out_channels])
            self.model.gtm2.initialize(h_dataset[:, self.model.out_channels:self.model.out_channels*2])
            self.model.gtm3.initialize(h_dataset[:, self.model.out_channels * 2:])

            del h_dataset, h_conv
            print("------ GTM PCA Initialized ------")

        train_log, test_log, valid_log = prepare_log_files("_gtm_part_" + test_name + "--split-" + str(split_id),
                                                           log_path)

        train_loss, n_samples = 0.0, 0
        epoch_time_sum = 0
        best_epoch_gtm = 0
        best_gtm_loss_so_far = -1
        gtm_n_epochs_without_improvements = 0

        # Plots
        self.verbose = 1
        if self.verbose == 1:
            h_conv_1_all = torch.empty((0, self.model.out_channels), device=self.device)
            h_conv_2_all = torch.empty((0, self.model.out_channels), device=self.device)
            h_conv_3_all = torch.empty((0, self.model.out_channels), device=self.device)
            gtm_losses = np.empty((0, 3))
            y_all = np.array([], )

        for epoch in range(n_epochs_gtm):
            self.model.train()

            epoch_start_time = time.time()
            gtm_losses_batch = np.empty((0, 3))
            first_idx = 0
            second_idx = 0
            for batch in loader_train:

                data = batch.to(self.device)
                second_idx = first_idx + data.x.shape[0]

                if self.verbose == 1 and split_id == 0:
                    _, reps = torch.unique(data.batch.data, sorted=True, return_counts=True)
                    y_all = np.append(y_all, np.repeat(data.y.detach().cpu().numpy(), reps.cpu().numpy()))

                _, h_conv, _ = self.model(data, gtm_train=True)  # h, h_conv, gnn_out

                h_conv_1 = h_conv[:, 0:self.model.out_channels]  # is equal to x1
                h_conv_2 = h_conv[:,
                           self.model.out_channels:self.model.out_channels + self.model.out_channels]  # x2
                h_conv_3 = h_conv[:,self.model.out_channels * 2:]  # x3

                gtm_1_loss = self.model.gtm1.train_aggregator(h_conv_1, epoch, n_epochs_gtm, self.lr_gtm, first_idx, second_idx)
                gtm_2_loss = self.model.gtm2.train_aggregator(h_conv_2, epoch, n_epochs_gtm, self.lr_gtm, first_idx, second_idx)
                gtm_3_loss = self.model.gtm3.train_aggregator(h_conv_3, epoch, n_epochs_gtm, self.lr_gtm, first_idx, second_idx)

                train_loss += gtm_1_loss + gtm_2_loss + gtm_3_loss
                n_samples += len(batch) # TODO sitemare early stopping per incremental learning/complete data likelihood
                first_idx = second_idx

                if self.verbose == 1 and split_id == 0:
                    h_conv_1_all = torch.cat((h_conv_1_all, h_conv_1), 0)
                    h_conv_2_all = torch.cat((h_conv_2_all, h_conv_2), 0)
                    h_conv_3_all = torch.cat((h_conv_3_all, h_conv_3), 0)
                    gtm_losses_batch = np.append(gtm_losses_batch, [[gtm_1_loss, gtm_2_loss, gtm_3_loss]], axis=0)

            epoch_time = time.time() - epoch_start_time
            epoch_time_sum += epoch_time

            if self.verbose == 1 and split_id == 0: gtm_losses = np.append(gtm_losses,[gtm_losses_batch.mean(axis=0)], axis=0)

            if epoch == 0 and self.verbose == 1 and split_id == 0:
                fig, axs = plt.subplots(2, 3, figsize=(20, 14))
                fig.suptitle(test_name)
                colormap = np.array(['navy', 'firebrick'])
                y_all = y_all.astype(int)
                points = h_conv_1_all.matmul(torch.linalg.pinv(self.model.gtm1.W)).matmul(self.model.gtm1.matM)
                axs[0, 0].scatter(points[:, 0].cpu().detach().numpy(), points[:, 1].cpu().detach().numpy(), c=colormap[y_all])
                axs[0, 0].text(0.01, 0.01, "Beta = "+"{:.4f}".format(self.model.gtm1.beta.item()), verticalalignment='bottom', horizontalalignment='left', transform=axs[0, 0].transAxes)
                axs[0, 0].set_title("Pre GTM Training - 1st layer")
                points = h_conv_2_all.matmul(torch.linalg.pinv(self.model.gtm2.W)).matmul(self.model.gtm2.matM)
                axs[0, 1].scatter(points[:, 0].cpu().detach().numpy(), points[:, 1].cpu().detach().numpy(), c=colormap[y_all])
                axs[0, 1].text(0.01, 0.01, "Beta = " + "{:.4f}".format(self.model.gtm2.beta.item()),
                               verticalalignment='bottom', horizontalalignment='left', transform=axs[0, 1].transAxes)
                axs[0, 1].set_title("Pre GTM Training - 2nd layer")
                points = h_conv_3_all.matmul(torch.linalg.pinv(self.model.gtm3.W)).matmul(self.model.gtm3.matM)
                axs[0, 2].scatter(points[:, 0].cpu().detach().numpy(), points[:, 1].cpu().detach().numpy(), c=colormap[y_all])
                axs[0, 2].text(0.01, 0.01, "Beta = " + "{:.4f}".format(self.model.gtm3.beta.item()),
                               verticalalignment='bottom', horizontalalignment='left', transform=axs[0, 2].transAxes)
                axs[0, 2].set_title("Pre GTM Training - 3rd layer")

                fig2, axs2 = plt.subplots(2,3, figsize = (10,7))
                fig2.suptitle(test_name)
                im = axs2[0,0].imshow(
                    self.model.gtm1.pdf_data_space(h_conv_1_all[42, :]))
                axs2[0,0].set_title("Pre GTM Training - p(t|x,W)")
                plt.colorbar(im, ax=axs2[0,0])
                im = axs2[0,1].imshow(
                    self.model.gtm2.pdf_data_space(h_conv_2_all[42, :]))
                axs2[0,1].set_title("Pre GTM Training - p(t|x,W)")
                plt.colorbar(im, ax=axs2[0,1])
                im = axs2[0,2].imshow(
                    self.model.gtm3.pdf_data_space(h_conv_3_all[42, :]))
                axs2[0,2].set_title("Pre GTM Training - p(t|x,W)")
                plt.colorbar(im, ax=axs2[0,2])

            if epoch % test_epoch == 0:
                print("split : ", split_id, " -- epoch : ", epoch, " -- loss: ", train_loss / n_samples)

                train_log.write(
                    "{:d}\t{:d}\t{:.8f}\t{:.8f}\t{:.8f}\t{:.8f}\n".format(
                        epoch,
                        split_id,
                        0,  # loss_train_set,
                        0,  # acc_train_set,
                        epoch_time_sum / test_epoch,
                        train_loss / n_samples))

                train_log.flush()

                # early_stopping
                if (train_loss / n_samples) < best_gtm_loss_so_far or best_gtm_loss_so_far == -1:
                    best_gtm_loss_so_far = train_loss / n_samples
                    gtm_n_epochs_without_improvements = 0
                    best_epoch_gtm = epoch
                    print("--ES--")
                    print("save_new_best_model, with loss:", best_gtm_loss_so_far)
                    print("------")

                    self.save_model(test_name)

                elif (train_loss / n_samples) >= best_gtm_loss_so_far + early_stopping_threshold_gtm:
                    gtm_n_epochs_without_improvements += 1
                else:
                    gtm_n_epochs_without_improvements = 0

                if gtm_n_epochs_without_improvements >= max_n_epochs_without_improvements:
                    print("___Early Stopping at epoch ", best_epoch_gtm, "____")
                    break

                train_loss, n_samples = 0, 0
                epoch_time_sum = 0

        if self.verbose == 1 and split_id == 0:
            points = h_conv_1_all.matmul(torch.linalg.pinv(self.model.gtm1.W)).matmul(self.model.gtm1.matM)
            axs[1, 0].scatter(points[:, 0].cpu().detach().numpy(), points[:, 1].cpu().detach().numpy(), c=colormap[y_all])
            axs[1, 0].text(0.01, 0.01, "Beta = " + "{:.4f}".format(self.model.gtm1.beta.item()),
                           verticalalignment='bottom', horizontalalignment='left', transform=axs[1, 0].transAxes)
            axs[1, 0].set_title("Post GTM Training - 1st layer")
            points = h_conv_2_all.matmul(torch.linalg.pinv(self.model.gtm2.W)).matmul(self.model.gtm2.matM)
            axs[1, 1].scatter(points[:, 0].cpu().detach().numpy(), points[:, 1].cpu().detach().numpy(), c=colormap[y_all])
            axs[1, 1].text(0.01, 0.01, "Beta = " + "{:.4f}".format(self.model.gtm2.beta.item()),
                           verticalalignment='bottom', horizontalalignment='left', transform=axs[1, 1].transAxes)
            axs[1, 1].set_title("Post GTM Training - 2nd layer")
            points = h_conv_3_all.matmul(torch.linalg.pinv(self.model.gtm3.W)).matmul(self.model.gtm3.matM)
            axs[1, 2].scatter(points[:, 0].cpu().detach().numpy(), points[:, 1].cpu().detach().numpy(), c=colormap[y_all])
            axs[1, 2].text(0.01, 0.01, "Beta = " + "{:.4f}".format(self.model.gtm3.beta.item()),
                           verticalalignment='bottom', horizontalalignment='left', transform=axs[1, 2].transAxes)
            axs[1, 2].set_title("Post GTM Training - 3rd layer")

            im2 = axs2[1,0].imshow(
                self.model.gtm1.pdf_data_space(h_conv_1_all[42, :]))
            axs2[1,0].set_title("Post GTM Training - p(t|x,W)")
            plt.colorbar(im2, ax=axs2[1,0])
            im = axs2[1, 1].imshow(
                self.model.gtm2.pdf_data_space(h_conv_2_all[42, :]))
            axs2[1, 1].set_title("Post GTM Training - p(t|x,W)")
            plt.colorbar(im, ax=axs2[1, 1])
            im = axs2[1, 2].imshow(
                self.model.gtm3.pdf_data_space(h_conv_3_all[42, :]))
            axs2[1, 2].set_title("Post GTM Training - p(t|x,W)")
            plt.colorbar(im, ax=axs2[1, 2])

            fig3, axs3 = plt.subplots()
            axs3.plot(gtm_losses, label=['GTM1', "GTM2", "GTM3"])
            axs3.set_title("GTM losses - lr = " + str(self.lr_gtm))
            axs3.legend()
            plt.show()
            curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            fig.savefig(log_path / Path('Points_projections_' + curr_time + '.png'))
            fig2.savefig(log_path / Path('Prob_dist_' + curr_time + '.png'))
            fig3.savefig(log_path / Path('Losses_' + curr_time + '.png'))

            del h_conv_1_all, h_conv_2_all, h_conv_3_all, y_all, gtm_losses, points

        print(" # READ_OUT TRAIN # ")
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()
        # load best model from previous step
        self.load_model(test_name)
        self.stop_grad("readout")
        self.training_phase(n_epochs=n_epochs_readout,
                            optimizer=self.out_optimizer,
                            loader_train=loader_train,
                            loader_test=loader_test,
                            loader_valid=loader_valid,
                            test_epoch=test_epoch,
                            log_file_name=test_name,
                            split_id=split_id,
                            log_path=log_path,
                            use_conv_out=False,
                            test_name=test_name,
                            early_stopping_threshold=early_stopping_threshold,
                            max_n_epochs_without_improvements=max_n_epochs_without_improvements
                            )

        print(" # FINE TUNING # ")
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()
        # load best model from previous step
        self.load_model(test_name)
        self.stop_grad("fine_tuning")
        self.training_phase(n_epochs=n_epochs_fine_tuning,
                            optimizer=self.fine_tune_optimizer,
                            loader_train=loader_train,
                            loader_test=loader_test,
                            loader_valid=loader_valid,
                            test_epoch=test_epoch,
                            log_file_name="_fine_tuning_" + test_name,
                            split_id=split_id,
                            log_path=log_path,
                            use_conv_out=False,
                            test_name=test_name,
                            early_stopping_threshold=early_stopping_threshold,
                            max_n_epochs_without_improvements=max_n_epochs_without_improvements
                            )
        # os.remove('./' + test_name + '.pt')
        # longname(Path(os.path.join(os.getcwd(), test_name + '.pt'))).unlink()

    def training_phase(self, n_epochs, optimizer, loader_train, loader_test, loader_valid, test_epoch, log_file_name,
                       split_id, log_path, use_conv_out, test_name, early_stopping_threshold,
                       max_n_epochs_without_improvements):

        train_log, test_log, valid_log = prepare_log_files(log_file_name + "--split-" + str(split_id), log_path)

        train_loss, n_samples = 0.0, 0

        epoch_time_sum = 0

        best_epoch = 0
        best_loss_so_far = -1
        n_epochs_without_improvements = 0

        for epoch in range(n_epochs):
            self.model.train()

            epoch_start_time = time.time()
            for batch in loader_train:

                data = batch.to(self.device)
                optimizer.zero_grad() # ! THIS IS IMPORTANT
                if use_conv_out:
                    _, _, out = self.model(data, conv_train=True)  # h, h_conv, gnn_out
                else:
                    out, _, _ = self.model(data)

                loss = self.criterion(out, data.y) # TODO The input given through a forward call is expected to contain log-probabilities of each class also in GTM?
                # TODO questa è loss per ogni elemento del batch

                loss.backward()
                optimizer.step()

                train_loss += loss.item() * len(out) # TODO why * len(out)? Is it for averaging over batches? Ho una loss per grafo? dim(out)? numbrafi x targhet (=2)
                n_samples += len(out)

            epoch_time = time.time() - epoch_start_time
            epoch_time_sum += epoch_time

            if epoch % test_epoch == 0:

                if use_conv_out:
                    acc_train_set, correct_train_set, n_samples_train_set, loss_train_set = self.eval_model(
                        loader_train,
                        "conv_out")
                    acc_test_set, correct_test_set, n_samples_test_set, loss_test_set = self.eval_model(loader_test,
                                                                                                        "conv_out")
                    acc_valid_set, correct_valid_set, n_samples_valid_set, loss_valid_set = self.eval_model(
                        loader_valid,
                        "conv_out")
                else:
                    acc_train_set, correct_train_set, n_samples_train_set, loss_train_set = self.eval_model(
                        loader_train)  # Qui è la valutazione esatta alla fine, non durante epoch
                    acc_test_set, correct_test_set, n_samples_test_set, loss_test_set = self.eval_model(loader_test)
                    acc_valid_set, correct_valid_set, n_samples_valid_set, loss_valid_set = self.eval_model(
                        loader_valid)

                print("epoch : ", epoch, " -- loss: ", train_loss / n_samples, "--- valid_loss: ", loss_valid_set)

                print("split : ", split_id, " -- training acc : ",
                      (acc_train_set, correct_train_set, n_samples_train_set), " -- test_acc : ",
                      (acc_test_set, correct_test_set, n_samples_test_set),
                      " -- valid_acc : ", (acc_valid_set, correct_valid_set, n_samples_valid_set))
                print("------")

                train_log.write(
                    "{:d}\t{:d}\t{:.8f}\t{:.8f}\t{:.8f}\t{:.8f}\n".format(
                        epoch,
                        split_id,
                        loss_train_set,
                        acc_train_set,
                        epoch_time_sum / test_epoch,
                        train_loss / n_samples))

                train_log.flush()

                test_log.write(
                    "{:d}\t{:d}\t{:.8f}\t{:.8f}\t{:.8f}\t{:.8f}\n".format(
                        epoch,
                        split_id,
                        loss_test_set,
                        acc_test_set,
                        epoch_time_sum / test_epoch,
                        train_loss / n_samples))

                test_log.flush()

                valid_log.write(
                    "{:d}\t{:d}\t{:.8f}\t{:.8f}\t{:.8f}\t{:.8f}\n".format(
                        epoch,
                        split_id,
                        loss_valid_set,
                        acc_valid_set,
                        epoch_time_sum / test_epoch,
                        train_loss / n_samples))

                valid_log.flush()

                # -- VALIDATION --
                if loss_valid_set < best_loss_so_far or best_loss_so_far == -1:
                    best_loss_so_far = loss_valid_set
                    n_epochs_without_improvements = 0
                    best_epoch = epoch
                    print("--ES--")
                    print("save_new_best_model, with loss:", best_loss_so_far)
                    print("------")
                    self.save_model(test_name)

                elif loss_valid_set >= best_loss_so_far + early_stopping_threshold:
                    n_epochs_without_improvements += 1
                else:
                    n_epochs_without_improvements = 0

                if n_epochs_without_improvements >= max_n_epochs_without_improvements:  # questa è la "pazienza"
                    print("___Early Stopping at epoch ", best_epoch, "____")
                    break

                train_loss, n_samples = 0, 0
                epoch_time_sum = 0

    def eval_model(self, loader, sub_model="read_out"):
        """
        Function that compute the accuracy of the model given a dataset
        :param loader: dataset used to evaluate the model performance
        :return: accuracy, number samples classified correctly, total number of samples, average loss
        """
        self.model.eval()
        correct = 0
        n_samples = 0
        loss = 0.0
        for batch in loader:
            data = batch.to(self.device)

            if sub_model == "conv_out":
                _, _, model_out = self.model(data, conv_train=True)
            else:
                model_out, _, _ = self.model(data)

            pred = predict_fn(model_out)
            n_samples += len(model_out)
            correct += pred.eq(data.y.detach().cpu().view_as(pred)).sum().item()
            loss += self.criterion(model_out, data.y).item() * len(model_out)

        acc = 100. * correct / n_samples
        return acc, correct, n_samples, loss / n_samples

    def save_model(self, test_name):
        if sys.platform == 'win32':
            torch.save(self.model.state_dict(), longname(Path(os.path.join(os.getcwd(), test_name + '.pt'))))
        elif sys.platform == 'linux':
            torch.save(self.model.state_dict(), os.path.join(os.getcwd(), "Thesis_GTM/experiments/TORUN", test_name + '.pt'))
            os.chmod(os.path.join(os.getcwd(), "Thesis_GTM/experiments/TORUN", test_name + '.pt'), 0o755)

    def load_model(self, test_name):
        if sys.platform == 'win32':
            self.model.load_state_dict(torch.load(longname(Path(os.path.join(os.getcwd(), test_name + '.pt')))))
        elif sys.platform == 'linux':
            self.model.load_state_dict(torch.load(os.path.join(os.getcwd(), "Thesis_GTM/experiments/TORUN", test_name + '.pt')))
