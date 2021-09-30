import torch

import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

torch.pi = torch.acos(torch.zeros(1)).item() * 2
torch.set_default_dtype(torch.float32)  # TODO is it necessary? Replace NaNs manually?
torch.set_printoptions(profile="full")

add_bias = torch.nn.ConstantPad1d((0, 1), 1)  # Add bias (1s) column on the right


def exp_normalize(x):
    """
    Exp-normalize trick: compute exp(x-max(x))
    https://en.wikipedia.org/wiki/LogSumExp
    :param x: 2D torch tensor
    """
    return torch.exp(x - x.max(axis=0).values)


class GTM(nn.Module):
    def __init__(self, input_size, out_size=(20, 18), m=12, sigma=None, gtm_lr=1e-3, method='full_prob',
                 learning='standard', device=None):
        """
        Create a GTM model.
        :param input_size: Dimension of t = (t1, ..., t_D), a.k.a D, e.g. 30
        :param out_size: nodes of a regular grid in latent space x = (x1, ..., x_L), L=2, #latent variables is K in papers
        :param m: We consider y(W,x) = W.phi(x). The elements of phi(x) consist of m^2 fixed radial basis functions.
        :param W: D x m Matrix in paper, here is M x D
        :param phi: M fixed basis functions, K x M (=n_rbf)
        :param sigma: RBF width factor, impacts manifold flexibility. If "None" takes the maximum distance, if 'int' set its value, if 'str' multiply average distance.
        :param gtm_lr: regularization, called lambda in paper
        """
        super(GTM, self).__init__()

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.input_size = input_size  # n_dimensions of input data from GraphConv
        self.out_size = out_size
        self.n_latent_variables = out_size[0] * out_size[1]
        self.n_rfb = m  # Sqrt of the number of RBF centers
        self.method = method
        self.learning = learning
        self.prev_likelihood_ = -float('inf')
        self.gtm_lr = torch.Tensor([gtm_lr]).to(
            self.device)  # regularization, = lambda in paper TODO fix with adaptive value as in SOM
        self.to_be_initialized = True  # True -> PCA, else random

        a = torch.linspace(-1, 1, out_size[0])
        b = torch.linspace(-1, 1, out_size[1])
        self.matX = torch.cartesian_prod(a, b).to(self.device)  # z
        a = torch.linspace(-1 + 1. / m, 1 - 1. / m, m)
        self.matM = nn.Parameter(torch.cartesian_prod(a, a).to(self.device), requires_grad=False)  # rbf locations

        # Initializes radial basis function width using hyperparameter sigma
        if sigma is None or type(sigma) is str:
            # estimate sigma as average min distance among rbfs
            rbfWidth = torch.cdist(self.matM, self.matM, p=2,
                                   compute_mode='donot_use_mm_for_euclid_dist')  # .pow_(2) squared distances [for sigma^2 or sigma]
            self.sigma = torch.mean(torch.min(rbfWidth[torch.nonzero(rbfWidth, as_tuple=True)].view(m * m, m * m - 1),
                                              dim=1).values).item()  # so it excludes distances = 0 with itself -> there is some issue with self distances https://gitmemory.com/issue/pytorch/pytorch/57690/833204080
            if type(sigma) is str:
                self.sigma *= float(sigma)
        else:
            self.sigma = float(sigma)
        print("Sigma = ", self.sigma)

        # Create matrix of RBF functions (plus one dimension to include a term for bias)
        d = torch.cdist(self.matX, self.matM, p=2, compute_mode='donot_use_mm_for_euclid_dist').pow(2)  # squared distances
        # self.phi = nn.Parameter(torch.exp(-d / ( 2. * self.sigma) ).to(self.device), requires_grad=False) # TODO are phi "Parameter"? Prameter = tensor with grad
        self.phi = torch.exp(-d / (2. * self.sigma)).to(self.device)
        self.phi = add_bias(self.phi)

        # Random init or continue initialization later with input data
        if not self.to_be_initialized:
            w = torch.empty(self.n_rfb ** 2 + 1, input_size)
            self.W = nn.init.normal_(w).to(self.device)
            b = torch.empty(1)
            self.beta = nn.init.ones_(b).to(self.device)
            print("------ GTM Random Initialized ------")

    def initialize(self, t):
        """
        PCA initialization.
        W = (n_rbf_centers+1, n_dimensions) initialized to approximate PCA projection.
        Initialize beta to be the larger between:
         (1) the 3rd eigenvalue of the data covariance matrix
         (2) half the average distance between centers of Gaussian components.
        """
        # store mean (also used to center data in PCA) for any further computation
        # self.scale_mean = t.mean(dim=(-2,), keepdim=True)
        # self.scale_std  = t.std(dim=(-2,), unbiased=False, keepdim=True)  # store stdev
        # t -= self.scale_mean
        # t /= self.scale_std
        if self.learning == 'incremental':
            # Initialize R for the whole training set with uniform probability 1/K
            self.R_inc = torch.empty((self.n_latent_variables, t.shape[0]))  # K x N matrix
            self.R_inc = self.R_inc.fill_(1. / self.n_latent_variables)#.to(self.device)
            # self.G_inc = torch.diag(self.R_inc.sum(dim=1)) # K x K matrix
            self.RX_inc = self.R_inc.matmul(t)  # K x D matrix
            self.X_inc = t.to(self.device)
            self.R_inc = self.R_inc.to(self.device)
            self.RX_inc = self.RX_inc.to(self.device)

        if self.to_be_initialized:  # == True
            U, S, V = torch.pca_lowrank(t.cpu(), center=False)
            self.W = torch.linalg.pinv(self.phi).matmul(self.matX).matmul(V.to(self.device)[:, :2].T).to(self.device)
            # self.W = nn.Parameter(self.W, requires_grad=False)

            betainv1 = (S ** 2 / (t.shape[0] - 1))[2].item()  # take L+1 eigenvalue
            inter_dist = torch.cdist(self.phi.matmul(self.W), self.phi.matmul(self.W),
                                     compute_mode='donot_use_mm_for_euclid_dist')
            inter_dist.fill_diagonal_(float('inf'))
            betainv2 = inter_dist.min(dim=0).values.mean().item() / 2.

            self.beta = torch.Tensor([1. / max(betainv1, betainv2)]).double().to(self.device)
            # self.beta = nn.Parameter(self.beta, requires_grad=False)

            print("------ GTM PCA Initialized ------")
            print("Input dataset size: ", t.shape)
            self.to_be_initialized = False

    def responsibility(self, X):
        """
        R = posterior probability (w/o L!) = p(x|t,W,beta) = p(t|x,W,beta) / sum_i p(t|x_i,W,beta)
        X = (batch_nodes, nn_units)
        """
        with torch.no_grad():
            p = exp_normalize(
                -(self.beta / 2.) * torch.cdist(self.phi.matmul(self.W).double(), X.to(self.device).double(), p=2,
                                                compute_mode='donot_use_mm_for_euclid_dist') ** 2)
            return p.div(p.sum(dim=0))

    def pdf_data_space(self, X):
        """
        Probability density function of a single point projected to latent space from data space.
        p(t|x_i,W,beta) = (beta/2pi)^D/2 * exp(-beta/2 * ||y(W,x_i) - t||^2)
        """
        if self.learning == 'incremental':
            R = self.R_inc[:, 42]
        else:
            R = self.responsibility(X.view(1, -1))
        return R.view(self.out_size[0], self.out_size[1]).cpu().detach().numpy()
        # older implementation:  # return torch.exp(-torch.cdist(self.phi.matmul(self.W), t.to(self.device), p=2, compute_mode= 'donot_use_mm_for_euclid_dist') ** 2).view(self.out_size[0], self.out_size[1])

    def log_likelihood(self, X=None):
        """
        p(t|x,W,beta) = (beta/2pi)^D/2 * exp(-beta/2 * ||y(W,x) - t||^2)
        L = sum_n ln(1/K sum_i p(t|x,W,beta))
        """
        with torch.no_grad():
            if self.learning == 'incremental' and X is None:
                try:
                    X = self.X_inc
                except AttributeError:
                    return torch.ones_like(self.beta)
            D = X.shape[1]
            k1 = (self.beta / (2 * torch.pi)).pow(D / 2)  # .add_(1e-8)
            k2 = k1 * exp_normalize(
                (-self.beta.double() / 2) * torch.cdist(self.phi.matmul(self.W).double(), X.to(self.device).double(),
                                                        compute_mode='donot_use_mm_for_euclid_dist',
                                                        p=2) ** 2)  # .add_(1e-8)
            # prior = 1/n_latent_variables
            return torch.log(k2.sum(axis=0) / self.n_latent_variables).sum() / X.shape[0]  # divide per #samples

    def likelihood(self, X=None):
        """
        p(t|x,W,beta) = (beta/2pi)^D/2 * exp(-beta/2 * ||y(W,x) - t||^2)
        ln p = k1 + k2 = D/2 * ln(beta/2pi) + (-beta/2 * ||W.psi - t||^2)
        This leads to the complete-data log likelihood used in EM Eq (10)
        """
        with torch.no_grad():
            if self.learning == 'incremental' and X is None:
                R = self.R_inc
                try:
                    X = self.X_inc
                except AttributeError:
                    return torch.ones_like(self.beta)
            else:
                R = self.responsibility(X)
            D = X.shape[1]
            k1 = torch.log(self.beta / (2 * torch.pi)).mul(D / 2)
            k2 = (torch.cdist(self.phi.matmul(self.W), X.to(self.device), p=2,
                              compute_mode='donot_use_mm_for_euclid_dist') ** 2).mul(-(self.beta.div(2)))
            temp = k2.add(k1) * R
            return temp.sum() / X.shape[0]

    def transform(self, Xt):
        """
        Bayes' theorem is used to invert the transformation from latent space to data space,
        i.e. projecting the data X (t) to the L-lattice (x).
        There are 2 suggested approaches, 'mean' and 'mode', see 2.2 in Bishop (1998)
        """
        assert self.method in ('mean', 'mode', 'full_prob')
        with torch.no_grad():
            if self.method == 'full_prob':
                return self.responsibility(Xt).T.float()  # this is equal to self.R_inc
            elif self.method == 'mean':
                R = self.responsibility(Xt)
                return R.T.matmul(self.matX)
            elif self.method == 'mode':
                return self.matX[self.responsibility(Xt).argmax(dim=0), :]
            else:
                return print("Invalid forward step")

    def forward(self, t, sigma=None):
        '''
        Project data t onto the latent space.
        :param t: input data, shape (N, D)
        :return:
        '''

        # Forward step = self.transform()
        # n_nodes = t.size()[0]
        # t -= self.scale_mean
        # t /= self.scale_std
        return -self.log_likelihood(t).item(), self.transform(t)  # loss, GTM_output

        t = t.view(n_nodes, -1, 1).to(self.device)  # add 3rd dimension
        # use the same weights for each row of the input
        node_weight = self.weight.expand(n_nodes, -1, -1)  # Shape (n_nodes, input_size, out_size[0]*[1])

        dists = self.pdist_fn(t,
                              node_weight)  # Eq (15): distance between a GTM neuron and the input embedding for node v at layer k

        # Find best matching unit
        losses, bmu_indexes = dists.min(dim=1,
                                        keepdim=True)  # Eq (16): compute the BMU for each forward pass of the GTM
        # losses = dist of BMU, w/ (i*,j*)

        bmu_locations = self.locations[bmu_indexes]

        # coefficenti per i neighborhood
        distance_squares = self.locations - bmu_locations
        distance_squares.pow_(2)
        distance_squares = torch.sum(distance_squares, dim=2)

        if sigma is not None:
            lr_locations = self._neighborhood_fn(distance_squares, sigma)
        else:
            lr_locations = self._neighborhood_fn(distance_squares, self.sigma)

        GTM_output = torch.exp(-dists.to(self.device) + losses.expand_as(dists).to(self.device)) * lr_locations.to(
            self.device)  # Eq (19)

        return bmu_locations, losses.sum().div_(n_nodes).item(), GTM_output

    def train_aggregator(self, x, current_iter, max_iter, lr, first, second):
        '''
        Train the GTM
        :param x: training data (aka t)
        :param current_iter: current epoch of total epoch
        :param max_iter: total epoch
        :return: loss (minimum distance)
        '''
        # x -= self.scale_mean
        # x /= self.scale_std

        if self.learning == 'standard':
            # Set learning rate # TODO Learning rate/sigma vs regularization/gtm_lr -> meno iperparametri usi meno c'è da convalidare
            iter_correction = 1.0 - current_iter / max_iter
            lr = lr * iter_correction
            sigma = self.sigma * iter_correction

            # E-step
            R = self.responsibility(x)

            # M-step
            G = torch.diag(R.sum(dim=1))
            self.W = torch.linalg.solve(
                self.phi.T.matmul(G).matmul(self.phi) + (self.gtm_lr / self.beta) * torch.eye(self.phi.shape[1],
                                                                                              device=self.device),
                self.phi.T.matmul(R).matmul(x))  # W is already transposed, W = M x D, contrary to the paper
            self.beta = x.numel() / torch.cdist(self.phi.matmul(self.W), x.to(self.device), p=2,
                                                compute_mode='donot_use_mm_for_euclid_dist').pow_(2).mul_(R).sum()

            return -self.log_likelihood(x).item()  # - due to MINIMIZATION

        elif self.learning == 'incremental':
            with torch.no_grad():

                # E-step
                R_old = self.R_inc[:, first:second].clone()
                self.R_inc[:, first:second] = self.responsibility(x)
                # torch.diagonal(self.G_inc) = torch.diagonal(self.G_inc) + self.R_inc[:,first:second] - R_old
                self.RX_inc += (self.R_inc[:, first:second] - R_old).matmul(x)

                # M-step
                G = torch.diag(self.R_inc.sum(dim=1))  # compute G not incrementally
                W_old = self.W.clone()
                self.W = torch.linalg.solve(
                    self.phi.T.matmul(G).matmul(self.phi) + (self.gtm_lr / self.beta.float()) * torch.eye(
                        self.phi.shape[1], device=self.device), self.phi.T.matmul(self.RX_inc))
                # estimate inverse variance beta non incrementally - that would require also a copy of W...
                # self.beta = self.X_inc.numel() / torch.cdist(self.phi.matmul(self.W), self.X_inc, p=2,
                #            compute_mode='donot_use_mm_for_euclid_dist').pow_(2).mul_(self.R_inc).sum().double()

                # estimate inverse variance beta INCREMENTALLY
                # beta_new = ( beta^-1 + ( a - b ) / ND )^-1
                # self.beta_old = self.beta.clone()
                a = torch.cdist(self.phi.matmul(self.W), x, p=2, compute_mode='donot_use_mm_for_euclid_dist').pow(2).mul(
                    self.R_inc[:, first:second]).sum()
                b = torch.cdist(self.phi.matmul(W_old), x, p=2, compute_mode='donot_use_mm_for_euclid_dist').pow(2).mul(
                    R_old).sum()
                self.beta = (self.beta.pow(-1) + (a - b) / self.X_inc.numel()).pow(-1)

                return -self.log_likelihood(x).item()  # - due to MINIMIZATION

        else:
            return print("Invalid Learning Method")

        batch_size = x.size()[0]

        # Set learning rate
        iter_correction = 1.0 - current_iter / max_iter
        lr = lr * iter_correction
        sigma = self.sigma * iter_correction

        # Find best matching unit
        bmu_locations, loss, _ = self.forward(x)

        # bmu_locations contains the winner location for each node
        distance_squares = self.locations.float() - bmu_locations.float()
        distance_squares.pow_(2)
        distance_squares = torch.sum(distance_squares, dim=2)

        lr_locations = self._neighborhood_fn(distance_squares, sigma)
        lr_locations.mul_(lr).unsqueeze_(1)

        delta = lr_locations * (x.unsqueeze(2) - self.weight)

        delta = delta.sum(dim=0)

        delta.div_(batch_size)

        # update the weights
        self.weight.data.add_(delta)

        return loss

    def save_result(self, dir, im_size=(0, 0, 0)):
        '''
        Visualizes the weight of the Self Organizing Map(GTM)
        :param dir: directory to save
        :param im_size: (channels, size x, size y)
        :return:
        '''
        images = self.weight.view(im_size[0], im_size[1], im_size[2], self.out_size[0] * self.out_size[1])

        images = images.permute(3, 0, 1, 2)
        save_image(images, dir, normalize=True, padding=1, nrow=self.out_size[0])