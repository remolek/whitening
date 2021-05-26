#!/usr/bin/python
# -*- coding: utf-8 -*-

# ------------------------------------
# file: whiten.py
# date: Tue May 25 19:00 2021
# author:
# Jeremi Ochab
# github.com/remolek
# jeremi.ochab AT uj DOT edu DOT pl
#
# Licensed under GPLv3
# 
# Based on:
# 1. github.com/mwv/zca by Maarten Versteegh
# 2. https://gist.github.com/joelouismarino/ce239b5601fff2698895f48003f7464b by Joe Marino
# 3. https://CRAN.R-project.org/package=whitening by Korbinian Strimmer, Takoua Jendoubi, Agnan Kessy, Alex Lewin
#
# ------------------------------------
"""whiten: Implements the whitening methods (ZCA, PCA, Cholesky, ZCA-cor, and PCA-cor) discussed in Kessy, Lewin, and Strimmer (2018)
``Optimal whitening and decorrelation'', <doi:10.1080/00031305.2016.1277159>
with a sklearn-like interface
"""

from __future__ import division

import numpy as np

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array, as_float_array

class whiten(BaseEstimator, TransformerMixin):
    def __init__(self, regularization = np.finfo(np.float64).eps, copy=False):
        self.regularization = regularization
        self.copy = copy

    def fit(self, X, method='zca', y=None):
        """Compute the mean, whitening and dewhitening matrices using specified whitening method.

        Parameters
        ----------
        X : array-like with shape [n_samples, n_features]
            The data used to compute the mean, whitening and dewhitening
            matrices.
        method: Whitening method. Must be one of 'zca', 'zca_cor', 'pca',
                'pca_cor', or 'cholesky'.
        """
        X = check_array(X, accept_sparse=None, copy=self.copy,
                        ensure_2d=True)
        X = as_float_array(X, copy=self.copy)
        self.mean_ = X.mean(axis=0)
        X_ = X - self.mean_
        cov = np.dot(X_.T, X_) / (X_.shape[0]-1)

        if method in ['zca', 'pca', 'cholesky']:
            Lambda, U = np.linalg.eigh(cov)
            Lambda = np.flip(Lambda)
            U = np.flip(U,axis=1)
            U = np.sign(np.diag(U))*U
            s = np.sqrt(Lambda.clip(self.regularization))
            s_inv = np.diag(1./s)
            s = np.diag(s)
            if method == 'zca':
                self.whiten_ = np.dot(np.dot(U, s_inv), U.T)
                self.dewhiten_ = np.dot(np.dot(U, s), U.T)
            elif method =='pca':
                self.whiten_ = np.dot(s_inv, U.T)
                self.dewhiten_ = np.dot(U, s)
            elif method == 'cholesky':
                self.whiten_ = np.linalg.cholesky(np.dot(U, np.dot(np.diag(1./Lambda.clip(self.regularization)), U.T))).T
                self.dewhiten_ = np.dot(cov,self.whiten_.T)
        elif method in ['zca_cor', 'pca_cor']:
            V_sqrt = np.std(X, axis=0)
            V_inv = np.diag(1./V_sqrt)
            V_sqrt = np.diag(V_sqrt)

            P = np.dot(np.dot(V_inv, cov), V_inv)
            Theta, G = np.linalg.eigh(P)
            Theta = np.flip(Theta)
            G = np.flip(G,axis=1)
            G = np.sign(np.diag(G))*G
            p = np.sqrt(Theta.clip(self.regularization))
            p_inv = np.diag(1./p)
            p = np.diag(p)
            if method == 'zca_cor':
                self.whiten_ = np.dot(np.dot(np.dot(G, p_inv), G.T), V_inv)
                self.dewhiten_ = np.dot(V_sqrt,np.dot(np.dot(G, p), G.T))
            elif method == 'pca_cor':
                self.whiten_ = np.dot(np.dot(p_inv, G.T), V_inv)
                self.dewhiten_ = np.dot(V_sqrt, np.dot(G, p))
        else:
            raise Exception('Whitening method not found.')
        
        return self
    

    def transform(self, X, y=None, copy=None):
        """Perform whitening

        Parameters
        ----------
        X : array-like with shape [n_samples, n_features]
            The data to whiten along the features axis.
        """
        check_is_fitted(self, 'mean_')
        X = as_float_array(X, copy=self.copy)
        return np.dot(X - self.mean_, self.whiten_.T)

    def inverse_transform(self, X, copy=None):
        """Undo the whitening transform and rotate back
        to the original representation

        Parameters
        ----------
        X : array-like with shape [n_samples, n_features]
            The data to rotate back.
        """
        check_is_fitted(self, 'mean_')
        X = as_float_array(X, copy=self.copy)
        return np.dot(X, self.dewhiten_.T) + self.mean_

