"""
Author: Gabriel Teixeira

Neural network modeling utilities for photometric redshift estimation.

This module defines:
- A Mixture Density Network (MDN) architecture using LMU layers
- Utilities to compute redshift probability density functions (PDFs)
- Helper routines for visualizing training convergence

Associated with:
Teixeira et al. (2024), A&C, DOI: 10.1016/j.ascom.2024.100886
"""

import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import keras_lmu
import matplotlib.pyplot as plt
import tensorflow.keras as tfk

def create_mdn_lmu(num_components, params_size, inp_shape, n_pixels, event_shape = [1]):
    """
    Build a Mixture Density Network (MDN) with an LMU backbone.

    The network outputs a mixture of Gaussians whose parameters are learned
    from the input photometry sequence.

    Parameters
    ----------
    num_components : int
        Number of Gaussian components in the mixture.
    params_size : int
        Total number of parameters required by the mixture distribution.
    inp_shape : tuple
        Shape of the input features.
    n_pixels : int
        Sequence length (used as LMU theta parameter).
    event_shape : list
        Shape of the target variable.

    Returns
    -------
    tf.keras.Model
        Compiled MDN model returning a TensorFlow Probability distribution.
    """
    
    # LMU layer for temporal representation learning
    lmu_layer = keras_lmu.LMU(
                            memory_d=1,
                            order=128,
                            theta=n_pixels,
                            hidden_cell=tfk.layers.SimpleRNNCell(212),
                            hidden_to_memory=False,
                            memory_to_memory=False,
                            input_to_hidden=True,
                            kernel_initializer="glorot_normal",
                        )

    # Model input
    inp = tfk.Input((n_pixels, 1))
    x = lmu_layer(inp)

    # Fully connected feature extraction layers
    x = tfk.layers.Dense(96, activation='relu')(x)
    x = tfk.layers.Dropout(0.2)(x)
    x = tfk.layers.BatchNormalization()(x)
    x = tfk.layers.Dense(96, activation='relu')(x)
    x = tfk.layers.Dropout(0.2)(x)
    x = tfk.layers.BatchNormalization()(x)

    # Linear layer producing MDN parameters
    x = tfk.layers.Dense(params_size)(x)

    # Mixture distribution definition using TensorFlow Probability
    # x = tfp.layers.MixtureNormal(num_components, event_shape)(x)
    x = tfp.layers.DistributionLambda(
        make_distribution_fn=lambda t: tfp.distributions.MixtureSameFamily(
            mixture_distribution=tfp.distributions.Categorical(
                logits=t[..., :num_components]
            ),
            components_distribution=tfp.distributions.Normal(
                loc=t[..., num_components:2*num_components],
                scale=tf.nn.softplus(t[..., 2*num_components:])
            )
        )
    )(x)

    model = tfk.models.Model(inp, x)
    
    return model

def calc_PDF_series(weights, means, stds, x_range=None, optimize_zml=False):
    """
    Compute photometric redshift PDFs from MDN outputs.

    PDFs are constructed as weighted sums of Gaussian components and can be
    optionally refined to obtain optimized maximum-likelihood estimates.

    Parameters
    ----------
    weights : array-like
        Mixture weights.
    means : array-like
        Means of Gaussian components.
    stds : array-like
        Standard deviations of Gaussian components.
    x_range : array-like, optional
        Redshift grid for PDF evaluation.
    optimize_zml : bool
        Whether to refine the maximum-likelihood redshift estimate.

    Returns
    -------
    PDFs : ndarray
        Array of probability density functions.
    zmls : ndarray
        Maximum-likelihood redshift estimates.
    x : ndarray
        Redshift grid.
    """
    
    # Default redshift grid
    if x_range is None:
        x = np.arange(-0.005, 1+0.001, 0.001) 
    else:
        x = x_range
                      
    # Ensure numpy arrays
    if type(weights) != np.ndarray:
        weights = np.array(weights)
        means   = np.array(means)
        stds    = np.array(stds)

    PDFs           = []
    optimized_zmls = np.empty(len(means))
    
    # Case: multiple objects
    if np.ndim(weights) == 2:
        for i in tqdm(range(len(weights))):
            PDF = np.sum(
                weights[i] * (1/(stds[i]*np.sqrt(2*np.pi))) *
                np.exp((-1/2) * ((x[:,None]-means[i])**2)/(stds[i])**2),
                axis=1
            )
            PDFs.append(PDF)
        zmls = x[np.argmax(PDFs, axis=1)]
        
    # Case: single object
    if np.ndim(weights) == 1:
        PDF = np.sum(
            weights * (1/(stds*np.sqrt(2*np.pi))) *
            np.exp((-1/2) * ((x[:,None]-means)**2)/(stds)**2),
            axis=1
        )
        PDFs = PDF
        zmls = x[np.argmax(PDFs)]

    # Optional refinement of the maximum-likelihood redshift
    if optimize_zml == True:
        for i in tqdm(range(len(weights))):
            optimized_x   = np.linspace(zmls[i]-0.002, zmls[i]+0.002, 500, endpoint=True)
            optimized_PDF = np.sum(
                weights[i] * (1/(stds[i]*np.sqrt(2*np.pi))) *
                np.exp((-1/2) * ((optimized_x[:,None]-means[i])**2)/(stds[i])**2),
                axis=1
            )
            optimized_zml = optimized_x[np.argmax(optimized_PDF)]

            optimized_x   = np.linspace(optimized_zml-0.001, optimized_zml+0.001, 300, endpoint=True)
            optimized_PDF = np.sum(
                weights[i] * (1/(stds[i]*np.sqrt(2*np.pi))) *
                np.exp((-1/2) * ((optimized_x[:,None]-means[i])**2)/(stds[i])**2),
                axis=1
            )
            optimized_zmls[i] = optimized_x[np.argmax(optimized_PDF)]

        zmls = optimized_zmls
                
    return np.vstack(PDFs), zmls, x

def loss_plot(Fits, initial_epoch=0, save_dir=None):
    """
    Plot mean and dispersion of training and validation loss curves
    across cross-validation folds.

    Parameters
    ----------
    Fits : dict
        Dictionary containing Keras History objects for each fold.
    initial_epoch : int
        Epoch index from which to start plotting.
    save_dir : str, optional
        Directory where the plot will be saved.
    """
    
    all_train_losses = np.vstack([Fits[xn].history['loss'] for xn in Fits]).T
    all_val_losses = np.vstack([Fits[xn].history['val_loss'] for xn in Fits]).T

    epochs = all_val_losses.shape[0]
    
    train_loss_mean = np.mean(all_train_losses, axis=1)[initial_epoch:]
    val_loss_mean = np.mean(all_val_losses, axis=1)[initial_epoch:]

    train_loss_std = np.std(all_train_losses, axis=1)[initial_epoch:]
    val_loss_std = np.std(all_val_losses, axis=1)[initial_epoch:]

    plt.figure(figsize=(6,6))

    # Mean loss curves
    plt.plot(range(initial_epoch, epochs),
             train_loss_mean,
             color='blue', label='train mean')
    plt.plot(range(initial_epoch, epochs),
             val_loss_mean,
             color='orange', label='validation mean')

    # Uncertainty bands
    plt.fill_between(range(initial_epoch, epochs),
                     train_loss_mean+train_loss_std,
                     train_loss_mean-train_loss_std,
                     color='blue',
                     alpha=0.5)
    plt.fill_between(range(initial_epoch, epochs),
                     val_loss_mean+val_loss_std,
                     val_loss_mean-val_loss_std,
                     color='orange',
                     alpha=0.5)

    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'loss.png'))