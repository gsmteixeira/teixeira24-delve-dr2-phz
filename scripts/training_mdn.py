"""
Author: Gabriel Teixeira

Training and inference pipeline for Mixture Density Networks (MDNs) with LMU layers,
applied to photometric redshift probability density estimation.

This script:
- Trains MDN+LMU models using k-fold cross-validation
- Saves best-performing models and training histories
- Performs inference on test data to extract mixture parameters
- Computes full redshift PDFs and point estimates

Associated with:
Teixeira et al. (2024), A&C, DOI: 10.1016/j.ascom.2024.100886
"""

import numpy as np
from sklearn.model_selection import KFold
from utils.utils import mkdir, shuffle_idx, get_z_percentile
from utils.modeling import create_mdn_lmu, calc_PDF_series
from utils.processing import prepare_data

from sklearn.preprocessing import StandardScaler
import tensorflow_probability as tfp
import os
import tensorflow as tf
from tqdm import tqdm
import pickle
import tensorflow.keras as tfk

# ===============================
# GPU / DISTRIBUTION CONFIG
# ===============================

gpus="0"
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = gpus

if len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) > 1:
    strategy = tf.distribute.MirroredStrategy(
        cross_device_ops=tf.distribute.HierarchicalCopyAllReduce()
    )
else:
    strategy = tf.distribute.get_strategy()

# ===============================
# MAIN PIPELINE
# ===============================

def main():
    """
    Main execution function.

    - Loads training and test datasets
    - Trains MDN models using cross-validation
    - Runs inference and stores PDFs and summary statistics
    """

    RESULTS_DIR = mkdir(f'results/griz/')

    TRAIN_PATH = "data/dlcat/dltrain_a.csv"
    TEST_PATH = "data/dlcat/test_data.csv"

    # Training configuration
    EPOCHS = 2
    BATCH_SIZE = 512*2
    RANDOM_STATE = 137
    N_FOLDS = 2
    SCALER = StandardScaler()
    LOAD_MODELS = False

    # MDN parameters
    NUM_COMPONENTS = 20
    EVENT_SHAPE = [1]

    PARAMS_SIZE = int(
        tfp.layers.MixtureNormal.params_size(NUM_COMPONENTS, EVENT_SHAPE)
    )

    # Inference configuration
    ZAXIS = np.linspace(0,1,1000)
    BATCH_PRED = 4096
    N_OBJ = 10  # None for full dataset

    tf.random.set_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    # Data preparation
    x_train, y_train, x_test, _, __ = prepare_data(
        TRAIN_PATH, TEST_PATH, dataset=None, scaler=SCALER
    )

    # Model training
    My_Models, _ = cross_val_fit(
        x_train, y_train, n_folds=N_FOLDS,
        random_state=RANDOM_STATE, save=True,
        save_dir=RESULTS_DIR, batch_size=BATCH_SIZE,
        epochs=EPOCHS, load_models=LOAD_MODELS,
        num_components=NUM_COMPONENTS,
        params_size=PARAMS_SIZE,
        event_shape=EVENT_SHAPE
    )

    # Inference stage
    inference(
        My_Models, x_test, save_dir=RESULTS_DIR, n_folds=N_FOLDS,
        n_obj=N_OBJ, zaxis=ZAXIS,
        batch_pred=BATCH_PRED,
        num_components=NUM_COMPONENTS
    )

    return

# ===============================
# TRAINING WITH CROSS-VALIDATION
# ===============================

def cross_val_fit(X, Y, n_folds=5, random_state=42, 
                  save=True, save_dir='', batch_size=512, 
                  epochs=60, load_models=False,
                  num_components=20, params_size=60, event_shape=[1]):
    """
    Train MDN models using k-fold cross-validation.

    Handles both:
    - n_folds = 1 (manual train/validation split)
    - n_folds > 1 (sklearn KFold)
    """

    My_Models = {}
    My_Fits = {}

    fold = [f'fold_{j}' for j in range(n_folds)]

    # Shuffle data before splitting
    shuf_idx = shuffle_idx(X)
    X = X[shuf_idx]
    Y = Y[shuf_idx]

    # Single-fold special case
    if n_folds == 1:
        val_cut = np.random.uniform(0,1, len(X)) < .1
        val_idx = np.arange(len(X))[val_cut]
        train_idx = np.arange(len(X))[~val_cut]
        splits = [(train_idx, val_idx)]
    else:
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        splits = list(kfold.split(X))

    # Loop over folds
    for i, (train_idx, val_idx) in enumerate(splits):

        mkdir(os.path.join(save_dir, fold[i]))

        best_loss_ckp = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(save_dir, fold[i], 'best_model.h5'),
            monitor='val_loss',
            mode='auto',
            save_best_only=True
        )

        with strategy.scope():
            My_Models[fold[i]] = create_mdn_lmu(
                num_components=num_components, 
                params_size=params_size,
                inp_shape=X.shape[1:],
                n_pixels=X.shape[1],
                event_shape=event_shape
            )

            My_Models[fold[i]].compile(
                loss=lambda y, model: -model.log_prob(y),
                optimizer=tfk.optimizers.Nadam(learning_rate=2e-4)
            )

            My_Models[fold[i]].summary()

            if load_models:
                My_Models[fold[i]].load_weights(
                    os.path.join(save_dir, fold[i], 'best_model.h5')
                )

        if not load_models:
            My_Fits[fold[i]] = My_Models[fold[i]].fit(
                X[train_idx], Y[train_idx],
                validation_data=(X[val_idx], Y[val_idx]),
                verbose=1,
                batch_size=batch_size,
                epochs=epochs,
                callbacks=[best_loss_ckp]
            )

            with open(os.path.join(save_dir, fold[i], 'history.pkl'), 'wb') as fp:
                pickle.dump(My_Fits[fold[i]].history, fp)
                print('dictionary saved successfully to file')

        print('\n')
        i += 1

    return My_Models, My_Fits

# ===============================
# INFERENCE & PDF GENERATION
# ===============================

def inference(My_Models, x_test, save_dir='', n_folds=4,
              n_obj=None, zaxis=np.linspace(0,2,2000),
              batch_pred=1024, num_components=20):
    """
    Run inference on trained models.

    Extracts:
    - Mixture weights
    - Component means and variances
    - Full redshift PDFs
    - Point estimates and uncertainty metrics
    """

    My_Alphas = {}
    My_Mus = {}
    My_Sigmas = {}

    if n_obj:
        new_x_test = x_test[:n_obj]
    else:
        new_x_test = x_test

    fold = [f'fold_{j}' for j in range(n_folds)]
    nsteps = int(len(new_x_test)/batch_pred)

    # Extract MDN parameters
    for j in range(n_folds):

        print('Getting Parameters --- ', fold[j].upper())

        My_Alphas[fold[j]] = np.zeros((len(new_x_test), num_components))
        My_Mus[fold[j]] = np.zeros((len(new_x_test), num_components))
        My_Sigmas[fold[j]] = np.zeros((len(new_x_test), num_components))

        for i in tqdm(range(nsteps)):

            if i == nsteps - 1:
                gm = My_Models[fold[j]](new_x_test[i*batch_pred:])
                My_Alphas[fold[j]][i*batch_pred:] = gm.mixture_distribution.probs_parameter().numpy()
                My_Mus[fold[j]][i*batch_pred:] = np.squeeze(gm.components_distribution.mean().numpy())
                My_Sigmas[fold[j]][i*batch_pred:] = np.squeeze(
                    np.sqrt(gm.components_distribution.variance().numpy())
                )
            else:
                gm = My_Models[fold[j]](new_x_test[i*batch_pred:(i+1)*batch_pred])
                My_Alphas[fold[j]][i*batch_pred:(i+1)*batch_pred] = gm.mixture_distribution.probs_parameter().numpy()
                My_Mus[fold[j]][i*batch_pred:(i+1)*batch_pred] = np.squeeze(gm.components_distribution.mean().numpy())
                My_Sigmas[fold[j]][i*batch_pred:(i+1)*batch_pred] = np.squeeze(
                    np.sqrt(gm.components_distribution.variance().numpy())
                )

        np.save(mkdir(save_dir + fold[j] + '/') + 'my_alphas.npy', My_Alphas[fold[j]])
        np.save(mkdir(save_dir + fold[j] + '/') + 'my_mus.npy', My_Mus[fold[j]])
        np.save(mkdir(save_dir + fold[j] + '/') + 'my_sigmas.npy', My_Sigmas[fold[j]])

    # Compute PDFs and statistics
    My_PDFs = {}
    My_Photoz = {}
    My_Errors = {}
    My_Medians = {}

    for j in range(n_folds):

        print('Getting PDFs --- ', fold[j].upper())

        My_PDFs[fold[j]], My_Photoz[fold[j]], _ = calc_PDF_series(
            weights=My_Alphas[fold[j]],
            means=My_Mus[fold[j]],
            stds=My_Sigmas[fold[j]], 
            x_range=zaxis, optimize_zml=True
        )

        My_Errors[fold[j]] = (
            get_z_percentile(My_PDFs[fold[j]], zaxis, p=.8405) -
            get_z_percentile(My_PDFs[fold[j]], zaxis, p=.1585)
        )

        My_Medians[fold[j]] = get_z_percentile(My_PDFs[fold[j]], zaxis, p=.5)

        np.save(mkdir(save_dir + fold[j] + '/') + 'my_pdfs.npy', My_PDFs[fold[j]])
        np.save(mkdir(save_dir + fold[j] + '/') + 'my_photoz.npy', My_Photoz[fold[j]])
        np.save(mkdir(save_dir + fold[j] + '/') + 'my_zerr.npy', My_Errors[fold[j]])
        np.save(mkdir(save_dir + fold[j] + '/') + 'my_medians.npy', My_Medians[fold[j]])

    return My_Photoz, My_PDFs, My_Alphas, My_Mus, My_Sigmas

if __name__=="__main__":
    main()