import numpy as np
from scipy.cluster.vq import kmeans2

import batman


def get_flux(x, rp, a, ecc, inc, w, fac=None):
    """
    Calculate flux on time axis x given these parameters:

    rp : float
        planet radius in stellar radii

    a : semimajor axis in stellar radii

    ecc : float
        eccentricity

    inc : float
        inclination

    w : float
        argument of periastron

    Returns
    -------
    array : flux on the x-grid
    """
    params = batman.TransitParams()       # object to store transit parameters
    params.t0 = 0.                        # time of inferior conjunction
    params.per = 1.                       # orbital period
    params.rp = rp                        # planet radius (in units of stellar radii)
    params.a = a                          # semi-major axis (in units of stellar radii)
    params.inc = inc                      # orbital inclination (in degrees)
    params.ecc = ecc                      # eccentricity
    params.w = w                          # longitude of periastron (in degrees)
    params.limb_dark = "nonlinear"        # limb darkening model
    params.u = [0.5, 0.1, 0.1, -0.1]      # limb darkening coefficients [u1, u2, u3, u4]

    m = batman.TransitModel(params, x, fac=fac)    # initializes model
    flux = m.light_curve(params)          # calculates light curve
    return flux


def prune_parameters(logp, p0, N=3, frac=0.25):
    """
    This clusters the logp values into N clusters. Then it takes all the best
    clusters up to the cluster that brings the fraction of samples over a certaing
    fraction of all samples (`frac`). It replaces the remaining bad samples by
    randomly picking from the good ones.

    Arguments
    ---------

    logp : array
        log probability values of each coordinate  shape (nwalkers)

    p0 : array
        coordinates of the walkers, shape (nwalkers, ndim)

    Keywords
    --------
    N : int
        number of clusters to search for

    frac : float
        0<frac<1: pick the best clusters until you have more than
        frac of the total walkers

    Returns
    -------
    array : a pruned version of only good coordinates
    """
    # Find clusters in the log probability of the walkers

    x = logp
    X = np.vstack((x, np.zeros_like(x))).T
    kclust, label = kmeans2(X, N, minit='points')

    # Count how many members they have

    cluster_names = np.unique(label)
    counts = np.array([(label == name).sum() for name in cluster_names])

    # Sort them by their logP values

    order = kclust[:, 0].argsort()[::-1]

    # now find out how many we need to use to have the better X%. If we take all belonging to those clusters, we should have most reasonable walkers.

    n_clusters = np.where(np.cumsum(counts[order]) / counts.sum() > frac)[0][0]
    good_clusters = cluster_names[order][:n_clusters + 1]
    good_walkers = np.array([p for p, _label in zip(np.arange(len(label)), label) if _label in good_clusters])
    good_params = p0[good_walkers]
    good_logp = logp[good_walkers]

    # pick new good parameters

    new_good_idx = np.random.choice(np.arange(len(good_logp)), len(x) - len(good_logp))
    new_good_params = good_params[new_good_idx]

    return np.vstack((new_good_params, good_params))


def log_prob(p, x, y, dy, return_blob=False, fac=0.001):

    rp, a, ecc, inc, w = p

    if rp < 0.01 or rp > 100:
        return -np.inf

    if a < 0.01 or a > 100:
        return -np.inf

    if ecc < 0.0 or ecc > 1.0:
        return -np.inf

    if inc < 0.0 or inc > 90:
        return -np.inf

    if w < 0.0 or w > 360:
        return -np.inf

    params = batman.TransitParams()  # object to store transit parameters
    params.t0 = 0.  # time of inferior conjunction
    params.per = 1.  # orbital period
    params.rp = rp  # planet radius (in units of stellar radii)
    params.a = a  # semi-major axis (in units of stellar radii)
    params.inc = inc  # orbital inclination (in degrees)
    params.ecc = ecc  # eccentricity
    params.w = w  # longitude of periastron (in degrees)
    params.limb_dark = "nonlinear"  # limb darkening model
    params.u = [0.5, 0.1, 0.1, -0.1]  # limb darkening coefficients [u1, u2, u3, u4]

    m = batman.TransitModel(params, x, fac=fac)    # initializes model
    flux = m.light_curve(params)  # calculates light curve

    logp = -0.5 * np.sum((y - flux) ** 2. / (2 * dy**2))

    if return_blob:
        return logp, flux
    else:
        return logp
