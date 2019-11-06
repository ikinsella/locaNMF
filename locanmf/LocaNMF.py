""" Module Docstring """
# Standard Imports
from time import time

# Third Party Imports
import torch
import numpy as np
from scipy import spatial
from sklearn.utils.extmath import randomized_svd

# Custom Module Imports
from .video import RegionMetadata
from .video import LowRankVideo
from .demix import HalsNMF
from .demix import LocalizedNMF


def adaptive_fit(video_mats,
                 valid_mask,
                 region_map,
                 rank_range=(2, 14, 1),
                 device='cuda',
                 **kwargs):
    """Perform LocaNMF Of A Low Rank Video With Automated Param Tuning

    Parameter:
        video_mats: brain masked spatial components from denoiser
        valid_mask: valid brain mask
        region_map: preprocessed allen Dorsal Map
        rank_range: rank range
        device: computation device, default is cuda
        **kwargs: optional additional input arguments

    Return:
        nmf_factors: localized NMF components

    """

    # Create Region Metadata
    region_mats = extract_region_metadata(valid_mask,
                                          region_map,
                                          min_size=rank_range[1])
    region_metadata = RegionMetadata(region_mats[0].shape[0],
                                     region_mats[0].shape[1:],
                                     device=device)
    region_metadata.set(torch.from_numpy(region_mats[0].astype(np.uint8)),
                        torch.from_numpy(region_mats[1]),
                        torch.from_numpy(region_mats[2].astype(np.int64)))

    # Perform SVD of Each Region & XFER to device
    region_videos = factor_region_videos(video_mats,
                                         region_mats[0],
                                         rank_range[1])

    # XFER Full FOV Mats to device
    low_rank_video = LowRankVideo(
        (int(np.sum(valid_mask)),) + video_mats[1].shape, device=device
    )
    low_rank_video.set(torch.from_numpy(video_mats[0].T),
                       torch.from_numpy(video_mats[1]))

    # Perform grid search for region-ranks
    return rank_linesearch(low_rank_video,
                           region_metadata,
                           region_videos,
                           rank_range=rank_range,
                           device=device,
                           **kwargs)


def extract_region_metadata(valid_mask,
                            region_map,
                            min_size=0):
    """Generates Support & Distance Matrices From FOV Mask + Region Map

    Parameter:
        valid_mask: valid brain mask
        region_map: preprocessed allen Dorsal Map
        min_size: minimum number of pixels in Allen map for it
            to be considered a brain region

    Return:
        region_metadata: with
        region_mats[0] = [unique regions x pixels] the mask of each region,
        region_mats[1] = [unique regions x pixels] the distance penalty of each region,
        and region_mats[2] = [unique regions] area code

    """
    # Get All (Potentially Signed) Labels From Map
    region_labels = np.unique(region_map[valid_mask])
    region_labels = region_labels[region_labels != 0]
    # Initialize Output Matrices
    keep_mask = np.ones(len(region_labels), dtype=bool)
    support = np.zeros((len(region_labels), np.sum(valid_mask)),
                       dtype=bool)
    distance = np.zeros((len(region_labels), np.sum(valid_mask)),
                        dtype=np.float32)
    # For Each Region, Detect Support & Generate Distmat
    allpixels = np.stack(np.where(valid_mask)).T
    for idx, label in enumerate(region_labels):
        full_region_mask = region_map == label
        support[idx, :] = full_region_mask[valid_mask]
        if np.sum(support[idx, :]) > min_size:
            pixels_in_roi = np.stack(np.where(full_region_mask)).T
            tree = spatial.cKDTree(pixels_in_roi)
            mindists, _ = tree.query(allpixels)
            distance[idx, :] = mindists.T
        else:
            keep_mask[idx] = False
    return (support[keep_mask, :],
            distance[keep_mask, :],
            region_labels[keep_mask])


def factor_region_videos(video_mats,
                         region_masks,
                         max_rank,
                         device='cuda'):
    """Perform within-region SVD for use in full-fov initialization

    Parameter:
        video_mats: brain masked spatial components from denoiser
        region_masks: brain mask of each region
        max_rank: maximum number of components per brain region
        device: computation device

    Return:
        region_low_rank_videos: low rank videos in each region


    """
    # Allocate Space For Within-Region Low Rank Videos On Device
    region_low_rank_videos = [
        LowRankVideo((int(np.sum(mask)), max_rank, video_mats[1].shape[-1]),
                     device=device) for mask in region_masks
    ]

    # If Rank < Num Frames, Orthogonalize Right Components
    orthogonalize = video_mats[1].shape[0] < video_mats[1].shape[1]
    if orthogonalize:
        temporal_basis, rank_mixing_weights = np.linalg.qr(video_mats[1].T)
        right_factor = rank_mixing_weights.T
    else:
        right_factor = video_mats[1]

    # Create Low Rank Video For Each Region
    for mask, video in zip(region_masks, region_low_rank_videos):

        # Peform Truncated SVD
        region_mov = video_mats[0][mask, :].dot(right_factor)
        svd_mats = randomized_svd(region_mov,
                                  n_components=min(max_rank, np.sum(mask)),
                                  n_iter=2)

        # Rescale Results
        scale = np.mean(svd_mats[0], axis=0)
        spatial = svd_mats[0] / scale[None, ...]
        temporal = (scale * svd_mats[1])[..., None] * svd_mats[2]
        if orthogonalize:
            temporal = np.dot(temporal, temporal_basis.T)
        video.set(torch.from_numpy(spatial).t(), torch.from_numpy(temporal))

    return region_low_rank_videos


def rank_linesearch(low_rank_video,
                    region_metadata,
                    region_lr_videos,
                    rank_range=(2, 14, 1),
                    maxiter_rank=20,
                    nnt=False,
                    verbose=[False, False, False],
                    indent='',
                    device='cuda',
                    **kwargs):
    """Increment Per-Region Rank Until Local R^2 Fits Are Sufficiently High

    Parameter:
        low_rank_video: LowRankVideo class object
        region_metadata: RegionMetadata class object
        region_lr_videos: low rank videos in each region
        rank_range: rank range
        maxiter_rank: maximum iteration rank
        nnt: whether or not temporal components should be constrained to be nonnegative
        verbose: whether or not print status update
        indent: previous identation for printing status update
        device: computation device
        **kwargs: additional optional input arguments

    Return:
        nmf_factors: localized NMF components

    """

    # Parse Args
    if isinstance(verbose, bool):
        verbose = [verbose] * 3

    # Initialize Per-Region Factorizations
    max_ranks, curr_ranks, region_factors = [], [], []
    for video in region_lr_videos:
        max_ranks.append(len(video))
        region_factors.append(HalsNMF(max_ranks[-1],
                                      (video.shape[0],
                                       len(video),
                                       video.shape[-1]),
                                      device=device))
        init_from_low_rank_video(video,
                                 region_factors[-1],
                                 rank_range[0],
                                 nnt=nnt,
                                 verbose=verbose[-1],
                                 device=device,
                                 **kwargs)
        curr_ranks.append(len(region_factors[-1]))

    # Initialize Full Factorization
    nmf_factors = LocalizedNMF(np.sum(max_ranks),
                               (low_rank_video.shape[0],
                                len(low_rank_video),
                                low_rank_video.shape[-1]),
                               device=device)

    # Line Search To Update Per-Region Rank
    refit_flags = [False] * len(region_lr_videos)
    for itr in range(maxiter_rank):
        if verbose[0]:
            if device=='cuda': torch.cuda.synchronize()
            print(indent + '|--v Rank Line Search Iteration {}'.format(itr))
            print(indent + '|  |--v Initialization')
            itr_t0 = time()
            step_t0 = itr_t0

        # Perform Within-Region NMF TO Init Components
        for rdx, refit in enumerate(refit_flags):
            if refit:
                init_from_low_rank_video(region_lr_videos[rdx],
                                         region_factors[rdx],
                                         len(region_factors[rdx]) + rank_range[-1],
                                         verbose=verbose[-1],
                                         nnt=nnt,
                                         indent=indent+'|  |  ',
                                         device=device,
                                         **kwargs)
                curr_ranks[rdx] += 1
                refit_flags[rdx] = False
        nmf_factors.set_from_regions(region_factors, region_metadata)
        if verbose[0]:
            if device=='cuda': torch.cuda.synchronize()
            print(indent + '|  |  \'-total : %f seconds' % (time()-step_t0))
            print(indent + '|  |--v Lambda Line Search')
            step_t0 = time()

        # Tune Per-Component Lambdas
        lambda_iters = lambda_linesearch(low_rank_video,
                                         region_metadata,
                                         nmf_factors,
                                         nnt=nnt,
                                         verbose=verbose[1:],
                                         indent=indent + '|  |  ',
                                         device=device,
                                         **kwargs)
        if verbose[0]:
            if device=='cuda': torch.cuda.synchronize()
            print(indent + '|  |  \'- {:g} iterations took {:g} seconds'.format(lambda_iters, time()-step_t0))
            step_t0 = time()

        # Increment K For Each Region Below r^2 thresh
        for rdx, (rank, max_rank) in enumerate(zip(curr_ranks, max_ranks)):
            if rank < max_rank:
                refit_flags[rdx],_ = evaluate_fit_to_region(low_rank_video,
                                                          nmf_factors,
                                                          region_metadata.support.data[rdx],
                                                          **kwargs)
        if verbose[0]:
            if device=='cuda': torch.cuda.synchronize()
            print(indent + '|  |--> R2 Evaluation took {:g} seconds'.format(time()-step_t0))

        # Update Progress
        if verbose[0]:
            if device=='cuda': torch.cuda.synchronize()
            print(indent + '|  \'-total : {:g} seconds'.format(time()-itr_t0))
        if not np.any(refit_flags):
            break
    return nmf_factors


def init_from_low_rank_video(low_rank_video,
                             factorization,
                             num_components,
                             nnt=False,
                             verbose=False,
                             indent='',
                             device='cuda',
                             **kwargs):
    """Initialize Region Using NMF On Low Rank Approx

    Parameter:
        low_rank_video: LowRankVideo class object
        factorization: localized NMF components
        num_components: number of components
        nnt: whether or not temporal components should be constrained to be nonnegative
        verbose: whether or not print status update
        indent: previous identation for printing status update
        device: computation device
        **kwargs: additional optional input arguments

    """
    factorization.num_components = num_components
    factorization.spatial.data.fill_(1.0)
    factorization.temporal.data.copy_(
        low_rank_video.temporal.data[:num_components]
    )
    if verbose:
        print(indent + '|--v HALS Itertations')
    hals(low_rank_video,
         factorization,
         nnt=nnt,
         verbose=verbose,
         indent=indent+'|  ',
         device=device,
         **kwargs)


def evaluate_fit_to_region(low_rank_video,
                           video_factorization,
                           region_mask,
                           r2_thresh=.98,
                           sample_prop=(1, 1),
                           device='cuda',
                           **kwargs):
    """Compute Coef Of Determination Of Current Fit

    Parameter:
        low_rank_video: low rank factored video
        video_factorization: localized NMF components
        region_mask: region masks
        r2_thresh: minimum r squared for a fit to be considered to be good
        sample_prop: proportion of pixels and frames used to compute r squared
        device: computation device
        **kwargs: optional additional input arguments

    Return:
        r2_est < r2_thresh: whether r2 estimated is lower that r2_thresh
        r2_est: r2 estimated

    """

    region_idx = torch.nonzero(region_mask).squeeze()
    if sample_prop[0] < 1.0:
        perm = torch.randperm(region_idx.size(0), device=device)
        k = int(np.ceil(sample_prop[0] * region_idx.size(0)))
        region_idx = torch.index_select(region_idx,
                                        0,
                                        perm[:k])

    if sample_prop[1] < 1.0:
        perm = torch.randperm(low_rank_video.temporal.shape[1], device=device)
        temporal_idx = perm[:int(np.ceil(sample_prop[1] * low_rank_video.temporal.shape[1]))]
        mov = torch.matmul(torch.index_select(low_rank_video.spatial.data,
                                              1,
                                              region_idx).t(),
                           torch.index_select(low_rank_video.temporal.data,
                                              1,
                                              temporal_idx))
        var = torch.mean(torch.var(mov, dim=1, unbiased=False)) # TODO: Precompute this
        torch.addmm(beta=1,
                    input=mov,
                    alpha=-1,
                    mat1=torch.index_select(video_factorization.spatial.data,
                                            1,
                                            region_idx).t(),
                    mat2=torch.index_select(video_factorization.temporal.data,
                                            1,
                                            temporal_idx),
                    out=mov)
    else:
        mov = torch.matmul(torch.index_select(low_rank_video.spatial.data,
                                              1,
                                              region_idx).t(),
                           low_rank_video.temporal.data)
        var = torch.mean(torch.var(mov, dim=1, unbiased=False))  # TODO: Precompute this
        torch.addmm(beta=1,
                    input=mov,
                    alpha=-1,
                    mat1=torch.index_select(video_factorization.spatial.data,
                                            1,
                                            region_idx).t(),
                    mat2=video_factorization.temporal.data,
                    out=mov)
    r2_est = 1 - (torch.mean(mov.pow_(2)).item() / var.item())
    return r2_est < r2_thresh, r2_est


def lambda_linesearch(video,
                      region_metadata,
                      localized_factorization,
                      lambda_init=.005,
                      lambda_step=1.5,
                      loc_thresh=20,
                      maxiter_lambda=15,
                      nnt=False,
                      verbose=False,
                      indent='',
                      device='cuda',
                      **kwargs):
    """Tune Lambdas Until Components Are Sufficiently Region-Localized

    Parameter:
        video: LowRankVideo class object
        region_metadata: RegionMetadata class object
        localized_factorization: localized NMF factors
        lambda_init: initial lambda value for lambda search
        lambda_step: step size for tuning lambda parameter
        loc_thresh: minimum localization required for a component
        maxiter_lambda: maximum number of iteration for tuning lambda parameter
        nnt: whether or not temporal components should be constrained to be nonnegative
        verbose: whether or not print status update
        indent: previous identation for printing status update
        device: computation device
        **kwargs: optional additional input arguments

    Return:
        hals iteration counter

    """


    # Parse Args
    if isinstance(verbose, bool):
        verbose = [verbose] * 2

    # Initialize Lambdas
    localized_factorization.lambdas.data.fill_(lambda_init)

    # Line Search To Update Lambdas
    breakout = False
    for itr in range(maxiter_lambda):
        if verbose[0]:
            if device=='cuda': torch.cuda.synchronize()
            print(indent + '|--v Lambda Line Search Iteration {:g}'.format(itr+1))
            itr_t0 = time()
            step_t0 = itr_t0

        # Set Distance Matrix
        torch.mul(localized_factorization.distance.data,
                  .5 * localized_factorization.lambdas.data[..., None],
                  out=localized_factorization.distance.scratch)
        if verbose[0]:
            if device=='cuda': torch.cuda.synchronize()
            print(indent + '|  |--> Setting Distmat took {:g} seconds'.format(time()-step_t0))
            print(indent + '|  |--v HALS Iterations')
            step_t0 = time()

        # Perform HALS
        hals_iters = hals(video,
                          localized_factorization,
                          nnt=nnt,
                          verbose=verbose[-1],
                          indent=indent+'|  |  ',
                          device=device,
                          **kwargs)
        if verbose[0]:
            if device=='cuda': torch.cuda.synchronize()
            print(indent + '|  |  \'- {:g} iterations took : {:g} seconds'.format(hals_iters, time()-step_t0))
            step_t0 = time()

        # Update Lambdas
        torch.index_select(region_metadata.support.data.to(torch.float32),
                           0,
                           localized_factorization.regions.data,
                           out=localized_factorization.distance.scratch)
        torch.mul(localized_factorization.spatial.data,
                  localized_factorization.distance.scratch,
                  out=localized_factorization.spatial.scratch)
        torch.div(
            torch.norm(localized_factorization.spatial.data, p=2, dim=-1),
            torch.norm(localized_factorization.spatial.scratch, p=2, dim=-1),
            out=localized_factorization.scale.data
        )
        localized_factorization.scale.data.gt_(np.sqrt(100 / loc_thresh))
        if torch.sum(localized_factorization.scale.data) > 0:
            torch.mul(localized_factorization.scale.data,
                      lambda_step - 1.0,
                      out=localized_factorization.scale.data)
            torch.add(localized_factorization.scale.data,
                      1.0,
                      out=localized_factorization.scale.data)
            torch.mul(localized_factorization.scale.data,
                      localized_factorization.lambdas.data,
                      out=localized_factorization.lambdas.data)
        else:
            breakout = True
        if verbose[0]:
            if device=='cuda': torch.cuda.synchronize()
            print(indent + '|  |--> Lambda update took {:g} seconds'.format(time()-step_t0))

        # Update Progress
        if verbose[0]:
            if device=='cuda': torch.cuda.synchronize()
            print(indent + '|  \'-total : {:g} seconds'.format(time()-itr_t0))
        if breakout:
            break

    return itr + 1


def hals(video,
         video_factorization,
         maxiter_hals=30,
         nnt=False,
         verbose=False,
         indent='',
         device='cuda',
         **kwargs):
    """Perform maxiter HALS updates To Temporal & Spatial Components

    Parameter:
        video: LowRankVideo class object
        video_factorization: localized NMF factors
        maxiter_hals: maximum number of iterations to tune hals
        nnt: whether or not temporal components should be constrained to be nonnegative
        verbose: whether or not print status update
        indent: previous identation for printing status update
        device: computation device
        **kwargs: optional additional input arguments

    Return:
        hals iteration counter

    """
    for itr in range(maxiter_hals):
        if verbose:
            if device=='cuda': torch.cuda.synchronize()
            print(indent + '|--v HALS Iteration {:g}'.format(itr+1))
            itr_t0 = time()
            step_t0 = itr_t0

        # Spatial Update Step
        video_factorization.update_spatial(video)
        if verbose:
            if device=='cuda': torch.cuda.synchronize()
            print(indent + '|  |--> Spatial update took {:g} seconds'.format(time()-step_t0))
            step_t0 = itr_t0

        # Remove Empty Components
        video_factorization.prune_empty_components()
        video_factorization.normalize_spatial()
        if verbose:
            if device=='cuda': torch.cuda.synchronize()
            print(indent + '|  |--> Component prune after spatial update took {:g} seconds'.format(time()-step_t0))
            step_t0 = itr_t0

        # Temporal Update Step
        video_factorization.update_temporal(video, nonnegative=nnt)
        if verbose:
            if device=='cuda': torch.cuda.synchronize()
            print(indent + '|  |--> Temporal update took {:g} seconds'.format(time()-step_t0))
            print(indent + '|  \'-total : {:g} seconds'.format(time()-itr_t0))
        if nnt:
            # Remove Empty Components
            video_factorization.prune_empty_components()
            if verbose:
                if device=='cuda': torch.cuda.synchronize()
                print(indent + '|  |--> Component prune after temporal update took {:g} seconds'.format(time()-step_t0))
                step_t0 = itr_t0

    return itr + 1

def version():
    """ Get version of LocaNMF

    """
    print('version = 1.0')


