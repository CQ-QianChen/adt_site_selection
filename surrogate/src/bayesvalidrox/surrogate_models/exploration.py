#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exploration for sequential training of metamodels
"""

import numpy as np
from scipy.spatial import distance


class Exploration:
    """
    Created based on the Surrogate Modeling Toolbox (SUMO) [1].

    [1] Gorissen, D., Couckuyt, I., Demeester, P., Dhaene, T. and Crombecq, K.,
        2010. A surrogate modeling and adaptive sampling toolbox for computer
        based design. Journal of machine learning research.-Cambridge, Mass.,
        11, pp.2051-2055. sumo@sumo.intec.ugent.be - http://sumo.intec.ugent.be

    Attributes
    ----------
    exp_design : obj
        bvr.ExpDesign object.
    n_candidates : int
        Number of candidate samples.
    mc_criterion : str
        Selection crieterion. The default is `'mc-intersite-proj-th'`. Another
        option is `'mc-intersite-proj'`.
    w : int
        Number of random points in the domain for each sample of the
        training set.
    """

    def __init__(
        self,
        exp_design,
        n_candidates,
        mc_criterion="mc-intersite-proj-th",
        verbose=False,
    ):
        self.exp_design = exp_design
        self.n_candidates = n_candidates
        self.mc_criterion = mc_criterion
        self.verbose = verbose

        # Other settings
        self.w = 100
        self.closest_points = None

    def get_exploration_samples(self):
        """
        This function generates candidates to be selected as new design and
        their associated exploration scores.

        Returns
        -------
        all_candidates : array of shape (n_candidates, n_params)
            A list of samples.
        exploration_scores: arrays of shape (n_candidates)
            Exploration scores.
        """
        explore_method = self.exp_design.explore_method

        if explore_method.lower() == "voronoi":
            # Generate samples using the Voronoi method
            all_candidates, exploration_scores = self.get_vornoi_samples()
        else:
            # Generate samples using the MC method
            all_candidates, exploration_scores = self.get_mc_samples()

        return all_candidates, exploration_scores

    # -------------------------------------------------------------------------
    def get_vornoi_samples(self):
        """
        This function generates samples based on voronoi cells and their
        corresponding scores

        Returns
        -------
        new_samples : array of shape (n_candidates*n_old_x, n_params)
            A list of samples.
        exploration_scores: arrays of shape (n_candidates)
            Exploration scores.
        """
        # Get the old exp_design samples
        old_ed_x = self.exp_design.x
        ndim = old_ed_x.shape[1]

        # Calculate error
        error_voronoi = self.approximate_voronoi(old_ed_x)

        # Pick the best candidate point in the voronoi cell
        # for each best sample
        selected_samples = np.empty((0, ndim))
        bad_samples = []

        for index in range(len(error_voronoi)):
            # Get candidate new samples from voronoi tesselation
            candidates = self.closest_points[index]

            # Get total number of candidates
            n_can = candidates.shape[0]

            # Skip if no candidate samples around
            if n_can == 0:
                if self.verbose:
                    print(
                        "The following sample has been skipped because there "
                        "were no candidate samples around it..."
                    )
                    print(old_ed_x[index])
                bad_samples.append(index)
                continue

            # Find candidate that is farthest away from any existing sample
            max_min_distance = 0
            best_candidate = 0
            min_intersite_dist = np.zeros((n_can))
            min_projected_dist = np.zeros((n_can))

            for j in range(n_can):

                new_samples = np.vstack((old_ed_x, selected_samples))

                # Find min distorted distance from all other samples
                euclidean_dist = self._build_dist_matrix_point(
                    new_samples, candidates[j], do_sqrt=True
                )
                min_euclidean_dist = np.min(euclidean_dist)
                min_intersite_dist[j] = min_euclidean_dist

                # Check if this is the maximum minimum distance from all other
                # samples
                if min_euclidean_dist >= max_min_distance:
                    max_min_distance = min_euclidean_dist
                    best_candidate = j

                # Projected distance
                projected_dist = distance.cdist(
                    new_samples, [candidates[j]], "chebyshev"
                )
                min_projected_dist[j] = np.min(projected_dist)

            if self.mc_criterion == "mc-intersite-proj":
                weight_euclidean_dist = 0.5 * ((n_can + 1) ** (1 / ndim) - 1)
                weight_projected_dist = 0.5 * (n_can + 1)
                total_dist_scores = weight_euclidean_dist * min_intersite_dist
                total_dist_scores += weight_projected_dist * min_projected_dist

            elif self.mc_criterion == "mc-intersite-proj-th":
                alpha = 0.5  # chosen (tradeoff)
                d_min = 2 * alpha / n_can
                min_intersite_dist[min_projected_dist < d_min] = 0
                total_dist_scores = min_intersite_dist
            else:
                raise NameError("The MC-Criterion you requested is not available.")

            # Add the best candidate to the list of new samples
            best_candidate = np.argsort(total_dist_scores)[::-1][: self.n_candidates]
            selected_samples = np.vstack((selected_samples, candidates[best_candidate]))

        exploration_scores = np.delete(error_voronoi, bad_samples, axis=0)

        return selected_samples, exploration_scores

    # -------------------------------------------------------------------------
    def get_mc_samples(self, all_candidates=None):
        """
        This function generates random samples based on Global Monte Carlo
        methods and their corresponding scores, based on [1].

        [1] Crombecq, K., Laermans, E. and Dhaene, T., 2011. Efficient
            space-filling and non-collapsing sequential design strategies for
            simulation-based modeling. European Journal of Operational Research
            , 214(3), pp.683-696.
            DOI: https://doi.org/10.1016/j.ejor.2011.05.032

        Implemented methods to compute scores:
            1) mc-intersite-proj
            2) mc-intersite-proj-th

        Arguments
        ---------
        all_candidates : array, optional
            Samples to compute the scores for. The default is `None`. In this
            case, samples will be generated by defined model input marginals.

        Returns
        -------
        new_samples : array of shape (n_candidates, n_params)
            A list of samples.
        exploration_scores: arrays of shape (n_candidates)
            Exploration scores.
        """
        # Generate all_candidates if not given
        if all_candidates is None:
            explore_method = self.exp_design.explore_method
            all_candidates = self.exp_design.generate_samples(
                self.n_candidates, explore_method
            )
            n_candidates = self.n_candidates
        else:
            n_candidates = all_candidates.shape[0]

        # Get the old exp_design #samples
        old_ed_x = self.exp_design.x
        ndim = old_ed_x.shape[1]

        # Init
        max_min_distance = 0
        min_intersite_dist = np.zeros((n_candidates))
        min_projected_dist = np.zeros((n_candidates))

        # Goal: Find candidate that is farthest away from any existing sample
        for i, candidate in enumerate(all_candidates):
            # Find min distorted distance from all other samples
            euclidean_dist = self._build_dist_matrix_point(
                old_ed_x, candidate, do_sqrt=True
            )
            min_euclidean_dist = np.min(euclidean_dist)
            min_intersite_dist[i] = min_euclidean_dist

            # Check if this is the maximum minimum distance from all other
            # samples
            max_min_distance = max(min_euclidean_dist, max_min_distance)

            # Projected distance
            projected_dist = self._build_dist_matrix_point(
                old_ed_x, candidate, "chebyshev"
            )
            min_projected_dist[i] = np.min(projected_dist)

        if self.mc_criterion == "mc-intersite-proj":
            weight_euclidean_dist = ((n_candidates + 1) ** (1 / ndim) - 1) * 0.5
            weight_projected_dist = (n_candidates + 1) * 0.5
            total_dist_scores = weight_euclidean_dist * min_intersite_dist
            total_dist_scores += weight_projected_dist * min_projected_dist

        elif self.mc_criterion == "mc-intersite-proj-th":
            alpha = 0.5  # chosen (tradeoff)
            d_min = 2 * alpha / n_candidates
            min_intersite_dist[min_projected_dist < d_min] = 0
            total_dist_scores = min_intersite_dist
        else:
            raise NameError("The MC-Criterion you requested is not available.")

        exploration_scores = total_dist_scores
        exploration_scores /= np.nansum(total_dist_scores)

        return all_candidates, exploration_scores

    # -------------------------------------------------------------------------
    def approximate_voronoi(self, samples):
        """
        An approximate (monte carlo) version of Matlab's voronoi command.

        Arguments
        ---------
        samples : array
            Old experimental design to be used as center points for voronoi
            cells.

        Returns
        -------
        areas : array
            An approximation of the voronoi cells' areas.
        """
        n_samples = samples.shape[0]
        ndim = samples.shape[1]

        # Compute the number of random points
        n_points = self.w * samples.shape[0]
        # Generate w random points in the domain for each sample
        points = self.exp_design.generate_samples(n_points, "random")

        # Calculate the nearest sample to each point
        areas = np.zeros((n_samples))
        closest_points = [np.empty((0, ndim)) for i in range(n_samples)]

        # Compute the minimum distance from all the samples of old_ed_x for
        # each test point
        for idx in range(n_points):
            # Calculate the minimum distance
            distances = self._build_dist_matrix_point(
                samples, points[idx], do_sqrt=True
            )
            closest_sample = np.argmin(distances)

            # Add to the voronoi list of the closest sample
            areas[closest_sample] = areas[closest_sample] + 1
            prev_closest_points = closest_points[closest_sample]
            closest_points[closest_sample] = np.vstack(
                (prev_closest_points, points[idx])
            )

        # Save closest points
        self.closest_points = closest_points

        # Divide by the amount of points to get the estimated volume of each
        # voronoi cell
        areas /= n_points
        return areas

    # -------------------------------------------------------------------------
    def _build_dist_matrix_point(
        self, samples, point, method="euclidean", do_sqrt=False
    ):
        """
        Calculates the intersite distance of all points in samples from point.

        Parameters
        ----------
        samples : np.ndarray of shape (n_samples, n_params)
            The old experimental design.
        point : np.ndarray
            A candidate point, array has shape (n_params).
        method : str
            Distance method.
        do_sqrt : bool, optional
            Whether to return distances or squared distances. The default is
            `False`.

        Returns
        -------
        distances : array
            Distances.

        """
        distances = distance.cdist(samples, np.array([point]), method)

        # do square root?
        if do_sqrt:
            return distances
        return distances**2
