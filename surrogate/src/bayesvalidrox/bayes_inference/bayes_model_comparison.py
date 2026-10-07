#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Class for Bayesian Model Selection and comparison.
"""

import os
import platform
import copy
from tqdm import tqdm
from scipy import stats
import numpy as np
import seaborn as sns
from matplotlib import patches
import matplotlib.pylab as plt
import pandas as pd
from .bayes_inference import BayesInference

# Load the mplstyle
plt.style.use(os.path.join(os.path.split(__file__)[0], "../", "bayesvalidrox.mplstyle"))


class BayesModelComparison:
    """
    A class to perform Bayesian multi-model comparison.

    Attributes
    ----------
    model_dict : dict
        A dictionary of model names and bvr.engine objects for each model.
    bayes_opts : dict
        A dictionary given the `BayesInference` options.
    n_bootstrap : int, optional
        Number of samples generated from each engine for the confusion matrix.
        The default is `1000`.
    use_emulator : bool, optional
        Set to True if the emulator/metamodel should be used in the analysis.
        If False, the model is run. The default is True.
    plot : bool, optional
        Toggles plots of the BMC results.
        The default is True.
    out_dir : string, optional
        Name of output folder for the generated plots. The default
        is './Outputs_Comparison/'.
    out_format : str, optional
        Format that the generated plots are stored as. Supports 'pdf' and 'png'.
        The default is 'pdf'.

    """

    def __init__(
        self,
        model_dict,
        bayes_opts,
        n_bootstrap=1000,
        use_emulator=True,
        plot=True,
        out_dir="./Outputs_Comparison/",
        out_format="pdf",
    ):
        # Inputs
        self.model_dict = model_dict
        self.bayes_opts = bayes_opts
        self.n_bootstrap = n_bootstrap
        self.use_emulator = use_emulator
        self.plot = plot
        self.out_dir = out_dir
        self.out_format = out_format

        # Other parameters
        self.model_names = None
        self.n_meas = None
        self.bme_dict = {}
        self.dtype = None

    # --------------------------------------------------------------------------
    def setup(self):
        """
        Initialize parameters that are needed for all types of model comparison
        """
        # Check model_dict
        if not isinstance(self.model_dict, dict):
            raise AttributeError(
                "To run model comparsion, you need to pass a dictionary of models."
            )

        # Check bayes_opts
        if not isinstance(self.bayes_opts, dict):
            raise AttributeError(
                "Give the options for Bayesian inference as a dictionary."
            )
        if not "observation" in self.bayes_opts:
            raise AttributeError("No observation found in self.bayes_opts.")

        # Extract model names
        self.model_names = [*self.model_dict]

        # Output directory
        os.makedirs(self.out_dir, exist_ok=True)

        # System settings
        if platform.system() == "Windows" or platform.system() == "Darwin":
            print("")
            print(
                "WARNING: Performing the inference on windows or MacOS \
                    can lead to reduced accuracy!"
            )
            print("")
            self.dtype = np.longdouble
        else:
            self.dtype = np.float128

    # --------------------------------------------------------------------------
    def model_comparison_all(self) -> dict:
        """
        Performs all three types of model comparison:
            * Calculation of Bayes Factors
            * Calculation of Model weights
            * Justifiability analysis via confusion matrix

        Returns
        -------
        results : dict
            A dictionary that contains the calculated BME values, model weights
            and confusion matrix

        """
        bme_dict = self.calc_bayes_factors()
        model_weights = self.calc_model_weights()
        confusion_matrix = self.calc_justifiability_analysis()

        results = {
            "BME": bme_dict,
            "Model weights": model_weights,
            "Confusion matrix": confusion_matrix,
        }
        return results

    # --------------------------------------------------------------------------
    def calc_bayes_factors(self) -> dict:
        """
        Calculate the BayesFactors for each pair of models in the model_dict
        with respect to given data.

        Returns
        -------
        bme_dict : dict
            The calculated BME values for each model

        """
        # Run the setup
        self.setup()

        # Run Bayesian validation for each model
        observation = self.bayes_opts["observation"]
        for model in self.model_names:
            print("-" * 20)
            print(f"Bayesian inference of {model}.\n")
            bayes = BayesInference(self.model_dict[model], observation, plot=False)

            # Set BayesInference options
            if self.bayes_opts is not None:
                for key, value in self.bayes_opts.items():
                    if key in bayes.__dict__:
                        setattr(bayes, key, value)
            bayes.use_emulator = self.use_emulator

            log_bme = bayes.run_validation()
            self.bme_dict[model] = np.exp(log_bme, dtype=self.dtype)
            print("-" * 20)

        # Create kde plot for bayes factors
        if self.plot:
            self.plot_bayes_factor(self.bme_dict)
        return self.bme_dict

    def calc_model_weights(self) -> dict:
        """
        Calculate the model weights from BME evaluations.

        Returns
        -------
        model_weights : dict
            The calculated weights for each model

        """
        # Get BMEs via Bayes Factors if not already calculated
        bme = self.bme_dict
        if len(list(bme.keys())) == 0:
            bme = self.calc_bayes_factors()

        # Stack the BME values for all models
        all_bme = np.vstack(list(bme.values()))

        # Model weights
        model_weights_arr = np.divide(all_bme, np.nansum(all_bme, axis=0))
        model_weights = {}
        for m_id, m_name in enumerate(self.model_names):
            model_weights[m_name] = model_weights_arr[m_id]

        # Create box plot for model weights
        if self.plot:
            self.plot_model_weights(model_weights_arr)
        return model_weights

    # -------------------------------------------------------------------------
    def calc_justifiability_analysis(self) -> dict:
        """
        Perform justifiability analysis by calculating the confusion matrix

        Returns
        -------
        confusion_matrix: dict
            The averaged confusion matrix.

        """
        # Run the setup
        self.setup()

        # Extend model names
        model_names = self.model_names
        if model_names[0] != "Observation":
            model_names.insert(0, "Observation")
        n_models = len(model_names)

        # Generate datasets of model evaluations and stdevs
        just_list, var_list, bayes = self.generate_ja_dataset()
        n_perturb = bayes.n_bootstrap_itrs
        sampler = bayes.sampler

        # Start and end of each model in likelihood matrix
        start = []
        end = []
        for i in range(n_models):
            if i == 0:
                start.append(0)
                end.append(n_perturb)
            else:
                start.append(end[i - 1])
                end.append(end[i - 1] + self.n_bootstrap)

        # Calculate Likelihood + Posterior of each column against each column
        total_n_runs = n_perturb + (n_models - 1) * self.n_bootstrap
        likelihoods = np.zeros((total_n_runs, total_n_runs))
        for i in range(n_models):
            for j in range(n_models):
                obs = copy.deepcopy(just_list[i])
                var = copy.deepcopy(var_list[i])
                valid_keys = [k for k in obs if k != "x_values"]
                if not valid_keys:
                    raise AttributeError(
                        "No valid output keys found in observation dictionary!"
                    )

                for cnt in tqdm(
                    range(end[i] - start[i]), desc=f"Likelihood model {i} vs {j}"
                ):
                    for out in valid_keys:
                        obs[out] = np.array(just_list[i][out][cnt, :])
                        var[out] = np.array(var_list[i][out][cnt, :])
                    sampler.observation.data = obs
                    sampler.observation.sigma2 = var

                    loglik = sampler.normpdf(
                        just_list[j], rmse=None, std_outputs=var_list[j], norm=False
                    )
                    likelihoods[start[i] + cnt, start[j] : end[j]] = np.exp(loglik)

        # Accumulate into posterior matrix
        accum_lik = np.zeros((n_models, n_models))
        sum_lik = np.sum(likelihoods, axis=1)
        for mod1 in range(n_models):
            for mod2 in range(n_models):
                accum_lik[mod1, mod2] = np.mean(
                    likelihoods[start[mod1] : end[mod1], start[mod2] : end[mod2]]
                    / sum_lik[start[mod2] : end[mod2]]
                )

        # Correction for surrogate
        if self.use_emulator:
            print("Correct for surrogates")
            correction_factor = np.zeros(accum_lik.shape[0])
            for i, name in enumerate(self.model_names):
                if i == 0:
                    correction_factor[i] = 1
                    continue
                engine = self.model_dict[name]
                train_mean, _ = engine.eval_metamodel(engine.exp_design.x)
                # Correction factor returned as loglikelihood
                correction_factor[i] = np.ravel(sampler.corr_factor_bme(
                    engine.exp_design.y, train_mean, None, norm=False))[0]

            # Sice it is multiplied, take exp of the corr_factor (lik from loglik)
            correction_matrix = np.ones((n_models, n_models))
            for i in range(n_models):
                correction_matrix[i, :] *= np.exp(correction_factor[i])
                correction_matrix[:, i] *= np.exp(correction_factor[i])
            accum_lik = np.multiply(accum_lik, correction_matrix)

        # Norm each 'generated by' column to sum 1
        just_model_weights = accum_lik
        for i in range(just_model_weights.shape[0]):
            just_model_weights[i, :] /= np.sum(just_model_weights[i, :])

        # Confusion matrix over all measurement points
        confusion_matrix = pd.DataFrame()
        confusion_matrix["Generated by"] = model_names
        for i, _ in enumerate(model_names):  # 'Associated to'
            confusion_matrix[model_names[i]] = just_model_weights[:, i]

        # Plot model weights
        self.plot_confusion_matrix(confusion_matrix)
        return confusion_matrix

    # -------------------------------------------------------------------------
    def generate_ja_dataset(self) -> tuple[list, list, object]:
        """
        Generates the data set for the justifiability analysis.

        Returns
        -------
        just_list: list
            List of the model outputs for each of the given models, as well as the
            perturbed observations.
        std_list : list
            List of the uncertainty/stdev associated with each model output and
            perturbed observation.
        bayes : object
            bvr.BayesInference object.

        """

        # Perturb observations
        observation = self.bayes_opts["observation"]
        bayes = BayesInference(self.model_dict[self.model_names[1]], observation)
        if self.bayes_opts is not None:
            for key, value in self.bayes_opts.items():
                if key in bayes.__dict__:
                    setattr(bayes, key, value)
        bayes.use_emulator = self.use_emulator

        # Perturb observations for Bayes Factor
        bayes.setup()
        perturbed_data = bayes.perturbed_data
        if len(perturbed_data.shape) != 2:
            perturbed_data = bayes.perturb_data(bayes.out_names)
        n_perturb = perturbed_data.shape[0]
        sigma2 = bayes.observation.sigma2

        std_list = [{}]
        for out in sigma2.keys():
            if out == "x_values":
                continue
            std_list[0][out] = np.array([sigma2[out]] * n_perturb)

        # Transform perturbed data into model output format
        just_list = [{}]
        cnt = 0
        for out, value in std_list[0].items():
            if out == "x_values":
                continue
            n_t = value.shape[1]
            just_list[0][out] = perturbed_data[:, n_t * cnt : n_t * (cnt + 1)]
            cnt += 1

        # Generate MC evaluations of the models
        for key, engine in self.model_dict.items():
            y_hat, y_std = engine.eval_metamodel(nsamples=self.n_bootstrap)
            just_list.append(y_hat)

            # Set stdev to 0 if not given
            if y_std is None:
                y_std = copy.deepcopy(y_hat)
                for key in y_hat:
                    if key != "x_values":
                        y_std[key] *= 0
            std_list.append(y_std)

        return just_list, std_list, bayes

    # -------------------------------------------------------------------------
    def plot_confusion_matrix(self, confusion_matrix):
        """
        Visualizes the confusion matrix and the model weights for the
        justifiability analysis.

        Parameters
        ----------
        confusion_matrix: pd.DataFrame
            The averaged confusion matrix.

        """
        model_names = [model.replace("_", "$-$") for model in self.model_names]

        # Plot the averaged confusion matrix
        cf_numpy = confusion_matrix[model_names].to_numpy()
        heatmap = sns.heatmap(
            cf_numpy.T,
            annot=True,
            cmap="Blues",
            xticklabels=model_names,
            yticklabels=model_names,
            annot_kws={"size": 24},
        )

        heatmap.xaxis.tick_top()
        heatmap.xaxis.set_label_position("top")
        heatmap.set_xlabel(r"\textbf{Data generated by:}", labelpad=15)
        heatmap.set_ylabel(r"\textbf{Model weight for:}", labelpad=15)
        heatmap.figure.savefig(
            f"{self.out_dir}confusion_matrix.{self.out_format}", bbox_inches="tight"
        )
        plt.close()

    # -------------------------------------------------------------------------
    def plot_model_weights(self, model_weights):
        """
        Visualizes the model weights resulting from BMS via the observation
        data.

        Parameters
        ----------
        model_weights : array
            Model weights.

        """
        # Create figure
        fig, ax = plt.subplots()
        font_size = 40

        # Filter data using np.isnan
        mask = ~np.isnan(model_weights.T)
        filtered_data = [d[m] for d, m in zip(model_weights, mask.T)]

        # Create the boxplot
        box_plt = ax.boxplot(filtered_data, patch_artist=True, showfliers=False)

        # change outline color, fill color and linewidth of the boxes
        for box in box_plt["boxes"]:
            # change outline color
            box.set(color="#7570b3", linewidth=4)
            # change fill color
            box.set(facecolor="#1b9e77")

        # change color and linewidth of the whiskers
        for whisker in box_plt["whiskers"]:
            whisker.set(color="#7570b3", linewidth=2)

        # change color and linewidth of the caps
        for cap in box_plt["caps"]:
            cap.set(color="#7570b3", linewidth=2)

        # change color and linewidth of the medians
        for median in box_plt["medians"]:
            median.set(color="#b2df8a", linewidth=2)

        # Customize the axes
        model_names = [model.replace("_", "$-$") for model in self.model_names]
        ax.set_xticklabels(model_names)
        ax.set_ylabel("Weight", fontsize=font_size)
        ax.set_ylim((-0.05, 1.05))
        for t in ax.get_xticklabels():
            t.set_fontsize(font_size)
        for t in ax.get_yticklabels():
            t.set_fontsize(font_size)

        # Title
        plt.title("Posterior Model Weights")

        # Save the figure
        fig.savefig(
            f"./{self.out_dir}model_weights.{self.out_format}", bbox_inches="tight"
        )

        plt.close()

    # -------------------------------------------------------------------------
    def plot_bayes_factor(self, bme_dict):
        """
        Plots the Bayes factor distibutions in a :math:`N_m \\times N_m`
        matrix, where :math:`N_m` is the number of the models.

        Parameters
        ----------
        bme_dict : dict
            A dictionary containing the BME values of the models.

        """
        # Plot setup
        font_size = 40
        colors = ["blue", "green", "gray", "brown"]
        model_names = list(bme_dict.keys())
        n_models = len(model_names)

        # Plots
        _, axes = plt.subplots(nrows=n_models, ncols=n_models, sharex=True, sharey=True)
        for i, key_i in enumerate(model_names):
            for j, key_j in enumerate(model_names):
                ax = axes[i, j]
                # Set size of the ticks
                for t in ax.get_xticklabels():
                    t.set_fontsize(font_size)
                for t in ax.get_yticklabels():
                    t.set_fontsize(font_size)

                if j != i:
                    # Null hypothesis: key_j is the better model
                    bayes_factor = np.log10(np.divide(bme_dict[key_i], bme_dict[key_j]))

                    # Plot single if only one bme value is given
                    if bayes_factor.shape[0] == 1:
                        ax.axvline(
                            x=bayes_factor[0], ymin=0, color=colors[i], linewidth=4
                        )

                    # Estimate KDE if multiple points are given
                    else:

                        # Taken from seaborn's source code (utils.py and
                        # distributions.py)
                        def seaborn_kde_support(data, bw_, gridsize, cut, clip):
                            if clip is None:
                                clip = (-np.inf, np.inf)
                            support_min = max(data.min() - bw_ * cut, clip[0])
                            support_max = min(data.max() + bw_ * cut, clip[1])
                            return np.linspace(support_min, support_max, gridsize)

                        kde_estim = stats.gaussian_kde(bayes_factor, bw_method="scott")

                        # Linearization of data, mimics seaborn's internal code
                        bw_ = kde_estim.scotts_factor() * np.std(bayes_factor)
                        linearized = seaborn_kde_support(
                            bayes_factor, bw_, 100, 3, None
                        )

                        # Compute values of the estimated function on the
                        # estimated linearized inputs
                        z_eval = kde_estim.evaluate(linearized)

                        # https://stackoverflow.com/questions/29661574/normalize-
                        # numpy-array-columns-in-python
                        def normalize(x):
                            return (x - x.min(0)) / np.ptp(x) #x.ptp(0)

                        # Normalize so it is between 0;1
                        z_2 = normalize(z_eval)
                        ax.plot(linearized, z_2, "-", color=colors[i], linewidth=4)
                        ax.fill_between(linearized, 0, z_2, color=colors[i], alpha=0.25)

                    ### Draw BF significant levels according to Jeffreys 1961
                    # Middle line
                    ax.axvline(x=0, ymin=0, linewidth=4, color="black")
                    # Strong evidence for both models
                    weak = np.log10(3)
                    ax.axvspan(-weak, weak, color="yellow", alpha=0.15)
                    # ax.axvline(x=np.log10(3), ymin=0, linewidth=4, color="dimgrey")
                    # Strong evidence for one model
                    strong = np.log10(10)
                    ax.axvspan(weak, strong, color="orange", alpha=0.3)
                    ax.axvspan(-strong, -weak, color="orange", alpha=0.3)

                    # ax.axvline(x=np.log10(10), ymin=0, linewidth=4, color="orange")
                    # Decisive evidence for one model
                    decicive = np.log10(100)
                    ax.axvspan(strong, decicive, color="red", alpha=0.3)
                    ax.axvspan(-strong, -strong, color="red", alpha=0.3)
                    # ax.axvline(x=np.log10(100), ymin=0, linewidth=4, color="r")

                    bf_label = (
                        key_i.replace("_", "$-$") + "/" + key_j.replace("_", "$-$")
                    )
                    legend_elements = [
                        patches.Patch(
                            facecolor=colors[i],
                            edgecolor=colors[i],
                            label=f"BF({bf_label})",
                        )
                    ]
                    ax.legend(
                        loc="upper left",
                        handles=legend_elements,
                        fontsize=font_size - (n_models + 1) * 5,
                    )

                elif j == i:
                    # Build a rectangle in axes coords
                    left, width = 0, 1
                    bottom, height = 0, 1

                    patch_rect = patches.Rectangle(
                        (left, bottom),
                        width,
                        height,
                        color="white",
                        fill=True,
                        transform=ax.transAxes,
                        clip_on=False,
                    )
                    ax.grid(False)
                    ax.add_patch(patch_rect)
                    fsize = font_size + 20 if n_models < 4 else font_size
                    ax.text(
                        0.5,
                        0.5,
                        key_i.replace("_", "$-$"),
                        horizontalalignment="center",
                        verticalalignment="center",
                        fontsize=fsize,
                        color=colors[i],
                        transform=ax.transAxes,
                    )

        # Customize axes
        custom_ylim = (0, 1.05)
        plt.setp(axes, ylim=custom_ylim)

        # Set labels
        for i in range(n_models):
            axes[-1, i].set_xlabel("log$_{10}$(BF)", fontsize=font_size)
            axes[i, 0].set_ylabel("Probability", fontsize=font_size)

        # Adjust subplots
        plt.subplots_adjust(wspace=0.2, hspace=0.1)
        plt.savefig(
            f"./{self.out_dir}Bayes_Factor_kde.{self.out_format}",
            bbox_inches="tight",
        )
        plt.close()
