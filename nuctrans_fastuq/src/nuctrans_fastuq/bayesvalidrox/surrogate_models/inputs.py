#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inputs and related marginal distributions
"""


class Input:
    """
    A class to define the uncertain input parameters.

    Attributes
    ----------
    Marginals : obj
        Marginal objects. See `inputs.Marginal`.
    Rosenblatt : bool
        If Rossenblatt transformation is required for the dependent input
        parameters.

    Examples
    -------
    Marginals can be defined as following:

    >>> inputs = Inputs()
    >>> inputs.add_marginals()
    >>> inputs.marginals[0].name = 'X_1'
    >>> inputs.marginals[0].dist_type = 'uniform'
    >>> inputs.marginals[0].parameters = [-5, 5]

    If there is no common data is avaliable, the input data can be given
    as following:

    >>> inputs.add_marginals()
    >>> inputs.marginals[0].name = 'X_1'
    >>> inputs.marginals[0].input_data = [0,0,1,0]
    """

    poly_coeffs_flag = True

    def __init__(self):
        self.marginals = []
        self.rosenblatt = False

    def add_marginals(
        self, name="$x_1$", dist_type=None, parameters=None, input_data=None
    ):
        """
        Adds a new Marginal object to the input object.

        Parameters
        ----------
        name : string, optional
            Name of the parameter. The default is '$x_1$'
        dist_type : string, optional
            Type of distribution of the marginal.
            This parameter has to be given in combination with parameters.
            The default is None.
        parameters : list, optional
            List of parameters of distribution of type dist_type.
            The default is None.
        input_data : np.array, optional


        Returns
        -------
        None.

        """
        self.marginals.append(Marginal())
        idx = len(self.marginals) - 1
        self.marginals[idx].name = name
        self.marginals[idx].dist_type = dist_type
        self.marginals[idx].parameters = parameters
        self.marginals[idx].input_data = [] if input_data is None else input_data


# Nested class
class Marginal:
    """
    An object containing the specifications of the marginals for each uncertain
    parameter.

    Attributes
    ----------
    name : string
        Name of the parameter. The default is `'$x_1$'`.
    dist_type : string
        Name of the distribution. The default is `None`.
    parameters : list
        Parameters corresponding to the distribution type. The
        default is `None`.
    input_data : array
        Available input data. The default is `[]`.
    moments : list
        Moments of the distribution. The default is `None`.
    """

    def __init__(self):
        self.name = "$x_1$"
        self.dist_type = None
        self.parameters = None
        self.input_data = []
        self.moments = None
