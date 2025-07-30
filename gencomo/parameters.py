# parameters.py
# This module provides a framework for managing scientific parameters, including real-valued parameters
# with optional sampling, constraints, and parameter sets.

import numpy as np
from scipy import stats
import pandas as pd


class IndependentParameter:
    """A real-valued parameter with optional sampling and range.

    The value should be set at initialization and may not be changed later. Same for the range, if specified.
    If is_sampled is True, the parameter will have a uniform distribution over the specified range.
    Attributes:
        value (float): The value of the parameter.
        is_sampled (bool): Whether the parameter is sampled from a distribution.
        range (tuple[float, float] | None): The range of the parameter, if applicable.

    """

    def __init__(
        self,
        value: float,
        is_sampled: bool = False,
        range: tuple[float, float] | None = None,
    ):
        self._validate_value_and_range(value, range)
        self._value = value
        self._range = range
        self._is_sampled = is_sampled

        if is_sampled and (range is None or len(range) != 2):
            raise ValueError("If is_sampled is True, range must be a tuple of two numeric values.")

        if is_sampled and range is not None:
            self._distribution = stats.uniform(loc=self.range[0], scale=self.range[1] - self.range[0])
        else:
            self._distribution = None

    @property
    def value(self):
        """Get the value of the parameter."""
        return self._value

    @value.setter
    def value(self, _):
        raise AttributeError("Value cannot be changed after initialization.")

    @property
    def range(self):
        """Get the range of the parameter."""
        return self._range

    @range.setter
    def range(self, _):
        raise AttributeError("Range cannot be changed after initialization.")

    def __repr__(self):
        return f"IndependentParameter(value={self.value}, range={self.range}, is_sampled={self._is_sampled})"

    def __str__(self):
        return self.__repr__()

    def _validate_value_and_range(self, value, range):
        """Validate the value and range of the parameter."""
        if not isinstance(value, (int, float)):
            raise ValueError("Value must be a number.")
        if range is not None:
            if not isinstance(range, tuple) or len(range) != 2:
                raise ValueError("Range must be a tuple of two elements.")
            if not all(isinstance(x, (int, float)) for x in range):
                raise ValueError("Range must contain only numeric values.")
            if not (range[0] <= value <= range[1]):
                raise ValueError(f"Value {value} is not within the range {range}.")

    def sample(self, size: int | None = None):
        """Sample a value from the parameter's distribution."""
        return self._distribution.rvs(size=size)

    def copy(self):
        """Create a copy of the IndependentParameter."""
        return IndependentParameter(value=self.value, is_sampled=self._is_sampled, range=self.range)


class ParameterSet(pd.Series):
    """A class for parameter sets (aka 'instances'): a single collection of parameters with their scalar values.
    Mainly to be used as samples from a ParameterBank, but can also be used independently.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __repr__(self):
        return f"ParameterSet({super().__repr__()})"

    def satisfies(self, constraint):
        """Check if the constraint is satisfied for this parameter set.
        The constraint should be a callable that takes a ParameterSet
        and returns a boolean value: True if the constraint is satisfied.

        Maybe redundant, but useful for clarity.
        """
        if not callable(constraint):
            raise ValueError("Constraint must be a callable function.")
        result = constraint(self)
        if not isinstance(result, (bool, np.bool)):
            raise ValueError("Constraint function must return a boolean value.")
        return result

    def copy(self):
        """Create a copy of the ParameterSet."""
        return ParameterSet(self.to_dict())

    def reindex(self, new_index):
        """Reindex the ParameterSet to a new index."""
        if not isinstance(new_index, (list, tuple)):
            raise ValueError("New index must be a list or tuple of parameter names.")
        new_series = super().reindex(new_index)
        return ParameterSet(new_series)


class DerivedParameter:
    """A parameter derived from a function of other parameters."""

    def __init__(self, function: callable):
        self.function = function
        self._is_sampled = False  # Derived parameters are never considered sampled.
        if not callable(self.function):
            raise ValueError("Function must be callable.")

    def __repr__(self):
        return f"DerivedParameter(function={self.function.__name__})"

    def compute(self, parameters: ParameterSet):
        """Compute the value of the derived parameter based on the provided parameters."""
        if not isinstance(parameters, ParameterSet):
            raise ValueError("Parameters must be an instance of ParameterSetInstance.")
        if not callable(self.function):
            raise ValueError("Function must be callable.")
        return self.function(parameters)

    def copy(self):
        """Create a copy of the DerivedParameter."""
        return DerivedParameter(function=self.function)


class ParameterBank:
    """A bank of parameters, allowing for collective sampling, and constraints.
    Parameters can be either IndependentParameter or DerivedParameter.
    A DerivedParameter should depend on IndependentParameters in the bank, not other DerivedParameters.

    Constraints can be added to ensure that sampled parameters meet certain conditions.
    """

    def __init__(
        self,
        parameters: dict[str, IndependentParameter | DerivedParameter] = None,
        constraints: list[callable] = None,
        theta_sampling: bool = False,
        texnames: dict[str, str] = None,
    ):
        self.parameters = parameters if parameters is not None else {}
        self.constraints = constraints if constraints is not None else []
        self.theta_sampling = theta_sampling
        self._max_attempts = 100  # Default maximum attempts for sampling with constraints
        self.texnames = texnames if texnames is not None else {}

        for key, value in self.parameters.items():
            if not isinstance(value, IndependentParameter) and not isinstance(value, DerivedParameter):
                raise ValueError(
                    f"Value for key '{key}' must be an instance of IndependentParameter or DerivedParameter."
                )

        # compute indices of sampled parameters in the canonical order
        self.sampled_indices = [
            self.names.index(key)
            for key, param in self.parameters.items()
            if isinstance(param, IndependentParameter) and param._is_sampled
        ]

    @property
    def names(self):
        """Get the names of all parameters in the bank.
        This also defines the canonical order of the parameters."""
        return list(self.parameters.keys())

    @property
    def sampled(self):
        """Get a list of all parameters that are set to be sampled."""
        return [key for key, param in self.parameters.items() if param._is_sampled]

    @property
    def lower_bounds(self):
        """Get the lower bounds of all sampled parameters."""
        return np.array(
            [
                param.range[0]
                for key, param in self.parameters.items()
                if isinstance(param, IndependentParameter) and param._is_sampled
            ]
        )

    @property
    def upper_bounds(self):
        """Get the upper bounds of all sampled parameters."""
        return np.array(
            [
                param.range[1]
                for key, param in self.parameters.items()
                if isinstance(param, IndependentParameter) and param._is_sampled
            ]
        )

    @property
    def sampled_texnames(self):
        """Get the TeX names of all sampled parameters."""
        return [self.texnames.get(key, key) for key in self.sampled]

    def __repr__(self):
        return f"ParameterBank(parameters={self.parameters}, constraints={self.constraints})"

    def __contains__(self, key):
        """Check if a parameter exists in the bank."""
        return key in self.parameters

    def __len__(self):
        """Get the number of parameters in the bank."""
        return len(self.parameters)

    def __iter__(self):
        """Iterate over the parameter names in the bank."""
        return iter(self.parameters)

    def __getitem__(self, key):
        """Get a parameter by its name."""
        if key in self.parameters:
            return self.parameters[key]
        else:
            raise KeyError(f"Parameter '{key}' not found in the bank.")

    def __setitem__(self, _):
        raise AttributeError("Parameters cannot be set after initialization.")

    def __delitem__(self, key):
        raise AttributeError("Parameters cannot be deleted from the bank.")

    def copy(self):
        """Create a shallow copy of the ParameterBank."""
        new_bank = ParameterBank()
        for key, value in self.items():
            new_bank[key] = value.copy()
        new_bank.constraints = self.constraints.copy()
        return new_bank

    def merge(self, other: "ParameterBank"):
        """Merge another ParameterBank into this one.
        If a parameter with the same name exists, it will be overwritten.
        """
        if not isinstance(other, ParameterBank):
            raise ValueError("Other must be an instance of ParameterBank.")
        for key, value in other.parameters.items():
            self.parameters[key] = value.copy()
        self.constraints.extend(other.constraints)

    def add_parameter(self, key: str, parameter: IndependentParameter | DerivedParameter):
        """Add a new parameter to the bank."""
        # if not isinstance(parameter, (IndependentParameter, DerivedParameter)):
        #     raise ValueError("Parameter must be an instance of IndependentParameter or DerivedParameter.")
        # if key in self.parameters:
        #     raise KeyError(f"Parameter '{key}' already exists in the bank.")
        # self.parameters[key] = parameter
        raise NotImplementedError(
            "Adding parameters is not implemented. Use the constructor to initialize the bank with parameters."
        )

    def add_constraint(self, constraint: callable):
        """Add a new constraint to the bank."""
        if not callable(constraint):
            raise ValueError("Constraint must be a callable function.")
        self.constraints.append(constraint)

    def get_constraints(self):
        """Get all constraints in the bank."""
        return self.constraints

    def get_default_values(self, return_theta=None):
        """Get the default values for all parameters in the bank.

        return_theta: If True, return a numpy array. If False, return a ParameterSet instance.
        """
        if return_theta is None:
            return_theta = self.theta_sampling  # default to self.theta_sampling if not specified
        if not isinstance(return_theta, bool):
            raise ValueError("theta_array must be a boolean value.")
        p = ParameterSet(
            {key: param.value for key, param in self.parameters.items() if isinstance(param, IndependentParameter)}
        )
        p = ParameterSet(
            {
                **p,
                **{
                    key: param.compute(p)
                    for key, param in self.parameters.items()
                    if isinstance(param, DerivedParameter)
                },
            }
        )
        p = self.order(p)
        if return_theta:
            return self.instance_to_theta(p)
        else:
            return p

    def instance_to_theta(self, input: ParameterSet | list[ParameterSet]):
        """Convert a ParameterSetInstance to a numpy array of sampled values."""
        if not isinstance(input, (ParameterSet, list)):
            raise ValueError("Input must be a ParameterSetInstance or a list of ParameterSetInstances.")
        if isinstance(input, ParameterSet):
            theta = np.array([input[key] for key in self.sampled])
        else:
            theta = np.array([instance[key] for instance in input for key in self.sampled])
        return theta

    def dataframe_to_theta(self, df: pd.DataFrame):
        """Convert a pandas DataFrame of sampled values to a numpy array."""
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")
        theta = df[self.sampled].to_numpy()
        return theta

    def theta_to_instance(self, theta: np.ndarray):
        """Convert a numpy array vector of sampled values to a ParameterSetInstance."""
        if not isinstance(theta, np.ndarray):
            raise ValueError("Input must be a numpy array, instead got: " + str(type(theta)))
        if len(theta) != len(self.sampled):
            raise ValueError(
                f"Array length {len(theta)} does not match number of sampled parameters {len(self.sampled)}."
            )  # ensure the length matches the number of sampled parameters
            # theta in this case must be a 1D array
        out = self.get_default_values(return_theta=False)  # start with default values
        # fill in the sampled parameters
        for i, key in enumerate(self.sampled):  # iterate over names of sampled parameters
            out[key] = theta[i]
        # compute derived parameters based on the sampled independent parameters
        out = ParameterSet(
            {
                **out,
                **{
                    key: param.compute(out)
                    for key, param in self.parameters.items()
                    if isinstance(param, DerivedParameter)
                },
            }
        )
        return out

    def _sample_once(self):
        """Sample a single set of parameters from the bank."""
        # first, sample all independent parameters that are set to be sampled
        p = ParameterSet(
            {
                key: param.sample()
                for key, param in self.parameters.items()
                if (param._is_sampled and isinstance(param, IndependentParameter))
            }
        )
        # then, compute all derived parameters based on the sampled independent parameters
        p = ParameterSet(
            {
                **p,
                **{
                    key: param.compute(p)
                    for key, param in self.parameters.items()
                    if isinstance(param, DerivedParameter)
                },
            }
        )
        # put result in canonical order according to self.canonical_order
        p = self.order(p)
        return p

    def _sample_once_constrained(self):
        """Sample a single set of parameters from the bank, satisfying a constraint function."""
        attempts = 0
        while attempts < self._max_attempts:
            attempts += 1
            sample = self._sample_once()
            # Check if the sample meets all constraints
            if all(sample.satisfies(c) for c in self.constraints):
                return sample

    def sample(self, size: int | tuple = None):
        """Sample a set of parameters from the bank."""
        if size is not None and not isinstance(size, int) and not isinstance(size, tuple):
            raise ValueError("Size must be None, an integer, or a tuple.")
        if size is None:
            n_samples = 1
        elif isinstance(size, int):
            n_samples = size
        elif isinstance(size, tuple):
            if len(size) > 1 and not self.theta_sampling:
                raise ValueError("Multiple dimensions are only supported for theta_sampling.")
            if len(size) == 1:
                n_samples = size[0]
            else:
                n_samples = int(np.prod(size))

        # print("n_samples (type):", n_samples, type(n_samples))
        # print("size (type):", size, type(size))
        samples = []
        for _ in range(n_samples):
            if self.constraints:
                sample = self._sample_once_constrained()
            else:
                sample = self._sample_once()
            # add any parameters that are not sampled but are required for the model
            sample = ParameterSet(
                {
                    **sample,
                    **{
                        key: param.value
                        for key, param in self.parameters.items()
                        if not param._is_sampled and not isinstance(param, DerivedParameter)
                    },
                }
            )
            samples.append(sample)
        # print("After sampling, there are", len(samples), "samples.")
        if self.theta_sampling:
            # out = np.array([self.instance_to_theta(sample) for sample in samples])
            if size is None:
                out = self.instance_to_theta(samples[0])
            elif isinstance(size, int):
                out = np.array([self.instance_to_theta(sample) for sample in samples]).reshape(
                    (size, len(self.sampled))
                )
            elif isinstance(size, tuple):
                out = np.array([self.instance_to_theta(sample) for sample in samples]).reshape(
                    size + (len(self.sampled),)
                )
        else:
            if size is None:
                out = samples[0]
            elif isinstance(size, int):
                out = self.instances_to_dataframe([sample for sample in samples])
            elif isinstance(size, tuple):
                out = self.instances_to_dataframe([sample for sample in samples])
        return out

    def instances_to_dataframe(self, instances: list[ParameterSet]):
        """Convert a list of ParameterSetInstances to a pandas DataFrame."""
        if not isinstance(instances, list):
            raise ValueError("Instances must be a list of ParameterSetInstance objects.")
        if not instances:
            raise ValueError("Instances list cannot be empty.")
        if not all(isinstance(instance, ParameterSet) for instance in instances):
            raise ValueError("All items in instances must be ParameterSetInstance objects.")
        df = pd.DataFrame([instance for instance in instances])
        df = df.astype(float)
        return df

    def log_prob(self, input: ParameterSet | pd.DataFrame | np.ndarray):
        """Calculate the log prior probability of a list of samples.
        Input can be a list of ParameterSet instances or a numpy array.
        In the case of a numpy array, it should be 1D or 2D, where each row is a sample.

        Returns 0 if within bounds, or -inf if the sample is outside the bounds.
        """
        # categorize inputs
        if isinstance(input, ParameterSet):  # if a single sample, package it in a list
            samples = [input]
        elif isinstance(input, pd.DataFrame):  # if a DataFrame, convert to list of ParameterSet instances
            samples = [ParameterSet(row) for _, row in input.iterrows()]
        elif isinstance(input, np.ndarray):  # if numpy array ...
            if input.ndim == 1:  # if 1D, treat as a single sample
                if (
                    input.shape[0] != len(self.sampled) and self.theta_sampling
                ):  # if theta_sampling is enabled, sample must match sampled parameters
                    raise ValueError(
                        f"1D numpy array must have length {len(self.sampled)} to match sampled parameters, since theta_sampling is enabled."
                    )
                elif (
                    input.shape[0] != len(self.parameters) and not self.theta_sampling
                ):  # if theta_sampling is disabled, sample must match all parameters
                    raise ValueError(
                        f"1D numpy array must have length {len(self.parameters)} to match all parameters, since theta_sampling is disabled."
                    )
                # print("Converting 1D numpy array to ParameterSet instance.")
                samples = [self.theta_to_instance(input)]  # convert to ParameterSet
            elif input.ndim == 2:  # if 2D, treat each row as a sample
                if input.shape[1] != len(self.sampled) and self.theta_sampling:
                    raise ValueError(
                        f"2D numpy array must have {len(self.sampled)} columns to match sampled parameters, since theta_sampling is enabled."
                    )
                elif input.shape[1] != len(self.parameters) and not self.theta_sampling:
                    raise ValueError(
                        f"2D numpy array must have {len(self.parameters)} columns to match all parameters, since theta_sampling is disabled."
                    )
                # print("Converting 2D numpy array to list of ParameterSet instances.")
                samples = [self.theta_to_instance(row) for row in input]
            else:
                raise ValueError("Samples must be a 1D or 2D numpy array.")
        elif not isinstance(input, list):
            raise ValueError("Samples must be a list of ParameterSet instances or a numpy array.")

        results = np.zeros(len(samples))
        for i, sample in enumerate(samples):
            results[i] = self._log_prob_single(sample)
        if len(results) == 1 and isinstance(input, ParameterSet):
            return results[0]
        else:
            return results

    def _log_prob_single(self, sample: ParameterSet):
        """Calculate the log prior probability of a single sample."""
        if sample is None or not isinstance(sample, ParameterSet):
            raise ValueError("Sample must be an instance of ParameterSet.")
        result = 0.0
        for key, param in self.parameters.items():
            if isinstance(param, IndependentParameter) and param._is_sampled:
                if not (param.range[0] <= sample[key] <= param.range[1]):
                    result = -np.inf
                if not all(sample.satisfies(c) for c in self.constraints):
                    result = -np.inf
        return result

    def order(self, instance: ParameterSet):
        """Reorder the parameters in the instance according to the canonical order."""
        if not isinstance(instance, ParameterSet):
            raise ValueError("Input must be an instance of ParameterSet.")
        try:
            out = instance.reindex(self.names)
        except Exception as e:
            raise ValueError("Error reordering parameters: " + str(e))
        return out

    # def make_torch_distribution(self):
    #     import torch
    #     from sbi.utils.torchutils import BoxUniform

    #     class ConstrainedBoxUniform(BoxUniform):
    #         def __init__(self, low, high, constraint=None, max_trials=1000):
    #             super().__init__(low, high)
    #             self.constraint = constraint
    #             self.max_trials = max_trials

    #         def sample(self, sample_shape=None) -> torch.Tensor:
    #             if sample_shape is None:
    #                 sample_shape = (1,)
    #             needed = sample_shape[0]
    #             samples = []
    #             total = 0
    #             trials = 0
    #             while total < needed and trials < self.max_trials:
    #                 batch = super().sample((needed - total,))
    #                 if self.constraint is not None:
    #                     mask = self.constraint(batch)
    #                     batch = batch[mask.bool()]
    #                 if batch.numel() > 0:
    #                     samples.append(batch)
    #                     total += batch.shape[0]
    #                 trials += 1
    #             if samples:
    #                 samples = torch.cat(samples, dim=0)
    #             else:
    #                 samples = torch.empty((0, self.low.shape[0]))
    #             if samples.shape[0] > needed:
    #                 samples = samples[:needed]
    #             if samples.shape[0] < needed:
    #                 raise RuntimeError("Could not sample enough constrained points.")
    #             return samples

    #     lows = np.array(
    #         [
    #             param.range[0]
    #             for param in self.parameters.values()
    #             if isinstance(param, IndependentParameter) and param._is_sampled
    #         ]
    #     )
    #     highs = np.array(
    #         [
    #             param.range[1]
    #             for param in self.parameters.values()
    #             if isinstance(param, IndependentParameter) and param._is_sampled
    #         ]
    #     )
    #     out = ConstrainedBoxUniform(
    #         torch.tensor(lows, dtype=torch.float32),
    #         torch.tensor(highs, dtype=torch.float32),
    #         constraint=lambda x: torch.tensor(
    #             [
    #                 param.satisfies(c)
    #                 for c in self.constraints
    #                 for param in self.parameters.values()
    #                 if isinstance(param, IndependentParameter)
    #             ]
    #         ),
    #         max_trials=self._max_attempts,
    #     )
    #     return out
