import numpy as np
from graph import Graph
from itertools import zip_longest


def process_data(data):
    """
    Process input data to ensure it is of a valid type (int, float, list, np.ndarray),
    converts it to a numpy array, and ensures all elements are of type float.

    Parameters:
    - data: int, float, list, np.ndarray
        Input data to be processed.

    Returns:
    - np.ndarray
        Processed and converted numpy array.
    """
    supported_types = (int, float, list, np.ndarray)
    if type(data) in supported_types:
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        try:
            data = data.astype(float)
        except ValueError:
            raise TypeError("Data elements must be either of type float or convertible to float")
    else:
        raise TypeError(f"Expected data of types {supported_types}, but received data of type {type(data)}")
    return data


def unbroadcast_data(data, origin_data_shape, broadcasted_shape):
    """
    Unbroadcasts data by summing along axes that were previously broadcasted.

    Parameters:
    - data: np.ndarray
        Input data.
    - origin_data_shape: tuple
        Original shape of the data before broadcasting.
    - broadcasted_shape: tuple
        Shape to which the data was broadcasted.

    Returns:
    - np.ndarray
        Unbroadcasted data.
    """

    def get_axes_to_be_summed(origin_data_shape, broadcast_shape):
        """
        Helper function to identify axes to be summed during unbroadcasting.
        """
        axes_to_be_summed = []
        zipped = list(zip_longest(tuple(reversed(broadcasted_shape)), tuple(reversed(origin_data_shape)), fillvalue=None))
        for dim, (dim_brodcasted, dim_orig) in enumerate(reversed(zipped)):
            if dim_brodcasted != dim_orig:
                axes_to_be_summed.append(dim)
        return tuple(axes_to_be_summed)

    if broadcasted_shape is not None:
        axes_to_be_summed = get_axes_to_be_summed(origin_data_shape, broadcasted_shape)
        unbroadcasted_data = np.sum(data, axis=axes_to_be_summed)
    else:
        unbroadcasted_data = data
    return unbroadcasted_data


def get_graph():
    """
    Get the global graph instance.

    Returns:
    - Graph
        Global graph instance.
    """
    if Graph.graph is None:
        from .. import _WH_GRAPH
        graph = _WH_GRAPH
    else:
        graph = _WH_GRAPH
    return graph


class new_graph():
    """
    Context manager for creating a new graph instance.
    """
    def __enter__(self):
        Graph.graph = Graph()

    def __exit__(self, exc_value, exc_traceback):
        Graph.graph = None


class no_track():
    """
    Context manager for temporarily disabling tracking in the graph.
    """
    def __init__(self):
        self.graph = get_graph()

    def __enter__(self):
        self.graph.track = False

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.graph.track = True


def _evaluate_grad_check(analytical_grads, calculated_grads, epsilon, print_vals):
    """
    Evaluate the distance between analytical and calculated gradients.

    Parameters:
    - analytical_grads: np.ndarray
        Analytically computed gradients.
    - calculated_grads: np.ndarray
        Gradients calculated using numerical perturbation.
    - epsilon: float
        Small perturbation value for numerical differentiation.
    - print_vals: bool
        Flag to print the gradient check result.

    Returns:
    - float
        Distance between analytical and calculated gradients.
    """
    dist = np.linalg.norm(analytical_grads - calculated_grads) / (np.linalg.norm(analytical_grads) + np.linalg.norm(calculated_grads))
    if print_vals:
        print("Gradient Check Distance", dist)
        if dist < epsilon:
            print("Gradient Check PASSED")
        else:
            print("Gradient Check FAILED")

    return dist


def _wiggle_params(analytical_grads, calculated_grads, params, get_loss, epsilon):
    """
    Perturb model parameters and calculate gradients for gradient checking.

    Parameters:
    - analytical_grads: list
        List to store analytically computed gradients.
    - calculated_grads: list
        List to store gradients calculated using numerical perturbation.
    - params: iterable
        Model parameters to be perturbed.
    - get_loss: callable
        Function to calculate the loss.
    - epsilon: float
        Small perturbation value for numerical differentiation.
    """
    for param in params:
        if param.requires_grad:
            if not(isinstance(param.grad, np.ndarray)):
                param.grad = np.array(param.grad)

            for idx in np.ndindex(param.shape):
                with no_track():
                    param.data[idx] += epsilon    # Plus
                    loss1 = get_loss()
                    param.data[idx] -= (2 * epsilon)  # MINUS
                    loss2 = get_loss()
                    param.data[idx] += epsilon  # ORIGINAL
                calculated_grads.append(param.grad[idx])
                analytical_grads.append((loss1.data - loss2.data) / (2 * epsilon))
        param.zero_grad()     # to prevent any side effects


def grad_check(model, inputs, targets, loss_fn, epsilon=1e-7, print_vals=True):
    """
    Perform gradient check for a given model.

    Parameters:
    - model: object
        Model for which gradient check is performed.
    - inputs: np.ndarray
        Input data.
    - targets: np.ndarray
        Target values.
    - loss_fn: callable
        Loss function used for gradient calculation.
    - epsilon: float
        Small perturbation value for numerical differentiation.
    - print_vals: bool
        Flag to print the gradient check result.

    Returns:
    - float
        Distance between analytical and calculated gradients.
    """

    params = model.parameters()
    analytical_grads = []
    calculated_grads = []

    for param in params:
        param.zero_grad()

    def get_loss():
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        return loss

    with new_graph():
        loss = get_loss()
        loss.backward()
        _wiggle_params(analytical_grads, calculated_grads, params, get_loss, epsilon)

    analytical_grads = np.array(analytical_grads)
    calculated_grads = np.array(calculated_grads)
    return _evaluate_grad_check(analytical_grads, calculated_grads, epsilon, print_vals)


def fn_grad_checks(fn, inputs, params, targets=None, loss_fn=None, epsilon=1e-7, print_vals=True, **kwargs):
    """
    Perform gradient check for a given function.

    Parameters:
    - fn: callable
        Function for which gradient check is performed.
    - inputs: tuple
        Input arguments for the function.
    - params: iterable
        Parameters to be perturbed during gradient check.
    - targets: np.ndarray, optional
        Target values for the function.
    - loss_fn: callable, optional
        Loss function used for gradient calculation. If not provided, MSE loss is used.
    - epsilon: float, optional
        Small perturbation value for numerical differentiation.
    - print_vals: bool, optional
        Flag to print the gradient check result.
    - **kwargs: dict
        Additional keyword arguments to be passed to the function.

    Returns:
    - float
        Distance between analytical and calculated gradients.
    """

    if loss_fn is None:
        from ..nn.loss import MSE 
        loss_fn = MSE()

    analytical_grads = []
    calculated_grads = []

    for param in params:
        param.zero_grad()

    def get_loss(targets=targets):
        outputs = fn(*inputs, **kwargs)
        if targets is None:
            from .tensor import tensor as tensor
            targets = tensor(np.ones(outputs.shape))
        loss = loss_fn(outputs, targets)

        return loss

    with new_graph():
        loss = get_loss()
        loss.backward()
        _wiggle_params(analytical_grads, calculated_grads, params, get_loss, epsilon)

    analytical_grads = np.array(analytical_grads)
    calculated_grads = np.array(calculated_grads)
    return _evaluate_grad_check(analytical_grads, calculated_grads, epsilon, print_vals)
