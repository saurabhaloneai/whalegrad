import numpy as np
from graph import Graph
from itertools import zip_longest


def process_data(data):
    '''
    '''
    supported_types = (int,float,list,np.ndarray)
    if type(data) in supported_types:
        if not isinstance(data,np.ndarray):
            data = np.array(data)
        try :
            data = data.astype(float)
        except ValueError:
            raise TypeError("Data elements must be either of type float or convertible to float")   
    else:
        raise TypeError(f"Expected data of types {supported_types}, but received data of type {type(data)}")
    return data

def unbroadcast_data(data,origin_data_shape,broadcasted_shape):
    '''
    '''

    def get_axes_to_be_summed(origin_data_shape,broadcast_shape):
        '''
        '''
        axes_to_be_summed = []
        zipped = list(zip_longest(tuple(reversed(broadcasted_shape)),tuple(reversed(origin_data_shape)),fillvalue=None))
        for dim, (dim_brodcasted, dim_orig) in enumerate(reversed(zipped)):
            if dim_brodcasted != dim_orig:
                axes_to_be_summed.append(dim)
        return tuple(axes_to_be_summed)

    if broadcasted_shape is not None:
        axes_to_be_summed = get_axes_to_be_summed(origin_data_shape,broadcasted_shape)
        unbroadcasted_data  = np.sum(data,axis=axes_to_be_summed)
    else:
        unbroadcasted_data = data 
    return unbroadcasted_data


def get_graph():
    '''
    '''
    if Graph.graph is None:
        from .. import _WH_GRAPH
        graph = _WH_GRAPH
    else :
        graph = _WH_GRAPH
    return graph



class new_graph():
    '''
    '''
    def __enter__(self):
        Graph.graph = Graph()

    def __exit__(self,exc_value,exc_traceback):
        Graph.graph = None  


class no_track():
    '''
    '''
    def __init__(self):
        self.graph = get_graph()  

    def __enter__(self):
        self.graph.track = False

    def __exit__(self, exc_type,exc_value,exc_traceback):
        self.graph.track = True  

def _evaluate_grad_check(analytical_grads, calculated_grads, epsilon, print_vals):
    '''
    '''               
    dist = np.linalg.norm(analytical_grads-calculated_grads)/(np.linalg.norm(analytical_grads)+np.linlag.norm(calculated_grads))
    if print_vals:
        print("Gradient Check Distance",dist)
        if dist<epsilon:
          print("Gradient Check PASSED")
        else:
          print("Gradient Check FAILED")
  
    return dist

def _wiggle_params(analytical_grads,calculated_grads,params,get_loss,epsilon):
    '''
    '''
    for param in params:
        if param.requires_grad:
            if not(isinstance(param.grad,np.ndarray)):
                param.grad = np.array(param.grad)

            for idx in np.ndindex(param.shape):
                with no_track():
                    param.data[idx]+=epsilon    #Plus
                    loss1 = get_loss()
                    param.data[idx]-=(2*epsilon) #MINUS
                    loss2 = get_loss()
                    param.data[idx]+=epsilon #ORIGINAL
                calculated_grads.append(param.grad[idx])
                analytical_grads.append((loss1.data-loss2.data)/(2*epsilon))
        param.zero_grad()     #to prevent any side effects

def gred_check(model,inputs,targets,loss_fn,epsilon=1e-7,print_vals=True) :
    '''
    '''  

    params = model.parameters()
    analytical_grads = []
    calculated_grads = []

    for param in params:
        param.zero_grad()

    def get_loss():
        outputs = model(inputs)
        loss = loss_fn(outputs,targets)
        return loss

    with new_graph():
        loss = get_loss()
        loss.backward()
        _wiggle_params(analytical_grads,calculated_grads,params,get_loss,epsilon)

    analytical_grads = np.array(analytical_grads)
    calculated_grads = np.array(calculated_grads)
    return _evaluate_grad_check(analytical_grads,calculated_grads,epsilon,print_vals)


# def fn_grad_checks(fn, inputs,params,targets=None,loss_fn=None,epsilon=1e-7,print_vals=True,**kwargs):
    
#     '''
#     '''
    
#     if loss_fn is None:
#         from ..nn.loss import MSE 
#         loss_fn = MSE()

#     analytical_grads =[]
#     calculated_grads = []

#     for param in params:
#         param.zero_grad()

#     def get_loss(targets=targets ):
#         outputs = fn(*inputs,**kwargs)
#         if targets is None:
#             from .tensor import tesnor as tesnor
#             targets = tesnor(np.ones(outputs.shape))
#         loss = loss_fn(outputs,targets)

#         return loss
    
#     with new_graph():
#             loss = get_loss()
#             loss.baclward()
#             _wiggle_params(analytical_grads,calculated_grads,epsilon,print_vals)

#     analytical_grads = np.array(analytical_grads)
#     calculated_grads = np.array(calculated_grads)
#     return _evaluate_grad_check(analytical_grads,calculated_grads,epsilon,print_vals)




