import numpy as np
import dnn_im2col

class linear_layer:

    def __init__(self, input_D, output_D):
        self.params = dict()
        self.params['W'] = np.random.normal(0, 0.1, (input_D, output_D))
        self.params['b'] = np.random.normal(0, 0.1, (1, output_D))

        self.gradient = dict()
        self.gradient['W'] = np.zeros((input_D, output_D))
        self.gradient['b'] = np.zeros((1, output_D))

    def forward(self, X):
        # TODO: Implement the linear forward pass. Store the result in forward_output  #
        forward_output = np.dot(X, self.params['W']) + self.params['b']
        return forward_output
        

    def backward(self, X, grad):
        # TODO: Implement the backward pass (i.e., compute the following three terms)                                            #
        self.gradient['W'] = np.dot(X.T, grad)
        self.gradient['b'] = np.sum(grad, axis=0, keepdims=True)
        backward_output = np.dot(grad, self.params['W'].T)
        return backward_output

class relu:
    def __init__(self):
        self.mask = None

    def forward(self, X):

         # TODO: Implement the relu forward pass. Store the result in forward_output    #
        self.mask = X > 0
        forward_output = np.maximum(0, X)
        return forward_output

    def backward(self, X, grad):
        # TODO: Implement the backward pass (i.e., compute the following term)                                                   #
        backward_output = grad * self.mask
        return backward_output

class dropout:
    def __init__(self, r):
        self.r = r
        self.mask = None

    def forward(self, X, is_train):
        #  TODO: We provide the forward pass to you. You only need to understand it.   #
        if not is_train:
            return X
        self.mask = np.random.rand(*X.shape) >= self.r
        forward_output = X * self.mask / (1 - self.r)
        return forward_output

    def backward(self, X, grad):
        # TODO: Implement the backward pass (i.e., compute the following term)                                                   #
        backward_output = grad * self.mask / (1 - self.r)
        return backward_output

class conv_layer:

    def __init__(self, num_input, num_output, filter_len, stride):
        self.params = dict()
        self.params['W'] = np.random.normal(0, 0.1, (num_output, num_input, filter_len, filter_len))
        self.params['b'] = np.random.normal(0, 0.1, (num_output, 1))

        self.gradient = dict()
        self.gradient['W'] = np.zeros((num_output, num_input, filter_len, filter_len))
        self.gradient['b'] = np.zeros((num_output, 1))

        self.stride = stride
        self.padding = int((filter_len - 1) / 2)
        self.X_col = None

    def forward(self, X):
        n_filters, d_filter, h_filter, w_filter = self.params['W'].shape
        n_x, d_x, h_x, w_x = X.shape
        h_out = int((h_x - h_filter + 2 * self.padding) / self.stride + 1)
        w_out = int((w_x - w_filter + 2 * self.padding) / self.stride + 1)

        self.X_col = dnn_im2col.im2col_indices(X, h_filter, w_filter, self.padding, self.stride)
        W_col = self.params['W'].reshape(n_filters, -1)
        #print(W_col.shape, self.X_col.shape)
        out = np.matmul(W_col, self.X_col) + self.params['b']
        out = out.reshape(n_filters, h_out, w_out, n_x)
        out_forward = out.transpose(3, 0, 1, 2)

        return out_forward

    def backward(self, X, grad):
        n_filters, d_filter, h_filter, w_filter = self.params['W'].shape

        self.gradient['b'] = np.sum(grad, axis=(0, 2, 3)).reshape(n_filters, -1)

        grad_reshaped = grad.transpose(1, 2, 3, 0).reshape(n_filters, -1)
        self.gradient['W'] = np.matmul(grad_reshaped, self.X_col.T).reshape(self.params['W'].shape)

        W_reshape = self.params['W'].reshape(n_filters, -1)
        out = np.matmul(W_reshape.T, grad_reshaped)
        out_backward = dnn_im2col.col2im_indices(out, X.shape, h_filter, w_filter, self.padding, self.stride)

        return out_backward


class max_pool:

    def __init__(self, max_len, stride):
        self.max_len = max_len
        self.stride = stride
        self.padding = 0 # int((max_len - 1) / 2)
        self.argmax_cols = None

    def forward(self, X):
        n_x, d_x, h_x, w_x = X.shape
        h_out = int((h_x - self.max_len + 2 * self.padding) / self.stride + 1)
        w_out = int((w_x - self.max_len + 2 * self.padding) / self.stride + 1)

        max_cols, self.argmax_cols = dnn_im2col.maxpool_im2col_indices(X, self.max_len, self.max_len, self.padding, self.stride)
        out_forward = max_cols.reshape(n_x, d_x, h_out, w_out)

        return out_forward

    def backward(self, X, grad):
        out_backward = dnn_im2col.maxpool_col2im_indices(grad, self.argmax_cols, X.shape, self.max_len, self.max_len, self.padding, self.stride)

        return out_backward


class flatten_layer:

    def __init__(self):
        self.size = None

    def forward(self, X):
        self.size = X.shape
        out_forward = X.reshape(X.shape[0], -1)

        return out_forward

    def backward(self, X, grad):
        out_backward = grad.reshape(self.size)

        return out_backward


### Loss functions ###

class softmax_cross_entropy:
    def __init__(self):
        self.expand_Y = None
        self.calib_logit = None
        self.sum_exp_calib_logit = None
        self.prob = None

    def forward(self, X, Y):
        self.expand_Y = np.zeros(X.shape).reshape(-1)
        self.expand_Y[Y.astype(int).reshape(-1) + np.arange(X.shape[0]) * X.shape[1]] = 1.0
        self.expand_Y = self.expand_Y.reshape(X.shape)

        self.calib_logit = X - np.amax(X, axis = 1, keepdims = True)
        self.sum_exp_calib_logit = np.sum(np.exp(self.calib_logit), axis = 1, keepdims = True)
        self.prob = np.exp(self.calib_logit) / self.sum_exp_calib_logit

        forward_output = - np.sum(np.multiply(self.expand_Y, self.calib_logit - np.log(self.sum_exp_calib_logit))) / X.shape[0]
        return forward_output

    def backward(self, X, Y):
        backward_output = - (self.expand_Y - self.prob) / X.shape[0]
        return backward_output


class sigmoid_cross_entropy:
    def __init__(self):
        self.expand_Y = None
        self.calib_logit = None
        self.sum_exp_calib_logit = None
        self.prob = None

    def forward(self, X, Y):
        self.expand_Y = np.concatenate((Y, 1 - Y), axis = 1)

        X_cat = np.concatenate((X, np.zeros((X.shape[0], 1))), axis = 1)
        self.calib_logit = X_cat - np.amax(X_cat, axis=1, keepdims=True)
        self.sum_exp_calib_logit = np.sum(np.exp(self.calib_logit), axis=1, keepdims=True)
        self.prob = np.exp(self.calib_logit[:, 0].reshape(X.shape[0], -1)) / self.sum_exp_calib_logit

        forward_output = - np.sum(np.multiply(self.expand_Y, self.calib_logit - np.log(self.sum_exp_calib_logit))) / X.shape[0]
        return forward_output

    def backward(self, X, Y):
        backward_output = - (self.expand_Y[:, 0].reshape(X.shape[0], -1) - self.prob) / X.shape[0]
        return backward_output


### Momentum ###

def add_momentum(model):
    momentum = dict()
    for module_name, module in model.items():
        if hasattr(module, 'params'):
            for key, _ in module.params.items():
                momentum[module_name + '_' + key] = np.zeros(module.gradient[key].shape)
    return momentum