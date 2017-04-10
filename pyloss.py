"""Python implementation of Caffe SoftmaxWithLossLayer."""
import caffe
import numpy as np


class SoftmaxWithLossLayer(caffe.Layer):
    """Pycaffe Layer for SoftmaxWithLoss."""

    def setup(self, bottom, top):
        """Setup the layer with params.

        Example params:
        layer {
          name: "loss"
          type: "Python"
          bottom: "score"
          bottom: "label"
          top: "loss"
          loss_weight: 1
          python_param {
            module: "pyloss"
            layer: "SoftmaxWithLossLayer"
            param_str: "{\'ignore_label\': 255, \'loss_weight\': 1, \'normalization\': 1, \'axis\': 1}"
          }
        }
        """
        # config: python param
        params = eval(self.param_str)
        # softmax_param
        self._softmax_axis = params.get('axis', 1)
        # loss_param
        self._normalization = params.get('normalization', 2)
        self._ignore_label = params.get('ignore_label', None)
        self._loss_weight = params.get('loss_weight', 1)
        # attributes initialization
        self.loss = None
        self.prob = None
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute softmax loss.")

    def reshape(self, bottom, top):
        """Reshape the layer."""
        # check input dimensions match
        if bottom[0].count != bottom[1].count * bottom[0].channels:
            raise Exception("Number of labels must match "
                            "number of predictions; "
                            "e.g., if softmax axis == 1 "
                            "and prediction shape is (N, C, H, W), "
                            "label count (number of labels) "
                            "must be N*H*W, with integer values "
                            "in {0, 1, ..., C-1}.")
        # loss output is scalar
        top[0].reshape(1)
        # softmax output is of same shape as bottom[0]
        if len(top) >= 2:
            top[1].reshape(*bottom[0].shape)

    def forward(self, bottom, top):
        """Forward computing on CPU."""
        # compute stable softmax probability
        score = bottom[0].data
        score -= np.max(score, axis=self._softmax_axis, keepdims=True)
        score_exp = np.exp(score)
        prob = score_exp / np.sum(score_exp, axis=self._softmax_axis,
                                  keepdims=True)
        # compute negative log-likelihood loss
        label = bottom[1].data.astype('int8')
        neg_log = -np.log(self.reduce_prob(prob, label).clip(min=1e-16))
        # if loss_param has ignore_label
        if self._ignore_label:
            neg_log[label == self._ignore_label] = 0
        # compute normalized loss
        loss = np.sum(neg_log) / float(self.get_normalizer(label))
        # pass loss top[0]
        top[0].data[...] = loss
        # update loss and prob
        self.loss = loss
        self.prob = prob

    def backward(self, top, propagate_down, bottom):
        """Backward computing on CPU."""
        if propagate_down[1]:
            raise Exception("SoftmaxWithLoss Layer cannot "
                            "backpropagate to label inputs.")
        if propagate_down[0]:
            # convert label to one hot
            label = bottom[1].data.astype('int8')
            n_cl = self.prob.shape[self._softmax_axis]
            label_1hot = self.one_hot_encode(label, n_cl)
            # compute derivative by y_ik - \delta(k=t_i)
            diff = self.prob - label_1hot
            loss_weight = self._loss_weight / float(self.get_normalizer(label))
            bottom_diff = loss_weight * diff
            # pass the derivatives to bottom[0]
            bottom[0].diff[...] = bottom_diff

    def get_normalizer(self, label):
        """Get the loss normalizer based normalization mode."""
        if self._normalization == 0:    # Full
            normalizer = label.size
        elif self._normalization == 1:  # VALID
            normalizer = np.sum(label != self._ignore_label)
        elif self._normalization == 2:  # BATCH_SIZE
            normalizer = label.shape[0]
        elif self._normalization == 3:  # NONE
            normalizer = 1.
        else:
            raise Exception("Unknown normalization mode: {}").format(
                    self._normalization)
        return max(1., normalizer)

    def reduce_prob(self, prob, label):
        """Return probabilities for given labels."""
        label_full = label.copy()
        label_full[label == self._ignore_label] = 0
        indices = np.indices(label_full.shape)
        indices[self._softmax_axis] = label_full
        return prob[tuple(indices)]

    def one_hot_encode(self, label, n_cl):
        """Return one hot encoded labels."""
        label_sqz = np.squeeze(label, (self._softmax_axis,))
        label_1hot = np.eye(n_cl)[label_sqz]
        label_1hot = np.rollaxis(label_1hot, -1, self._softmax_axis)
        return label_1hot
