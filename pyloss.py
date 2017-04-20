"""Python implementation of Caffe SoftmaxWithLossLayer."""
import caffe
import numpy as np


class SoftmaxWithLossLayer(caffe.Layer):
    """Pycaffe Layer for SoftmaxWithLoss."""

    def setup(self, bottom, top):
        r"""Setup the layer with params.

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
            param_str: "{\'ignore_label\': 255, \'loss_weight\': 1, "
                       "\'normalization\': 1, \'axis\': 1}"
          }
        }
        """
        # config: python param
        self.params = eval(self.param_str)
        # softmax_param
        self._softmax_axis = self.params.get('axis', 1)
        # loss_param
        self._normalization = self.params.get('normalization', 2)
        self._ignore_label = self.params.get('ignore_label', None)
        self._loss_weight = self.params.get('loss_weight', 1)
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
        loss = self.compute_loss(prob, label)
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
            label = bottom[1].data.astype('int8')
            bottom_diff = self.compute_diff(self.prob, label)
            # pass the derivatives to bottom[0]
            bottom[0].diff[...] = bottom_diff

    def compute_loss(self, prob, label):
        """Return softmax loss."""
        # indexing prob using labels
        indice = label.copy()
        indice[label == self._ignore_label] = 0
        prob_red = self.reduce_prob(prob, indice)
        # negative log likelihood loss
        neg_log = -np.log(prob_red.clip(min=1e-16))
        # if loss_param has ignore_label
        if self._ignore_label:
            neg_log[label == self._ignore_label] = 0
        # compute normalized loss
        loss = np.sum(neg_log) / float(self.get_normalizer(label))
        return loss

    def compute_diff(self, prob, label):
        """Return sofmax loss derivative."""
        # keep only valid labels
        label_val = label.copy()
        label_val[label == self._ignore_label] = 0
        # convert label to one hot
        n_cl = prob.shape[self._softmax_axis]
        label_1hot = self.one_hot_encode(label_val, n_cl)
        # compute derivative by y_j - a_j
        diff = prob - label_1hot
        diff[self.ignore_mask(label, dim=n_cl)] = 0
        # normalize diff
        loss_weight = self._loss_weight / float(self.get_normalizer(label))
        bottom_diff = loss_weight * diff
        return bottom_diff

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
        indices = np.indices(label.shape)
        indices[self._softmax_axis] = label
        return prob[tuple(indices)]

    def one_hot_encode(self, label, n_cl):
        """Return one hot encoded labels."""
        label_sqz = np.squeeze(label, (self._softmax_axis,))
        label_1hot = np.eye(n_cl)[label_sqz]
        label_1hot = np.rollaxis(label_1hot, -1, self._softmax_axis)
        return label_1hot

    def ignore_mask(self, label, dim=None):
        """Return label ignore mask."""
        if dim:
            repeats = [1] * len(label.shape)
            repeats[self._softmax_axis] = dim
            return np.tile(label == self._ignore_label, repeats)
        else:
            return label == self._ignore_label


class AsymmetricSoftmaxWithLossLayer(SoftmaxWithLossLayer):
    """An asymmetric softmax loss weighted by class dependent weights."""

    def setup(self, bottom, top):
        """Parse class-dependent weight from params."""
        super(AsymmetricSoftmaxWithLossLayer, self).setup(bottom, top)
        n_cl = bottom[0].shape[self._softmax_axis]
        self._class_weight = np.array(
                self.params.get('class_weight', np.ones(n_cl)))
        self._class_weight = self._class_weight / \
            float(np.max(self._class_weight))

    def compute_loss(self, prob, label):
        """Return softmax loss."""
        # indexing prob using labels
        indice = label.copy()
        indice[label == self._ignore_label] = 0
        prob_red = self.reduce_prob(prob, indice)
        # negative log likelihood loss
        neg_log = -np.log(prob_red.clip(min=1e-16))
        # weight loss with class dependent weights
        neg_log = neg_log * self._class_weight[indice]
        # if loss_param has ignore_label
        if self._ignore_label:
            neg_log[label == self._ignore_label] = 0
        # compute normalized loss
        loss = np.sum(neg_log) / float(self.get_normalizer(label))
        return loss

    def compute_diff(self, prob, label):
        """Return softmax loss derivative."""
        # keep only valid labels
        label_val = label.copy()
        label_val[label == self._ignore_label] = 0
        # convert label to one hot
        n_cl = prob.shape[self._softmax_axis]
        label_1hot = self.one_hot_encode(label_val, n_cl)
        # compute derivative by w_t (a_j - y_j)
        diff = prob - label_1hot
        class_weights = self.get_class_weights(self._class_weight,
                                               label_val)
        diff = diff * class_weights
        diff[self.ignore_mask(label, dim=n_cl)] = 0
        # normalize
        loss_weight = self._loss_weight / float(self.get_normalizer(label))
        bottom_diff = loss_weight * diff
        return bottom_diff

    def get_class_weights(self, class_weight, label):
        """Return of same shape as prob."""
        label_sqz = np.squeeze(label, (self._softmax_axis,))
        vget = np.vectorize(lambda cls: class_weight[cls])
        class_weights = vget(label_sqz)
        repeats = [1] * len(label.shape)
        repeats[self._softmax_axis] = len(class_weight)
        return np.tile(class_weights, repeats)

    def get_tiled_weights(self, class_weight, target_shape):
        """Return tiled class_weight."""
        unit_shape = [sz if ax == self._softmax_axis else 1
                      for ax, sz in enumerate(target_shape)]
        class_weight = class_weight.reshape(unit_shape)
        repeats = [1 if ax == self._softmax_axis else sz
                   for ax, sz in enumerate(target_shape)]
        tiled_weights = np.tile(class_weight, repeats)
        return tiled_weights
