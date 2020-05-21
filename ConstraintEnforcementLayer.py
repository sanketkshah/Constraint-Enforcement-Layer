from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Layer
from tensorflow import expand_dims, tensordot
from tensorflow.python.keras.initializers import Constant
from tensorflow import Print as tf_print
from tensorflow import check_numerics

import numpy as np
from docplex.mp.model import Model


class ConstraintEnforcementLayer(Layer):
    """Layer that enforces linear constraints on the output of a neural network
    layer.

    It implements an `onto` mapping to the convex hull defined by the
    inequality simplex. It does this by finding the point of intersection
    between a straight line joining the input point y and a fixed iterior point
    c. The point of intersection z is given by:

    ```
        z = \\alpha.y + (1 - \\alpha).c
    ```

    where \\alpha is:

    ```
        \\alpha = min_{0 < point <= 1}(elemwise division(b - Ac, A(y - c)))
    ```

    IMPORTANT:
    (1) Current implementation does not handle equality constraints.
    (2) Requires the addition of the regularisation term ||z - y|| to the overall
    loss function of the neural network.

    For more information refer to https://sanketkshah.github.io/publication/tsg/

    # Arguments
        A, b: numpy arrays defining the linear inequalities (Ax - b <= 0)
        c: An interior point to all the inequalities defined by Ax - b <= 0
    # Inputs
        y: Unconstrained output of neural network layer
    # Input shape
        nD tensor with shape: `(batch_size, ..., input_dim)`.
        The most common situation would be a 2D input with shape `(batch_size,
        input_dim)`.
    # Output shape
        Same as input.
    """

    def _is_interior_point(self, A, b, c):
        intersection_points = np.dot(A, c) - b
        return np.max(intersection_points) < 0

    def __init__(self, A, b, c=None, **kwargs):
        # Sanity check
        assert A.ndim == 2
        assert b.ndim == 1
        assert A.shape[0] == b.shape[0]

        # Initialisation
        self.A, self.b = A, b

        # Find interior point, if not specified
        self.c = c if c is not None else self._get_chebyshev_center(A, b)
        assert self.c.ndim == 1
        assert self.A.shape[1] == self.c.shape[0]
        assert self._is_interior_point(self.A, self.b, self.c)

        super(ConstraintEnforcementLayer, self).__init__(trainable=False, name='ConstraintEnforcementLayer', **kwargs)

    def _get_chebyshev_center(self, A, b):
        """Finds interior point c for inequality simplex. Runs an LP to find the
        chebyshev centre
        """

        # Solve an LP to find center
        model = Model()

        # Define c, min_margin
        num_vars = A.shape[1]
        c = model.continuous_var_list(num_vars, name='center')
        min_margin = model.continuous_var(lb=0, name='min_margin')

        # Define inequality constraints
        num_constraints = A.shape[0]
        for i in range(num_constraints):
            norm = np.linalg.norm(A[i, :])  # normalisation for constraints
            model.add_constraint(model.sum(c[j] * A[i, j] for j in range(num_vars)) + norm * min_margin <= b[i])

        # Add objective
        model.maximize(min_margin)

        # Solve
        solution = model.solve()
        assert solution  # making sure that LP doesn't fail
        assert solution.get_value(min_margin) > 0  # making sure that the point doesn't lie on the boundary

        # Return c
        return np.array([solution.get_value(c[i]) for i in range(num_vars)])

    def build(self, input_shape):
        # Sanity Check
        if isinstance(input_shape, list):
            raise ValueError('ConstraintEnforcementLayer has only 1 input.')
        assert input_shape[-1] == self.A.shape[1]

        # Create computation graph matrices and set values
        self.A_graph = self.add_weight(shape=self.A.shape, initializer=Constant(self.A), name='A', trainable=False)
        self.b_graph = self.add_weight(shape=self.b.shape, initializer=Constant(self.b), name='b', trainable=False)
        self.c_graph = self.add_weight(shape=self.c.shape, initializer=Constant(self.c), name='c', trainable=False)

        # Done building
        super(ConstraintEnforcementLayer, self).build(input_shape)

    def call(self, y):
        # Sanity Check
        if isinstance(y, list):
            raise ValueError('TSG layer has only 1 input')
        # y = tf_print(y, [y], message='{}: The unconstrained action is:'.format(y.name.split('/')[0]), summarize=-1)
        y = check_numerics(y, 'Problem with input y')

        # Calculate A.c
        Ac = tensordot(self.A_graph, self.c_graph, 1)

        # Calculate b - Ac
        bMinusAc = self.b_graph - Ac

        # Calculate y - c
        yMinusc = y - self.c_graph

        # Calculate A.(y - c)
        ADotyMinusc = K.sum((self.A_graph * expand_dims(yMinusc, -2)), axis=2)

        # Do elem-wise division
        intersection_points = bMinusAc / (ADotyMinusc + K.epsilon())  # Do we need the K.epsilon()?

        # Enforce 0 <= intersection_points <= 1 because the point must lie between c and y
        greater_1 = K.greater(intersection_points, K.ones_like(intersection_points))
        candidate_alpha = K.switch(greater_1, K.ones_like(intersection_points) + 1, intersection_points)

        less_0 = K.less(candidate_alpha, K.zeros_like(intersection_points))
        candidate_alpha = K.switch(less_0, K.ones_like(intersection_points) + 1, candidate_alpha)

        # Find farthest intersection point from y to get projection point
        alpha = K.min(candidate_alpha, axis=-1, keepdims=True)

        # If it is an interior point, y itself is the projection point
        interior_point = K.greater(alpha, K.ones_like(alpha))
        alpha = K.switch(interior_point, K.ones_like(alpha), alpha)
        # alpha = tf_print(alpha, [alpha], message="{}: The value of alpha is: ".format(alpha.name.split('/')[0]))

        # Return \alpha.y + (1 - \alpha).c
        z = alpha * y + ((1 - alpha) * self.c_graph)
        # z = tf_print(z, [z], message='{}: The constrained action is:'.format(z.name.split('/')[0]), summarize=-1)

        return z

    def compute_output_shape(self, input_shape):
        return input_shape


if __name__ == '__main__':
    # Set up debugger
    import pdb
    pdb.set_trace()

    # UNIT TEST FOR ConstraintEnforcementLayer
    # Define constraints
    #   Consider square of side 2 centered at origin
    A = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    b = np.array([1, 1, 1, 1])

    # Build dummy computation graph
    from tensorflow.python.keras.layers import Input
    from tensorflow.python.keras.models import Model as KerasModel
    inp = Input(shape=(2,))
    layer = ConstraintEnforcementLayer(A, b)
    out = layer(inp)
    model = KerasModel(inputs=inp, outputs=out)

    # Test for a predefined set of inputs
    inputs = np.array([[0, 0], [1, 1], [2, 2], [2, 0], [2, 0.5]])
    outputs = model.predict(np.array(inputs))

    # Check if they're what they should be
    outputs_correct = np.array([[0, 0], [1, 1], [1, 1], [1, 0], [1, 0.25]])
    if np.max(outputs - outputs_correct) < 1e-5:
        print("Unit Test Passed")
    else:
        print("Unit Test Failed")
