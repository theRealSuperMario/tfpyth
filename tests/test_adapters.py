import tensorflow as tf
import torch as th
import numpy as np
import tfpyth


def test_pytorch_in_tensorflow_eager_mode():
    tf.enable_eager_execution()
    tfe = tf.contrib.eager

    def pytorch_expr(a, b):
        return 3 * a + 4 * b * b

    x = tfpyth.eager_tensorflow_from_torch(pytorch_expr)

    assert tf.math.equal(x(tf.convert_to_tensor(1.0), tf.convert_to_tensor(3.0)), 39.0)

    dx = tfe.gradients_function(x)
    assert all(tf.math.equal(dx(tf.convert_to_tensor(1.0), tf.convert_to_tensor(3.0)), [3.0, 24.0]))
    tf.disable_eager_execution()


def test_pytorch_in_tensorflow_graph_mode():
    session = tf.Session()

    def pytorch_expr(a, b):
        return 3 * a + 4 * b * b

    a = tf.placeholder(tf.float32, name="a")
    b = tf.placeholder(tf.float32, name="b")
    c = tfpyth.tensorflow_from_torch(pytorch_expr, [a, b], tf.float32)
    c_grad = tf.gradients([c], [a, b], unconnected_gradients="zero")

    assert np.allclose(session.run([c, c_grad[0], c_grad[1]], {a: 1.0, b: 3.0}), [39.0, 3.0, 24.0])


def test_tensorflow_in_pytorch():
    session = tf.Session()

    def get_tf_function():
        a = tf.placeholder(tf.float32, name="a")
        b = tf.placeholder(tf.float32, name="b")
        c = 3 * a + 4 * b * b

        f = tfpyth.torch_from_tensorflow(session, [a, b], c).apply
        return f

    f = get_tf_function()
    a_ = th.tensor(1, dtype=th.float32, requires_grad=True)
    b_ = th.tensor(3, dtype=th.float32, requires_grad=True)
    x = f(a_, b_)

    assert x == 39.0

    x.backward()

    assert np.allclose((a_.grad, b_.grad), (3.0, 24.0))


class Test_tensorflow_in_pytorch:
    def test_single_output(self):
        session = tf.Session()

        def get_tf_function():
            a = tf.placeholder(tf.float32, name="a")
            b = tf.placeholder(tf.float32, name="b")
            c = 3 * a + 4 * b * b

            f = tfpyth.torch_from_tensorflow(session, [a, b], c).apply
            return f

        f = get_tf_function()
        a_ = th.tensor(1, dtype=th.float32, requires_grad=True)
        b_ = th.tensor(3, dtype=th.float32, requires_grad=True)
        x = f(a_, b_)

        assert x == 39.0

        x.backward()

        assert np.allclose((a_.grad, b_.grad), (3.0, 24.0))

    def test_multiple_outputs(self):
        session = tf.Session()

        def get_tf_function():
            a = tf.placeholder(tf.float32, name="a")
            b = tf.placeholder(tf.float32, name="b")
            c = 3 * a + 4 * b * b
            d = 6 * a + 8 * b ** 2

            f = tfpyth.torch_from_tensorflow(session, [a, b], [c, d])
            f1, f2 = [ff.apply for ff in f]
            return f1, f2

        f1, f2 = get_tf_function()

        def f(a, b):
            return f1(a, b), f2(a, b)

        a_ = th.tensor(1, dtype=th.float32, requires_grad=True)
        b_ = th.tensor(3, dtype=th.float32, requires_grad=True)
        x1, x2 = f(a_, b_)

        assert x1 == 39.0
        assert x2 == 78.0

        x1.backward()
        x2.backward()

        assert np.allclose((a_.grad, b_.grad), (9.0, 72.0))


class Test_wrap_torch_from_tensorflow:
    def test_image_operation(self):
        def tensorflow_function(a, size=(128, 128)):
            return tf.image.resize(a, size=size)

        from functools import partial

        session = tf.compat.v1.Session()
        tf_func = partial(tensorflow_function, size=(128, 128))
        f_pt = tfpyth.wrap_torch_from_tensorflow(tf_func, ["a"], [(None, 64, 64, 1)], session=session)
        x = th.ones((1, 64, 64, 1), dtype=th.float32)
        y = f_pt(x)
        assert y.shape == (1, 128, 128, 1)

    def test_no_gradient_operation(self):
        def tensorflow_function(a, size=(128, 128)):
            return tf.image.resize(a, size=size)

        from functools import partial

        session = tf.compat.v1.Session()
        tf_func = partial(tensorflow_function, size=(128, 128))
        f_pt = tfpyth.wrap_torch_from_tensorflow(tf_func, ["a"], [(None, 64, 64, 1)], session=session)
        x = th.ones((1, 64, 64, 1), dtype=th.float32, requires_grad=False)
        conv = th.nn.Conv2d(1, 1, 1)
        x = conv(tfpyth.th_2D_channels_last_to_first(x))
        x = tfpyth.th_2D_channels_first_to_last(x)
        y = f_pt(x)

        assert y.shape == (1, 128, 128, 1)
        assert y.sum().backward() is None
        assert conv.bias.grad

    def test_tensorflow_in_pytorch(self):
        session = tf.compat.v1.Session()

        def get_tf_function(a, b):
            c = 3 * a + 4 * b * b

            return c

        session = tf.compat.v1.Session()
        f = tfpyth.wrap_torch_from_tensorflow(get_tf_function, ["a", "b"], None, session=session)
        a_ = th.tensor(1, dtype=th.float32, requires_grad=True)
        b_ = th.tensor(3, dtype=th.float32, requires_grad=True)
        x = f(a_, b_)

        assert x == 39.0

        x.backward()

        assert np.allclose((a_.grad, b_.grad), (3.0, 24.0))

    def test_multiple_outputs(self):
        session = tf.compat.v1.Session()

        def get_tf_function(a, b):
            c = 3 * a + 4 * b * b
            d = 6 * a + 8 * b ** 2

            return c, d

        session = tf.compat.v1.Session()
        f = tfpyth.wrap_torch_from_tensorflow(get_tf_function, ["a", "b"], None, session=session)
        a_ = th.tensor(1, dtype=th.float32, requires_grad=True)
        b_ = th.tensor(3, dtype=th.float32, requires_grad=True)
        x1, x2 = f(a_, b_)

        assert x1 == 39.0
        assert x2 == 78.0

        x1.backward()
        assert np.allclose((a_.grad, b_.grad), (3.0, 24.0))
        x2.backward()  # partial derivatives are additive
        assert np.allclose((a_.grad, b_.grad), (9.0, 72.0))

    def test_autodetect_varnames(self):
        session = tf.compat.v1.Session()

        def get_tf_function(a, b):
            c = 3 * a + 4 * b * b

            return c

        session = tf.compat.v1.Session()
        f = tfpyth.wrap_torch_from_tensorflow(get_tf_function)
        a_ = th.tensor(1, dtype=th.float32, requires_grad=True)
        b_ = th.tensor(3, dtype=th.float32, requires_grad=True)
        x = f(a_, b_)

        assert x == 39.0

        x.backward()

        assert np.allclose((a_.grad, b_.grad), (3.0, 24.0))
