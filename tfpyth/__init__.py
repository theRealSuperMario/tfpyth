import tensorflow as tf
import torch as th


class TensorFlowFunction(th.autograd.Function):
    """
    Wrapper class for Tensorflow input/output nodes (incl gradient) in PyTorch.
    """

    inputs: list = None
    output: tf.Tensor = None
    gradient_placeholder = None
    gradient_outputs = None


def torch_from_tensorflow(tf_session, tf_inputs, tf_output, tf_dtype=tf.float32):
    """
    Create a PyTorch TensorFlowFunction with forward and backward methods which executes evaluates the passed
    TensorFlow tensors.

    ```python
    my_tensorflow_func = MyTensorFlowFunction.apply

    result = my_tensorflow_func(th_a, th_b)
    ```

    :param tf_session: TensorFlow session to use
    :param tf_inputs: TensorFlow input tensors/placeholders
    :param tf_output: TensorFlow output tensor
    :param tf_dtype: dtype to use for gradient placeholder.
    :return: TensorflowFunction which can be applied to PyTorch tensors.
    """
    # create gradient placeholders
    tf_gradient_placeholder = tf.placeholder(dtype=tf_dtype, name=f"gradient")
    tf_gradient_outputs = tf.gradients(
        ys=tf_output, xs=tf_inputs, grad_ys=[tf_gradient_placeholder], unconnected_gradients="zero"
    )

    class _TensorFlowFunction(TensorFlowFunction):
        inputs = tf_inputs
        output = tf_output
        gradient_placeholder = tf_gradient_placeholder
        gradient_outputs = tf_gradient_outputs

        @staticmethod
        def forward(ctx, *args):
            assert len(args) == len(tf_inputs)

            feed_dict = {tf_input: th_input.detach().numpy() for tf_input, th_input in zip(tf_inputs, args)}
            output = tf_session.run(tf_output, feed_dict)

            ctx.save_for_backward(*args)

            th_output = th.as_tensor(output)
            return th_output

        # See https://www.janfreyberg.com/blog/2019-04-01-testing-pytorch-functions/ for why "no cover"
        @staticmethod
        def backward(ctx, grad_output):  # pragma: no cover
            th_inputs = ctx.saved_tensors

            feed_dict = {}
            feed_dict.update({tf_input: th_input.detach().numpy() for tf_input, th_input in zip(tf_inputs, th_inputs)})
            feed_dict.update({tf_gradient_placeholder: grad_output.detach().numpy()})

            tf_gradients = tf_session.run(tf_gradient_outputs, feed_dict)
            return tuple(th.as_tensor(tf_gradient) for tf_gradient in tf_gradients)

    return _TensorFlowFunction()


def wrap_torch_from_tensorflow(func, tensor_inputs=None, input_shapes=None, session=None):
    """wrap func using `torch_from_tensorflow` and automatically create placeholders.

    By default, placeholders are assumed to be `tf.float32`.

    :param func: Callable.
        Tensorflow function to evaluate
    :param tensor_input: List[str] 
        List of argument names to `func` that represent a tensor input.
    :param input_shapes: List[Tuple[Int]]. 
        Shapes of input tensors if known. Some operations require these, such as all `tf.image.resize`.
        Basically these values are fed to `tf.placeholder`, so you can indicate unknown parameters using `(None, 64, 64, 1)`, for instance.
    :param session: tf.compat.v1.Session
        A session. If None, will instantiate new session.
    
    """
    if session is None:
        session = tf.compat.v1.Session()
    if tensor_inputs is None:
        tensor_inputs = func.__code__.co_varnames[: func.__code__.co_argcount]

    if input_shapes is not None:
        if len(tensor_inputs) != len(input_shapes):
            raise ValueError("Number of tensor inputs does not match number of input shapes")
        else:
            placeholders = {
                arg_name: tf.compat.v1.placeholder(tf.float32, shape=shape, name=arg_name)
                for arg_name, shape in zip(tensor_inputs, input_shapes)
            }
    else:
        placeholders = {arg_name: tf.compat.v1.placeholder(tf.float32, name=arg_name) for arg_name in tensor_inputs}
    output = func(**placeholders)
    f = torch_from_tensorflow(session, [placeholders[t] for t in tensor_inputs], output).apply
    return f


def eager_tensorflow_from_torch(func):
    """
    Wraps a PyTorch function into a TensorFlow eager-mode function (ie can be executed within Tensorflow eager-mode).

    :param func: Function that takes PyTorch tensors and returns a PyTorch tensor.
    :return: Differentiable Tensorflow eager-mode function.
    """

    @tf.custom_gradient
    def compute(*inputs):
        th_inputs = [th.tensor(tf_input.numpy(), requires_grad=True) for tf_input in inputs]
        th_output = func(*th_inputs)

        def compute_grad(d_output):
            th_d_output = th.tensor(d_output.numpy(), requires_grad=False)
            th_gradients = th.autograd.grad([th_output], th_inputs, grad_outputs=[th_d_output], allow_unused=True)
            tf_gradients = [tf.convert_to_tensor(th_gradient.numpy()) for th_gradient in th_gradients]
            return tf_gradients

        return tf.convert_to_tensor(th_output.detach().numpy()), compute_grad

    return compute


def tensorflow_from_torch(func, inp, Tout, name=None):
    """
    Executes a PyTorch function into a TensorFlow op and output tensor (ie can be evaluated within Tensorflow).\

    :param func: Function that takes PyTorch tensors and returns a PyTorch tensor.
    :param inp: TensorFlow input tensors
    :param Tout: TensorFlow output dtype
    :param name: Name of the output tensor
    :return: Differentiable Tensorflow output tensor.
    """
    eager_compute = eager_tensorflow_from_torch(func)

    return tf.py_function(eager_compute, inp, Tout, name=name)


def tf_NCHW_to_NHWC(x):
    return tf.transpose(x, (0, 2, 3, 1))


def tf_NHWC_to_NCHW(x):
    return tf.transpose(x, (0, 3, 1, 2))


tf_2D_channels_first_to_last = tf_NCHW_to_NHWC
tf_2D_channels_last_to_first = tf_NHWC_to_NCHW


def th_NCHW_to_NHWC(x):
    return x.permute((0, 2, 3, 1))


def th_NHWC_to_NCHW(x):
    return x.permute((0, 3, 1, 2))


th_2D_channels_last_to_first = th_NHWC_to_NCHW
th_2D_channels_first_to_last = th_NCHW_to_NHWC
