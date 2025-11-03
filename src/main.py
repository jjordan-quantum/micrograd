from engine import Value
from viz import draw_dot

print('hell world')

# ==============================================
#   Manual Back Prop Example
# ==============================================

def manual_backprop(use_opt=False):
    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')
    f = Value(-2.0, label='f')
    e = a * b; e.label = 'e'
    d = e + c; d.label = 'd'
    L = d * f; L.label = 'L'

    L.grad = 1.0
    f.grad = 4.0
    d.grad = -2.0
    e.grad = -2.0
    c.grad = -2.0
    a.grad = 6.0
    b.grad = -4.0

    print(L)
    print(L._prev)
    draw_dot(L)

    if use_opt:
        a.data += 0.01 * a.grad
        b.data += 0.01 * b.grad
        c.data += 0.01 * c.grad
        f.data += 0.01 * f.grad

        e = a * b; e.label = 'e'
        d = e + c; d.label = 'd'
        L = d * f; L.label = 'L'

        print(f'L: {L.data}')

# manual_backprop(True)

# ==============================================
#   Simple Neuron Example
# ==============================================

def simple_neuron():
    # inputs: x1,x2
    x1 = Value(2.0, label='x1')
    x2 = Value(0.0, label='x2')
    # weights: w1, w2
    w1 = Value(-3.0, label='w1')
    w2 = Value(1.0, label='w2')
    # bias of the neuron
    b = Value(6.88137, label='b')

    x1w1 = x1*w1; x1w1.label = 'x1*w1'
    x2w2 = x2*w2; x2w2.label = 'x2*w2'
    x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1w1 + x2w2'
    n = x1w1x2w2 + b; n.label = 'n'
    o = n.tanh(); o.label = 'o'

    # call _backward manually

    # o.grad = 1.0
    # o._backward()
    # n._backward()
    # b._backward()
    # x1w1x2w2._backward()
    # x2w2._backward()
    # x1w1._backward()
    #
    # x1._backward()
    # x2._backward()
    # w1._backward()
    # w2._backward()

    o.backward()
    draw_dot(o)

# ==============================================
#   Multi Reference Example
# ==============================================

def multi_reference():
    a = Value(-2.0, label='a')
    b = Value(3.0, label='b')
    d = a * b; d.label = 'd'
    e = a + b; e.label = 'e'
    f = d * e; f.label = 'f'

    f.backward()
    draw_dot(f)

# ==============================================
#   TanH Components Example
# ==============================================

def tanh_components():
    # inputs: x1,x2
    x1 = Value(2.0, label='x1')
    x2 = Value(0.0, label='x2')
    # weights: w1, w2
    w1 = Value(-3.0, label='w1')
    w2 = Value(1.0, label='w2')
    # bias of the neuron
    b = Value(6.88137, label='b')

    x1w1 = x1*w1; x1w1.label = 'x1*w1'
    x2w2 = x2*w2; x2w2.label = 'x2*w2'
    x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1w1 + x2w2'
    n = x1w1x2w2 + b; n.label = 'n'
    # ---
    e = (2*n).exp()
    o = (e - 1) / (e + 1)
    # ----
    o.label = 'o'

    o.backward()
    draw_dot(o)

if __name__ == '__main__':
    # manual_backprop(True)
    # simple_neuron()
    # multi_reference()
    tanh_components()
    pass