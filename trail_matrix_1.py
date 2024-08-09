import numpy as np
from numpy import array, dot
import tensorflow as tf
from Pinecone.System.SystemEnviron import SystemEnviron
SystemEnviron.disableCUDADevice()

# print(
#     np.asarray( [ [1.0], [2.0], [3.0]] )
#     *
#     np.asarray( [ [0.40784496], [1.191065], [1.9742855] ] )
# )


def trial_2():
    a = [
        [ 1.0, 2.0 ],
        [ 3.0, 4.0 ],
    ]

    b = [ [2.0], [2.0] ]

    print( tf.matmul( a, b ) )


def trial_3():
    f = array([ 1, 2 ])  # 1*2
    g = array([ [1, 2],
                [1, 2],
                [1, 2]
    ])

    a = [
        [1],
        [2]
    ]

    b = [ [1, 2],
          [1, 2],
          [1, 2]
    ]

    print( tf.matmul( b, a ) )
    print( g @ f )  # Auto reverse ??
    print( dot(g, f) )


def trial_4():
    a = array([
        [1, 2],
        [1, 2]
    ])

    b = array([
        [5, 6],
        [5, 6]
    ])

    a = array([
        [1, 2],
    ])

    b = array([
        [3],
        [4],
    ])

    print( b @ a )
    print( dot(a, b) )



which2Trail : any = 4
eval( F'trial_{ which2Trail }()' )
