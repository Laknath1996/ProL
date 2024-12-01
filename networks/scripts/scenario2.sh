#!/bin/bash
######### Synthetic data
# python3 01_generate.py seq_len=10000 num_seeds=10 data=synthetic scenario=2

# python3 02_train.py net.type=mlp name='mlp_erm' \
#     tag=scenario2_v2 numseeds=5 \
#     tstart=20 tskip=200 tend=2001

# python3 02_train.py net.type=prospective_mlp name='mlp_prospective' \
#     tag=scenario2_v2 numseeds=5 \
#     tstart=20 tskip=200 tend=2001 

# python3 02_train.py net.type=mlp name='mlp_ft1' \
#     tag=scenario2_v2 numseeds=5 \
#     tstart=20 tskip=200 tend=2001  \
#     train.epochs=1 fine_tune=16 data.bs=8

# python3 02_train.py net.type=mlp name='mlp_bgd' \
#     tag=scenario2_v2 numseeds=5 \
#     tstart=20 tskip=200 tend=2001 \
#     train.epochs=1 fine_tune=16 data.bs=8 bgd=True


######### MNIST
# python3 01_generate.py seq_len=40000 num_seeds=5 data=mnist scenario=2

python3 02_train.py  net.type=mlp_mnist name='erm_mlp' \
    dev='cuda:0' tag=mnist_s2_v2 numseeds=5 \
    tstart=20 tend=5021 tskip=250 \
    train.epochs=10 data.bs=32 data.path='./data/mnist/scenario2.pkl'

python3 02_train.py  net.type=prospective_mlp_mnist name='prospective_mlp' \
    dev='cuda:0' tag=mnist_s2_v2 numseeds=5 \
    tstart=20 tend=5021 tskip=250 \
    train.epochs=100 data.bs=32 data.path='./data/mnist/scenario2.pkl'

python3 02_train.py net.type=mlp_mnist name='mlp_ft1' \
    dev='cuda:0' tag=mnist_s2_v2 numseeds=5 \
    tstart=20 tend=5001 tskip=250 \
    train.epochs=1 fine_tune=16 data.bs=8 data.path='./data/mnist/scenario2.pkl'

python3 02_train.py net.type=mlp_mnist name='mlp_bgd' \
    dev='cuda:0' tag=mnist_s2_v2 numseeds=5 \
    tstart=20 tend=5001 tskip=250 \
    train.epochs=1 fine_tune=16 data.bs=8 bgd=True data.path='./data/mnist/scenario2.pkl'

######### CIFAR
# python3 01_generate.py seq_len=50000 num_seeds=5 data=cifar scenario=2

# python3 02_train.py  net.type=cnn_cifar name='erm_cnn' \
#     dev='cuda:2' tag=cifar_s2 numseeds=5 \
#     tstart=20 tend=30021 tskip=1500 \
#     train.epochs=100 data.bs=32 data.path='./data/cifar/scenario2.pkl'

# python3 02_train.py  net.type=prospective_cnn_cifar name='prospective_cnn_i' \
#     dev='cuda:2' tag=cifar_s2 numseeds=5 \
#     tstart=20 tend=30021 tskip=1500 \
#     train.epochs=100 data.bs=32 data.path='./data/cifar/scenario2.pkl'

# python3 02_train.py  net.type=prospective_cnn_cifar name='prospective_cnn_o' \
#     net.time_last=True \
#     dev='cuda:2' tag=cifar_s2 numseeds=5 \
#     tstart=20 tend=30021 tskip=1500 \
#     train.epochs=100 data.bs=32 data.path='./data/cifar/scenario2.pkl'

# python3 02_train.py  net.type=prospective_mlp_cifar name='prospective_mlp' \
#     dev='cuda:2' tag=cifar_s2 numseeds=5 \
#     tstart=20 tend=30021 tskip=1500 \
#     train.epochs=100 data.bs=32 data.path='./data/cifar/scenario2.pkl'


# python3 02_train.py  net.type=cnn_cifar name='cnn_o_ft1' \
#     dev='cuda:0' tag=cifar_s2 numseeds=5 \
#     tstart=20 tend=30021 tskip=500 \
#     fine_tune=16 data.bs=8 train.epochs=1 data.path='./data/cifar/scenario2.pkl'

# python3 02_train.py  net.type=cnn_cifar name='cnn_o_bgd' \
#     dev='cuda:1' tag=cifar_s2 numseeds=5 \
#     tstart=20 tend=30021 tskip=500 bgd=True \
#     fine_tune=16 data.bs=8 train.epochs=1 data.path='./data/cifar/scenario2.pkl'
