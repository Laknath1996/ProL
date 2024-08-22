
# python3 01_generate.py seq_len=10000 num_seeds=10 data=synthetic scenario=3

# python3 02_train.py  net.type=mlp3 name='erm_mlp' tag=scenario3_v2 numseeds=5 tstart=50 tend=4000 tskip=250 train.epochs=100 data.bs=32 data.path='./data/synthetic/scenario3_period20.pkl' dev='cuda:1'

# python3 02_train.py  net.type=prospective_mlp3 name='prospective_mlp' tag=scenario3_v2 numseeds=5 tstart=50 tend=4000 tskip=200 train.epochs=100 data.bs=32 data.path='./data/synthetic/scenario3_period20.pkl' dev='cuda:1'


# python3 01_generate.py seq_len=40000 num_seeds=5 data=mnist scenario=3

# python3 02_train.py net.type=mlp_mnist name='erm_mlp' \
#     dev='cuda:1' tag=mnist_s3_v2 numseeds=5 tstart=20 tend=5021 tskip=250 \
#     train.epochs=100 data.bs=32 data.path='./data/mnist/scenario3.pkl'

# python3 02_train.py net.type=prospective_mlp_mnist name='prospective_mlp' \
#     dev='cuda:1' tag=mnist_s3_v2 numseeds=5 tstart=20 tend=5021 tskip=250 \
#     train.epochs=100 data.bs=32 data.path='./data/mnist/scenario3.pkl'


# python3 01_generate.py seq_len=50000 num_seeds=5 data=cifar scenario=3

# python3 02_train.py  net.type=cnn_cifar name='erm_cnn' \
#     dev='cuda:3' tag=cifar_s3 numseeds=5 \
#     tstart=20 tend=30021 tskip=1500 \
#     train.epochs=100 data.bs=32 data.path='./data/cifar/scenario3.pkl'

# python3 02_train.py  net.type=prospective_cnn_cifar name='prospective_cnn_i' \
#     dev='cuda:3' tag=cifar_s3 numseeds=5 \
#     tstart=20 tend=30021 tskip=1500 \
#     train.epochs=100 data.bs=32 data.path='./data/cifar/scenario3.pkl'

# python3 02_train.py  net.type=prospective_cnn_cifar name='prospective_cnn_o' \
#     net.time_last=True \
#     dev='cuda:3' tag=cifar_s3 numseeds=5 \
#     tstart=20 tend=30021 tskip=1500 \
#     train.epochs=100 data.bs=32 data.path='./data/cifar/scenario3.pkl'

# python3 02_train.py  net.type=prospective_mlp_cifar name='prospective_mlp' \
#     dev='cuda:3' tag=cifar_s3 numseeds=5 \
#     tstart=20 tend=30021 tskip=1500 \
#     train.epochs=100 data.bs=32 data.path='./data/cifar/scenario3.pkl'

