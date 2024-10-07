python3 01_generate.py seq_len=50000 variant=markov2_s data=synthetic scenario=3 period=10 num_seeds=10 
python3 01_generate.py seq_len=50000 num_seeds=5 data=mnist scenario=3 period=10 variant=markov2_s
python3 01_generate.py seq_len=50000 num_seeds=5 data=cifar scenario=3 period=10 variant=markov2_s


##### Synthetic data
python3 02_train.py  name='prospective_mlp' tag=scenario3_markov2_s \
    numseeds=5 tstart=50 tend=10051 tskip=500  \
    net.type=prospective_mlp dev='cuda:0' \
    train.epochs=100 data.bs=32 \
    data.path='./data/synthetic/scenario3_markov2_s.pkl'

python3 02_train.py name='erm_mlp' tag=scenario3_markov2_s \
    numseeds=5 tstart=50 tend=10051 tskip=500 \
    train.epochs=100 data.bs=32 net.type=mlp dev='cuda:1' \
    data.path='./data/synthetic/scenario3_markov2_s.pkl'

#### MNIST
python3 02_train.py net.type=mlp_mnist name='erm_mlp' \
    dev='cuda:1' tag=mnist_s3_markov2_s numseeds=5 tstart=20 tend=10021 tskip=500 \
    train.epochs=100 data.bs=32 data.path='./data/mnist/scenario3_markov2_s.pkl'
#
python3 02_train.py net.type=prospective_mlp_mnist name='prospective_mlp' \
    dev='cuda:1' tag=mnist_s3_markov2_s numseeds=5 tstart=20 tend=10021 tskip=500 \
    train.epochs=100 data.bs=32 data.path='./data/mnist/scenario3_markov2_s.pkl'


#### CIFAR
python3 02_train.py name='erm_cnn' \
    dev='cuda:0' tag=cifar_s3_markov2_s numseeds=5 \
    tstart=20 tend=30021 tskip=1500 \
    net.type=cnn_cifar \
    train.epochs=100 data.bs=32 data.path='./data/cifar/scenario3_markov2_s.pkl'

python3 02_train.py name='prospective_cnn_o' \
    dev='cuda:0' tag=cifar_s3_markov2_s numseeds=5 \
    tstart=20 tend=30021 tskip=1500 \
    net.type=prospective_cnn_cifar net.time_last=True \
    train.epochs=100 data.bs=32 data.path='./data/cifar/scenario3_markov2_s.pkl'
