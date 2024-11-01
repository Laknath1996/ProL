#!/bin/bash
######### Synthetic data
 # python3 01_generate.py seq_len=10000 num_seeds=10 data=synthetic scenario=2

# python3 02_train.py net.type=mlp name='mlp_erm' \
#     tag=scenario2_v2 numseeds=1 \
#     tstart=20 tskip=200 tend=2001

python3 02_train.py net.type=prospective_mlp name='mlp_prospective' \
    tag=scenario2_v2 numseeds=1 \
    tstart=20 tskip=200 tend=2001 

# python3 02_train.py net.type=mlp name='mlp_ft1' \
#     tag=scenario2_v2 numseeds=1 \
#     tstart=20 tskip=200 tend=2001  \
#     train.epochs=1 fine_tune=16 data.bs=8

# python3 02_train.py net.type=mlp name='mlp_bgd' \
#     tag=scenario2_v2 numseeds=1 \
#     tstart=20 tskip=200 tend=2001 \
#     train.epochs=1 fine_tune=16 data.bs=8 bgd=True



# python3 02_train.py net.type=minimlp name='minimlp_erm' \
#     tag=scenario2_v2 numseeds=1 \
#     tstart=20 tskip=200 tend=2001
#
# python3 02_train.py net.type=miniprospective_mlp name='minimlp_prospective' \
#     tag=scenario2_v2 numseeds=1 \
#     tstart=20 tskip=200 tend=2001 
#
# python3 02_train.py net.type=minimlp name='minimlp_ft1' \
#     tag=scenario2_v2 numseeds=1 \
#     tstart=20 tskip=200 tend=2001  \
#     train.epochs=1 fine_tune=16 data.bs=8
#
# python3 02_train.py net.type=minimlp name='minimlp_bgd' \
#     tag=scenario2_v2 numseeds=1 \
#     tstart=20 tskip=200 tend=2001 \
#     train.epochs=1 fine_tune=16 data.bs=8 bgd=True


# python3 02_train.py net.type=minimlp name='minimlp_erm_all' \
#     tag=scenario2_v2 numseeds=1 \
#     tstart=20 tskip=1 tend=501
    
# python3 02_train.py net.type=minimlp name='minimlp_ft1_all' \
#     tag=scenario2_v2 numseeds=1 \
#     tstart=20 tskip=1 tend=501  \
#     train.epochs=1 fine_tune=8 data.bs=8
