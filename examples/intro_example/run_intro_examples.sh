#!bin/bash

for dir in dm_example gan_example gmm_example nf_example vae_example
do
    cd "$dir"
    MPLBACKEND=template python intro_example.py
    cd ..
done

