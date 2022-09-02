# CFGAN


# Environment setup

Clone the repo:
`https://github.com/ccshaoshao/CFGAN.git`


Python virtualenv:

    ```
    virtualenv inpenv --python=/usr/bin/python3
    source inpenv/bin/activate
    pip install torch==1.8.0 torchvision==0.9.0
    
    cd CFGAN
    pip install -r requirements.txt 
    ```



# Train and Eval


```
cd CFGAN
export TORCH_HOME=$(pwd) && export PYTHONPATH=$(pwd)
```

    
## CelebA
On the host machine:

    # Make shure you are in CFGAN folder
    cd CFGAN
    export TORCH_HOME=$(pwd) && export PYTHONPATH=$(pwd)

    # Download CelebA-HQ dataset
    # Download data256x256.zip from https://drive.google.com/drive/folders/11Vz0fqHS2rXDb5pprgTjpD7S2BAJhi1P
    
    # unzip 

    
    # Run training
    python3 bin/train.py -cn cfgan-celeba data.batch_size=10
### Links
- Test images from the paper https://disk.yandex.ru/d/xKQJZeVRk5vLlQ
- The pre-trained model https://disk.yandex.ru/d/EgqaSnLohjuzAg
- Our training logs are available at https://disk.yandex.ru/d/9Bt1wNSDS4jDkQ
