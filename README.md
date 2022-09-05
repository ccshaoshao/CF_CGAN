# CFGAN


# Environment setup

Clone the repo:
`https://github.com/ccshaoshao/CF_CGAN.git`


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

  
    # Download celeba-hq-dataset.zip from https://drive.google.com/file/d/1p-9I5cFGYG5N3x9TZtbov98joQgJhpsk/view?usp=sharing
    
    # unzip celeba-hq-dataset.zip

    
    # Run training
    python3 bin/train.py -cn cfgan-celeba data.batch_size=10
### Links
- Test images from the paper https://drive.google.com/drive/folders/1vAB2SoegeUIhwDeol4xRdtjni9en39Tb?usp=sharing
- The pre-trained model https://drive.google.com/file/d/1bs9b-gOlTMgSmCgBiCVx_wP3BleBdyZu/view?usp=sharing
- Our training logs are available at https://drive.google.com/drive/folders/1ufg5r5OdYwEphPuCGmJsJ-zx719IROIf?usp=sharing
