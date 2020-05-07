# D4Net: De-Deformation Defect Detection Network for Non-Rigid Products with Large Patterns
by Xuemiao Xu^, Jiaxing Chen^, Huaidong Zhang\*, and Wing~W. Y. Ng\* (^ joint 1st author, * joint corresponding author)[[paper link]()]

This implementation is written by Jiaxing Chen at the South China University of Technology.

## Citation

@article{xu2020d4net,   
&nbsp;&nbsp;&nbsp;&nbsp;  title={D4Net: De-Deformation Defect Detection Network for Non-Rigid Products with Large Patterns},    
&nbsp;&nbsp;&nbsp;&nbsp;  author={Xuemiao Xu, Jiaxing Chen, Huaidong Zhang, and Wing~W. Y. Ng},    
&nbsp;&nbsp;&nbsp;&nbsp;  journal={IEEE Transactions on Circuits and Systems for Video Technology},    
&nbsp;&nbsp;&nbsp;&nbsp;  year  = {2020},    
&nbsp;&nbsp;&nbsp;&nbsp;  publisher={Elsevier}    
}

## LFLP Dataset

Due to the influence of COVID-19, the LFLP dataset will be released after the author returns to school. [[LFLP dataset link]()]

## Trained Model

You can download the trained model which is reported in our paper at  [Google Drive](https://drive.google.com/file/d/1knTpVXt3gKGxqHMZQKz-T1q0r3TlsQxf/view?usp=sharing).

## Requirement

- Python 2.7
- PyTorch 0.4.0
- torchvision
- numpy

## Training

1. Set the path of pretrained ResNeXt model in resnext/config.py
2. Set the path of LFLP dataset in config.py
3. Run by `python train.py`

*Hyper-parameters* of training were gathered at the beginning of *train.py* and you can conveniently change them as you need.

## Testing

1. Put the trained model in ckpt/d4net
2. Run by `python infer.py`

*Settings* of testing were gathered at the beginning of *infer.py* and you can conveniently change them as you need.