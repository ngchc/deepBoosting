Deep Boosting for Image Denoising
====
**Official implementation** for Deep Boosting Framework introduced in the following paper <br/>
Chang Chen, Zhiwei Xiong, Xinmei Tian, Feng Wu. Deep Boosting for Image Denoising. In ECCV 2018. <br/>

## Requirements
Anaconda>=4.2.0 (Python 3.5) <br/>
TensorFlow>=1.4.0 <br/>

## Train the model
Usage example to train/evaluate a new model <br/>
```
cd train && cat train400.tfrecord.tar.gz.* | tar -xzv
python train.py && python inference.py
```
## Test the pre-trained models
Usage example to non-blind gray-level Gaussian image denoising <br/>
```
cd dn-nonblind-gray && python eval.py
```
Usage example to blind gray-level Gaussian image denoising <br/>
```
cd dn-blind-gray && python eval.py
```
Usage example to blind color Gaussian image denoising <br/>
```
cd dn-blind-color && python eval.py
```
Usage example to JPEG image deblocking
```
cd db-gray && python eval.py
```
Usage example to in-domain real-world image denoising
```
cd dn-real-in && python eval.py
```
Usage example to cross-domain real-world image denoising
```
cd dn-real-cross && python eval.py
```
## Dataset
For training and validation <br/>
Download RID.tar.gz.0~5 from <br/>
[http://pan.bitahub.com/index.php?mod=shares&  
sid=eTJ2bFFQR3BzTm5FTGdjdXFBUnl2Y3htd3puWjFwRDc4SU9vSXc](http://pan.bitahub.com/index.php?mod=shares&sid=eTJ2bFFQR3BzTm5FTGdjdXFBUnl2Y3htd3puWjFwRDc4SU9vSXc)
```
cat RID.tar.gz.* | tar -xzv
```
For testing <br/>
```
cd datas/Set60
```
