# NVCaffe-Converter
Convert NVCaffe model and weights into BVLC Caffe format

## Dependencies
* python2.7 （delete `from __future__ import print_function` for python3）
* pycaffe
* protobuf
* argparse

## How to use?

Convert both model and weights

```
python converter.py --model your_prototxt --weights your_weights --merge_bn False --save_dir your_save_dir
```

Convert model only, just do not set --weights

```
python converter.py --model your_prototxt --merge_bn False --save_dir your_save_dir
```

Convert and merge bn

```
python converter.py --model your_prototxt --weights your_weights --save_dir your_save_dir
```
