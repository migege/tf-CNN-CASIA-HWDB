# tf-CNN-CASIA-HWDB

A tensorflow implementation of recognition of handwritten Chinese characters.

## Output
```
# testing on 140 mostly used characters
python main.py inference 0
========================================================
CR(1):0.88513	CR(5):0.97975	CR(10):0.99038

# testing on 3755-character GB2312 characters
python main.py inference 1
============================================================
CR(1):0.93609	CR(5):0.98773	CR(10):0.99322
```
