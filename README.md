# TSDTVOS
<img src="https://github.com/zhouweii234/TSDTVOS/blob/main/img/overview.jpg?raw=true" width="900">

This is the implementation of TSDTVOS.    
This code is based on MiVOS: [[link]](https://github.com/hkchengrex/Mask-Propagation).

## Display
![image](https://github.com/zhouweii234/TSDTVOS/blob/main/gold-fish.gif)


## Dependencies
+ PyTorch 1.7.1  
+ torchvision 0.8.2  
+ OpenCV 4.2.0  
+ progressbar  

## DataSets
+ DAVIS  
download DAVIS2016 and DAVIS2017(480p) from [[link]](https://davischallenge.org/)
```
path/to/DAVIS
├── 2016
│   ├── Annotations
│   └── ...
└── 2017
    ├── trainval
    │   ├── Annotations
    │   └── ...
    ├── test-dev
        ├── Annotations
        └── ...
```

+ YouTubeVOS  
download YouTubeVOS2018 valid from [[link]](https://drive.google.com/uc?id=1-QrceIl5sUNTKz7Iq0UsWC6NLZq7girr)  
download YouTubeVOS2018 valid_all_frames from [[link]](https://drive.google.com/uc?id=1yVoHM6zgdcL348cFpolFcEl4IC1gorbV)  
download YouTubeVOS2019 valid from [[link]](https://drive.google.com/uc?id=1o586Wjya-f2ohxYf9C1RlRH-gkrzGS8t)  
download YouTubeVOS2019 valid_all_frames from [[link]](https://drive.google.com/uc?id=1rWQzZcMskgpEQOZdJPJ7eTmLCBEIIpEN)  
```
path/to/YouTubeVOS
├── 2018
│   ├── all_frames
│   │   └── valid_all_frames
│   └── valid
└── 2019
    ├── all_frames
    │   └── valid_all_frames
    └── valid
```

## Trained model
+ Download pre-trained weights into ```./saves```  
[[weights]](https://drive.google.com/file/d/1KXrzCenlzojbgiuOXIKD_c8IHN_RIXG2/view?usp=sharing)

## Code
+ DAVIS-2016 validation set  
```
python eval_davis_2016.py --davis_path [path/to/DAVIS-2016] --output [path/to/output]
```

+ DAVIS-2017 validation set  
```
python eval_davis.py --davis_path [path/to/DAVIS-2017] --split val/testdev --output [path/to/output]
```

+ DAVIS-2017 test-dev set  
```
python eval_davis.py --davis_path [path/to/DAVIS-2017] --split val/testdev --output [path/to/output] --conf_thr 0.3
```

+ YouTubeVOS-2018 validation set  
```
python eval_youtube.py --yv_path [path/to/YouTubeVOS-2018] --output [path/to/output]
```

+ YouTubeVOS-2019 validation set  
```
python eval_youtube.py --yv_path [path/to/YouTubeVOS-2019] --output [path/to/output]
```

## Pre-computed Results
We provide pre-computed results for benchmark sets.

[DAVIS-16-val](https://drive.google.com/file/d/15mLLZQz0L3sv6wKcOwLMGl12PRgUrjWQ/view?usp=sharing)  
[DAVIS-17-val](https://drive.google.com/file/d/1zkimQ5K9zWl4yntcBUEewsJz7mKnb9WZ/view?usp=sharing)  
[YouTube-VOS-18-valid](https://drive.google.com/file/d/1k_BEO5_CRcj1H6W05oFmJWXdLsAQFOZM/view?usp=sharing)  
[YouTube-VOS-19-valid](https://drive.google.com/file/d/1EvBJHjJMcfkqfEzP50LkcGuSqFZ9FTSX/view?usp=sharing)  
[DAVIS-17-testdev](https://drive.google.com/file/d/1eDFJ0cOAGQxFLn1ERDSXIgsIZQZextdh/view?usp=sharing)
