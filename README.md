## About TriU-Net
TriU-Net is a hybrid neural beamformer for multi-channel speech enhancement. In this repo, we share the weights of TriU-Net trained on a 9-channel, 4-cm interval uniform linear array.
For more details, please refer to https://pubs.aip.org/asa/jasa/article/153/6/3378/2897718/Three-stage-hybrid-neural-beamformer-for-multi

## How to Use
1. install python 3.10
2. install Torch == 1.12.1+cu116 (Other versions should also work but have not been tested), soundfile, numpy, tqdm
3. run "python demo.py"
4. the enhanced wav is 'wavs/enhanced.wav'

## Other enhanced wav
Other enhanced wav is available at https://ioa-audio.github.io/2023/03/20/TriU-Net_demo/.



If you find this repository is helpful to you, please cite

Kelan Kuang, Feiran Yang, Junfeng Li, Jun Yang; Three-stage hybrid neural beamformer for multi-channel speech enhancement. J Acoust Soc Am 1 June 2023; 153 (6): 3378–.

@article{10.1121/10.0019802,
    author = {Kuang, Kelan and Yang, Feiran and Li, Junfeng and Yang, Jun},
    title = "{Three-stage hybrid neural beamformer for multi-channel speech enhancement}",
    journal = {The Journal of the Acoustical Society of America},
    volume = {153},
    number = {6},
    pages = {3378-},
    year = {2023},
    month = {06},
    doi = {10.1121/10.0019802},
    url = {https://doi.org/10.1121/10.0019802},
    eprint = {https://pubs.aip.org/asa/jasa/article-pdf/153/6/3378/18009324/3378\_1\_10.0019802.pdf},
}


