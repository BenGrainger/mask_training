conda create -n segment-env python=3.11 -y
pip install napari[all]
pip install torch torchvision torchaudio  # for Linux, pip, python CUDA 12.4 
pip install git+https://github.com/guiwitz/napari-convpaint.git@a30a35334dc38495444007e32f03393366792838
pip install ipympl  # for interactive plots
pip install imagecodecs  # to read full res tiff files


pip install matplotlib.pyplot


pip install cuml-cu12 # for main_cuml
