# CS 229 Final Project
# Autumn 2019
# Authors: Carlos Gomez-Uribe, Viktor Krapivin, Grace Woods

Multi-method Fitting Finite Mixtures (of Gaussians)

1) Generate finite mixtures of multivariate gaussians
    this is done by running generateData.py
    if necessary modify the generation parameters in it
    parameters are hard coded in generateData_simple and fit_EM is unmainted file that fits those parameters

2) We use several 1st-Order Learning Algorithm: SGD
    2a) Expectation-Maximization (EM) Algorithm
    2b) Delta Method (Outlined in Project Paper)
        2b.i) Approximate l ~ l_0
        2b.ii) Approximate l ~ l_0 + l_2
    2c) The variants we have include using an approximation to the inverse in the algorithm, a weighted method described in paper that avoids it, and a full inverse method

3) Included also is code that runs simulations for multiple parameters and produces a chart how well they did

Notes on Python Environment used:
# Name                    Version                   Build  Channel
_libgcc_mutex             0.1                        main  
blas                      1.0                         mkl  
ca-certificates           2019.8.28                     0  
certifi                   2019.9.11                py36_0  
cycler                    0.10.0                   py36_0  
dbus                      1.13.6               h746ee38_0  
expat                     2.2.6                he6710b0_0  
fontconfig                2.13.0               h9420a91_0  
freetype                  2.9.1                h8a8886c_1  
glib                      2.56.2               hd408876_0  
gst-plugins-base          1.14.0               hbbd80ab_1  
gstreamer                 1.14.0               hb453b48_1  
icu                       58.2                 h9c2bf20_1  
intel-openmp              2019.4                      243  
jpeg                      9b                   h024ee3a_2  
kiwisolver                1.1.0            py36he6710b0_0  
libedit                   3.1.20181209         hc058e9b_0  
libffi                    3.2.1                hd88cf55_4  
libgcc-ng                 9.1.0                hdf63c60_0  
libgfortran-ng            7.3.0                hdf63c60_0  
libpng                    1.6.37               hbc83047_0  
libstdcxx-ng              9.1.0                hdf63c60_0  
libtiff                   4.0.10               h2733197_2  
libuuid                   1.0.3                h1bed415_2  
libxcb                    1.13                 h1bed415_1  
libxml2                   2.9.9                hea5a465_1  
matplotlib                2.2.2            py36hb69df0a_2  
mkl                       2018.0.3                      1  
mkl_fft                   1.0.6            py36h7dd41cf_0  
mkl_random                1.0.1            py36h4414c95_1  
ncurses                   6.1                  he6710b0_1  
numpy                     1.15.0           py36h1b885b7_0  
numpy-base                1.15.0           py36h3dfced4_0  
olefile                   0.46                     py36_0  
openssl                   1.0.2t               h7b6447c_1  
pcre                      8.43                 he6710b0_0  
pillow                    6.2.0            py36h34e0f95_0  
pip                       10.0.1                   py36_0  
pyparsing                 2.4.2                      py_0  
pyqt                      5.9.2            py36h05f1152_2  
python                    3.6.6                h6e4f718_2  
python-dateutil           2.8.0                    py36_0  
pytz                      2019.3                     py_0  
qt                        5.9.6                h8703b6f_2  
readline                  7.0                  h7b6447c_5  
scipy                     1.1.0            py36hd20e5f9_0  
setuptools                41.4.0                   py36_0  
sip                       4.19.8           py36hf484d3e_0  
six                       1.12.0                   py36_0  
sqlite                    3.30.0               h7b6447c_0  
tk                        8.6.8                hbc83047_0  
tornado                   6.0.3            py36h7b6447c_0  
wheel                     0.33.6                   py36_0  
xz                        5.2.4                h14c3975_4  
zlib                      1.2.11               h7b6447c_3  
zstd                      1.3.7                h0b5b093_0