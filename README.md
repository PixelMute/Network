# Network
C Denoiser Network

How to install/use

First, install the C++ API for Pytorch. This is only used for the LSTM layer, and I hope to eliminate that later. Instructions on how to do so are at https://pytorch.org/cppdocs/installing.html . 
It also needs the library sndfile. For easiest use, install the dev version with "apt-get install libsndfile-dev" , or from the website at https://libsndfile.github.io/libsndfile/ .

Next, make a directory inside the PyTorch installation called "network" and place "network.cpp", the cmake file, the weights, and the .wav files inside.
Create a "build" folder inside "network", then run "cmake .." inside it to set it up as a build folder. There are two options for compiling, which is either release (has -o2) or debug (has -g).
Then drop "network.h" into the include folder, or anywhere else "network.cpp" can reach it. To compile, run "cmake --build . --config Build" while inside the network folder.

The program is run with "./network.cpp [operation] [wav input file] [weights file] [wav output file]".

EG: "./network.cpp 0 1.wav weights.txt out.wav"

Operation refers to one of four operations the program can do. 0 is the standard denoiser, and the only one that's useful for non-debugging reasons. 1, 2, and 3 are all tests of convolution algoriths.

Wav input file is what file to load. I have included 1-5.wav, which are 5 random files from the Valintini data set that I picked for testing.

Weights file refers to a .txt file that contains the weight dumps of a network. I have included a simple Python script to create these from other models. I have also included the model I used for testing, which is weights.txt. This is just a very simple, bad model I made to test if my lottery ticket algorithm was working, so it might increase noise in some files.

Wav output file is just the filename to save the results to. If this is not provided, it will not save weights.

You can edit the input size by changing the constant value "INPUT_SIZE", located on line 7 of network.h. If the .wav file is too long, it will be truncated. If it is too short, it will be padded with 0 values.
