#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "network.h"
#include <iostream>
#include <vector>
#include <sndfile.h>

#include <torch/torch.h>
#include <iostream>

using namespace std::chrono;
namespace F = torch::nn::functional;

// Get it up on Github and Overleaf LATEX document
// Test

void print_array(double* arr, int start, int end, std::string name) {
  std::cout << name << std::endl;
  for (int i=start; i < end; i++) {
    printf("%i:%f\n", i, arr[i]);
  }
}

// Loads a .wav file into an array of doubles.
SF_INFO loadWavFile(char* filename, std::vector<double> &audioData) {
    // Open the .wav file
    SF_INFO sfinfo;
    SNDFILE* sndfile = sf_open(filename, SFM_READ, &sfinfo);
    if (!sndfile) {
        std::cerr << "Error opening input file: " << sf_strerror(sndfile) << std::endl;
        exit(1);
    }

    // Read the audio data
    const sf_count_t numFrames = sfinfo.frames;
    audioData = std::vector<double>(numFrames);

    sf_count_t numFramesRead = sf_read_double(sndfile, audioData.data(), numFrames);
    if (numFramesRead != numFrames) {
        std::cerr << "Error reading audio data: " << sf_strerror(sndfile) << std::endl;
        sf_close(sndfile);
        exit(1);
    }

    // Close the file and return the audio data
    sf_close(sndfile);
    return sfinfo;
}

// Converts a vector of doubles to an array of doubles.
void vectorToArray(std::vector<double> input, double* output, int max_length)
{
  if (max_length == -1)
  {
    max_length = input.size();
  }

  int min_length = std::min(int(input.size()), max_length);
  for (int i = 0; i < min_length; i++)
  {
    output[i] = input[i];
  }
}


void conv1d_unoptimized(double* input, double* weight, double bias, double* output,
            int input_size, int output_size, int kernel_size, int stride) 
{
    for (int i = 0; i < output_size; i++) 
    {
        output[i] += bias; // Add bias

        for (int j = 0; j < kernel_size; j++) 
        {
            if (weight[j] != 0)
            {
              int input_idx = i * stride + j;
              if (input_idx >= 0 && input_idx < input_size) 
              {
                  output[i] += input[input_idx] * weight[j];
              }
            }
        }
    }
}

void conv1d_loops_flipped(double* input, double* weight, double* output,
            int input_size, int output_size, int kernel_size, int stride) 
{
    int i, j, input_idx;

    for (j = 0; j < kernel_size; j++) 
    {
        if (weight[j] != 0)
        {
            for (i = 0; i < output_size; i++) 
            {
                input_idx = i * stride + j;
                output[i] += input[input_idx] * weight[j];
            }
        }
    }
}

void conv1d_shifted(double* input, double* weight, double bias, double* output,
            int input_size, int output_size, int kernel_size, int stride) 
{
    for (int i = 0; i < output_size; i++) 
    {
        output[i] += bias; // Add bias
        for (int j = 0; j < kernel_size; j++) 
        {
            if (weight[j] != 0)
            {
              int input_idx = (i+1) * stride + j;
              if (input_idx >= 0 && input_idx < input_size) 
              {
                  output[i] += input[input_idx] * weight[j];
              }
            }
        }
    }
}

double ReLU(double x) {
    return x > 0 ? x : 0;
}

void ReLU(double* x, int input_length, double* output) 
{
  for (int i=0; i < input_length; i++)
  {
    output[i] = x[i] > 0 ? x[i] : 0;
  }
}

// In place version of ReLU
void ReLU_(double* x, int input_length) 
{
  for (int i=0; i < input_length; i++)
  {
    x[i] = x[i] > 0 ? x[i] : 0;
  }
}

double Sigmoid(double x) {
    return (1 / (1 + exp(-x)));
}

void Sigmoid(double* x, int inp_size, double* output) 
{
    int i;
    for (i=0; i < inp_size; i++) 
    {
        output[i] = (1 / (1 + exp(-x[i])));
    }
}

// GLU activation function
// Returns an array of size inp_size / 2
void GLU(double* inp, int inp_size, double* output) {
    int i;
    int max = inp_size / 2;
    for (i=0; i < max; i++) {
        output[i] = inp[i] * Sigmoid(inp[i + max]);
    }
}

// GLU activation function
// Takes an N * K array, returns an N * K/2 array.
// K should be even

void GLU_split(double* inp, int N, int K, double* output) {
    int i;
    int j;
    int max = K / 2;

    //printf("GLU_split: N: %i, K: %i\n, max: %i", N, K, max);
    for (i=0; i < max; i += 1) 
    {
        for (j=0; j < N; j++)
        {
            output[i * N + j] = inp[i * N + j] * Sigmoid(inp[(max + i) * N + j]);
            //printf("GLU filling %i by matching %i with %i. Result:%f\n", i * N + j, i * N + j, (max + i) * N + j, output[i * N + j]);
        }
        //output[i] = inp[i] * Sigmoid(inp[i + max]);S
    }
    //printf("GLU_split: N: %i, K: %i\n, max: %i", N, K, max);
}

// Assumes that the padding of zeros is ZEROS, and the input size is VALID_LENGTH.
// Applies upsampling twice.
void double_upsample2_valid(double* inp, double* output, WorkingMemory* wm)
{
    // Before convolution, we need to include 112 / 2 = 56 zeros on each side of the input.
    // Use the padded_input array in the working memory struct. This should already have 56 zeros on each side.
    int i;
    for (i=ZEROS; i < VALID_LENGTH + ZEROS; i++)
    {
      wm->padded_upsample_input[i] = inp[i - ZEROS];
    }
    //printf("1\n");


    conv1d_shifted(wm->padded_upsample_input, kernel_upsample, 0, wm->upsample_working, VALID_LENGTH + 2*ZEROS, VALID_LENGTH, 112, 1);

    int output_size = (VALID_LENGTH) * 2;

    //printf("2\n");
    
    // Interweave
    for (int i=0; i < output_size; i++) 
    {
        if (i % 2 == 0) 
        {
            output[i] = inp[(i) / 2];
        }
        else 
        {
            output[i] = wm->upsample_working[(i) / 2];
        }
    }

    //printf("3\n");

    for (i = 0; i < ZEROS; i++)
    {
        wm->padded_upsample_double[i] = 0;
    }
    // Second upsample
    for (i=ZEROS; i < 2*VALID_LENGTH + ZEROS; i++) 
        wm->padded_upsample_double[i] = output[i - ZEROS];
    for (i = 2*VALID_LENGTH + ZEROS; i < 2*VALID_LENGTH + 2*ZEROS; i++)
    {
        wm->padded_upsample_double[i] = 0;
    }
    //printf("4\n");
    conv1d_shifted(wm->padded_upsample_double, kernel_upsample, 0, wm->upsample_working_double, 2 * (VALID_LENGTH + ZEROS), VALID_LENGTH * 2, 112, 1);

    int double_output_size = (VALID_LENGTH) * 4;
    
    //printf("5\n");
    // Interweave
    for (int i=0; i < double_output_size; i++) 
    {
        if (i % 2 == 0) 
        {
            output[i] = wm->padded_upsample_double[i / 2 + ZEROS];
        }
        else 
        {
            output[i] = wm->upsample_working_double[(i) / 2];
        }
    }

}

// Outputs a 2x downsampled version of the input.
// Length of output is inp_length / 2.
// Assumes that the padding of zeros is ZEROS
void downsample2_consts(double* inp, double* output, WorkingMemory* wm, int current_size)
{   
    // Divide into evens and odds.
    // Half input 1 is the evens, 2 is the odds.

    int i;
    for (i = 0; i < current_size; i++) 
    {
        if (i % 2 == 0)
        { // Evens
            wm->half_input_one[i / 2] = inp[i];
        }
        else 
        { // Odds
            wm->half_input_two[(i - 1) / 2] = inp[i];
        }
    }

    if (INPUT_SIZE % 2 == 1)
      wm->half_input_one[current_size/2 - 1] = 0;
    
    //print_array(wm.half_input_one, 0, current_size/2, "evens");
    //print_array(wm.half_input_two, 0, current_size/2, "odds");

    for (i = 0; i < ZEROS ; i++)
    {
      wm->padded_half_input[i] = 0;
      wm->padded_half_input[i + current_size/2 +  ZEROS] = 0;
    }
      

    // Pad the input to the conv1d function. 
    for (i=ZEROS; i < current_size/2 + ZEROS; i++) 
      wm->padded_half_input[i] = wm->half_input_two[i - ZEROS];
    
    for (i = 0; i < current_size/2 ; i++)
      wm->half_input_two[i] = 0;

    // Convolve the padded input with the kernel.
    // Don't need the odd input anymore as that's in wm.padded_half_input, so we use it as the output.
    //conv1d_loops_flipped(wm.padded_half_input, kernel_downsample, 0, wm.half_input_two, (current_size/2) + 2 * ZEROS, current_size / 2, 112, 1);
    conv1d_loops_flipped(wm->padded_half_input, kernel_downsample, wm->half_input_two, (current_size/2) + 2 * ZEROS, current_size / 2, 112, 1);
    
    //print_array(wm.half_input_two, 0, current_size/2, "Conv output");
    // Now add the even and conv_outputs together, then halve them.
    for (i=0; i < current_size / 2; i++) 
    {
        output[i] = (wm->half_input_one[i] + wm->half_input_two[i]) * 0.5;
    }
    //print_array(output, 0, current_size/2, "half_input_two");
    //exit(1);
}


at::Tensor upsample2Pytorch(at::Tensor input)
{
  auto out = F::conv1d(input.view({-1, 1, input.size(-1)}), kernel_upsample_tensor, F::Conv1dFuncOptions().stride(1).padding(56)).slice(-1, 1, input.size(-1) + 1);
  out = torch::stack({input.view({-1, 1, input.size(-1)}), out}, -1);
  out = out.view({1, 1, 2*input.size(-1)});
  return out;
}

at::Tensor downsample2Pytorch(at::Tensor input)
{
  // correct for odd lengths
  if (input.size(-1) % 2 != 0)
  {
    input = torch::pad(input, {0, 1}, "constant", 0);
  }
  at::Tensor evens = input.slice(-1, 0, input.size(-1), 2);
  at::Tensor odds = input.slice(-1, 1, input.size(-1), 2);

  int time = odds.size(-1);
  auto out = evens + F::conv1d(odds.view({-1, 1, odds.size(-1)}), kernel_downsample_tensor, F::Conv1dFuncOptions().stride(1).padding(ZEROS)).slice(-1, 0, time);
  out = out.view({odds.size(0), odds.size(1), -1}).mul(0.5);
  return out;
}

// Calculate multi-channel to multi-channel 1d convolution
// input has size [input_channels, input_size]
// Weight has size [output_channels, input_channels, kernel_size]
void conv1dChannels(double* input, double* weight, double* bias, double* output,
            int input_size, int output_size, int kernel_size, int input_channels, int output_channels, int stride)
{
  int in, out, i;
  // Initialize output channel to zeros
    for (i = 0; i < output_size * output_channels; i++)
    {
      output[i] = 0.0;
    }
    
    for (out = 0; out < output_channels; out++)
    {
        double* channel_out = &output[out * output_size];
        
        //std::cout << "out: " << out << std::endl;
        for (in = 0; in < input_channels; in++)
        {
            //std::cout << "in: " << out << std::endl;
            double* current_inp_channel = &input[in * input_size];
            double* current_weight = &weight[out * input_channels * kernel_size + in * kernel_size];
            
            //conv1d_unoptimized(current_inp_channel, current_weight, 0, channel_out, input_size, output_size, kernel_size, stride);
            //std::cout << "Convolving inp element " << (in * input_size) << " to " << (in + 1) * input_size - 1 << " with weight " << out * input_channels * kernel_size + in * kernel_size << std::endl;
            conv1d_loops_flipped(current_inp_channel, current_weight, channel_out, input_size, output_size, kernel_size, stride);
        }
        if (bias != NULL)
        {
          for (i = 0; i < output_size; i++) 
          {
            //std::cout << "Applying bias number " << out << " to " << i + (out * output_size) << std::endl;
            channel_out[i] += bias[out];
          }
        }
    }
}

// Calculate 1d transpose convolution
void conv1dTranspose(double* input, double* kernel, double* output,
                      int input_size, int kernel_size, int output_size,
                      int stride) {
    int i, k;
    
    // Perform Conv1dTranspose
    for (i = 0; i < input_size; i++)
    { 
      int i_prime = i * stride;
      for (k = 0; k < kernel_size; k++)
      {
        output[i_prime + k] += input[i] * kernel[k];
      }
    }
}

// Calculate multi-channel to multi-channel 1d convolution transpose
// input has size [input_channels, input_size]
// Weight has size [output_channels, input_channels, kernel_size]
void conv1dTransposeChannels(double* input, double* weight, double* bias, double* output,
            int input_size, int kernel_size, int input_channels, int output_channels, int stride)
{
    int i, out, in;
    int output_size = (input_size - 1) * stride + kernel_size;
    if (bias != NULL)
    {
        for (int out = 0; out < output_channels; out++)
        {
            for (int i = 0; i < output_size; i++)
            {
                output[out * output_size + i] = bias[out];
            }
        }
    }
    else
    {
      for (int i = 0; i < output_size * output_channels; i++)
      {
          output[i] = 0.0;
      }
    }

    for (int out = 0; out < output_channels; out++)
    {
        for (int in = 0; in < input_channels; in++)
        {
            double* current_inp_channel = &input[in * input_size];
            double* current_weight = &weight[in * output_channels * kernel_size + out * kernel_size];
            double* channel_out = &output[out * output_size];

            conv1dTranspose(current_inp_channel, current_weight, channel_out, input_size, kernel_size, output_size, stride);
        }
    }

    
}

// Sets weights to constant values.
void initializeDenoiserState(DenoiserState* ds, double sparsity, double setval)
{
  for (int i=0; i < 4 * 8; i++) {
    if (std::rand() % 100 < sparsity)
    {
      ds->encoder_0_0_weight[i] = 0;
    }
    else
    {
      ds->encoder_0_0_weight[i] = setval != -1 ? setval : std::rand() % 20000 / 10000.0 - 1;
    }
  }

  for (int i=0; i < 4; i++) {
    if (std::rand() % 100 < sparsity)
    {
      ds->encoder_0_0_bias[i] = 0;
    }
    else
    {
      ds->encoder_0_0_bias[i] = setval != -1 ? setval : std::rand() % 20000 / 10000.0 - 1;
    }
  }

  for (int i=0; i < 8 * 4 * 1; i++) {
    if (std::rand() % 100 < sparsity)
    {
      ds->encoder_0_2_weight[i] = 0;
    }
    else
    {
      ds->encoder_0_2_weight[i] = setval != -1 ? setval : std::rand() % 20000 / 10000.0 - 1;
    }
  }

  for (int i=0; i < 8; i++) {
    if (std::rand() % 100 < sparsity)
    {
      ds->encoder_0_2_bias[i] = 0;
    }
    else
    {
      ds->encoder_0_2_bias[i] = setval != -1 ? setval : std::rand() % 20000 / 10000.0 - 1;
    }
  }

  for (int i=0; i < 8 * 4 * 8; i++) {
    if (std::rand() % 100 < sparsity)
    {
      ds->encoder_1_0_weight[i] = 0;
    }
    else
    {
      ds->encoder_1_0_weight[i] = setval != -1 ? setval : std::rand() % 20000 / 10000.0 - 1;
    }
  }

  for (int i=0; i < 8; i++) {
    if (std::rand() % 100 < sparsity)
    {
      ds->encoder_1_0_bias[i] = 0;
    }
    else
    {
      ds->encoder_1_0_bias[i] = setval != -1 ? setval : std::rand() % 20000 / 10000.0 - 1;
    }
  }

  for (int i=0; i < 16 * 8 * 1; i++) {
    if (std::rand() % 100 < sparsity)
    {
      ds->encoder_1_2_weight[i] = 0;
    }
    else
    {
      ds->encoder_1_2_weight[i] = setval != -1 ? setval : std::rand() % 20000 / 10000.0 - 1;
    }
  }

  for (int i=0; i < 16; i++) {
    if (std::rand() % 100 < sparsity)
    {
      ds->encoder_1_2_bias[i] = 0;
    }
    else
    {
      ds->encoder_1_2_bias[i] = setval != -1 ? setval : std::rand() % 20000 / 10000.0 - 1;
    }
  }

  for (int i=0; i < 16 * 8 * 8; i++) {
    if (std::rand() % 100 < sparsity)
    {
      ds->encoder_2_0_weight[i] = 0;
    }
    else
    {
      ds->encoder_2_0_weight[i] = setval != -1 ? setval : std::rand() % 20000 / 10000.0 - 1;
    }
  }

  for (int i=0; i < 16; i++) {
    if (std::rand() % 100 < sparsity)
    {
      ds->encoder_2_0_bias[i] = 0;
    }
    else
    {
      ds->encoder_2_0_bias[i] = setval != -1 ? setval : std::rand() % 20000 / 10000.0 - 1;
    }
  }

  for (int i=0; i < 32 * 16 * 1; i++) {
    if (std::rand() % 100 < sparsity)
    {
      ds->encoder_2_2_weight[i] = 0;
    }
    else
    {
      ds->encoder_2_2_weight[i] = setval != -1 ? setval : std::rand() % 20000 / 10000.0 - 1;
    }
  }

  for (int i=0; i < 32; i++) {
    if (std::rand() % 100 < sparsity)
    {
      ds->encoder_2_2_bias[i] = 0;
    }
    else
    {
      ds->encoder_2_2_bias[i] = setval != -1 ? setval : std::rand() % 20000 / 10000.0 - 1;
    }
  }

  for (int i=0; i < 32 * 16 * 8; i++) {
    if (std::rand() % 100 < sparsity)
    {
      ds->encoder_3_0_weight[i] = 0;
    }
    else
    {
      ds->encoder_3_0_weight[i] = setval != -1 ? setval : std::rand() % 20000 / 10000.0 - 1;
    }
  }

  for (int i=0; i < 32; i++) {
    if (std::rand() % 100 < sparsity)
    {
      ds->encoder_3_0_bias[i] = 0;
    }
    else
    {
      ds->encoder_3_0_bias[i] = setval != -1 ? setval : std::rand() % 20000 / 10000.0 - 1;
    }
  }

  for (int i=0; i < 64 * 32 * 1; i++) {
    if (std::rand() % 100 < sparsity)
    {
      ds->encoder_3_2_weight[i] = 0;
    }
    else
    {
      ds->encoder_3_2_weight[i] = setval != -1 ? setval : std::rand() % 20000 / 10000.0 - 1;
    }
  }

  for (int i=0; i < 64; i++) {
    if (std::rand() % 100 < sparsity)
      ds->encoder_3_2_bias[i] = 0;
    else
      ds->encoder_3_2_bias[i] = setval != -1 ? setval : std::rand() % 20000 / 10000.0 - 1;
  }

  // Decoders
  for (int i=0; i < 64 * 32 * 1; i++) {
    if (std::rand() % 100 < sparsity)
      ds->decoder_0_0_weight[i] = 0;
    else
      ds->decoder_0_0_weight[i] = setval != -1 ? setval : std::rand() % 20000 / 10000.0 - 1;
  }

  for (int i=0; i < 64; i++) {
    if (std::rand() % 100 < sparsity)
      ds->decoder_0_0_bias[i] = 0;
    else
      ds->decoder_0_0_bias[i] = setval != -1 ? setval : std::rand() % 20000 / 10000.0 - 1;
  }

  for (int i=0; i < 32 * 16 * 8; i++) {
    if (std::rand() % 100 < sparsity)
      ds->decoder_0_2_weight[i] = 0;
    else
      ds->decoder_0_2_weight[i] = setval != -1 ? setval : std::rand() % 20000 / 10000.0 - 1;
  }

  for (int i=0; i < 16; i++) {
    if (std::rand() % 100 < sparsity)
      ds->decoder_0_2_bias[i] = 0;
    else
      ds->decoder_0_2_bias[i] = setval != -1 ? setval : std::rand() % 20000 / 10000.0 - 1;
  }

  for (int i=0; i < 32 * 16 * 1; i++) {
    if (std::rand() % 100 < sparsity)
      ds->decoder_1_0_weight[i] = 0;
    else
      ds->decoder_1_0_weight[i] = setval != -1 ? setval : std::rand() % 20000 / 10000.0 - 1;
  }

  for (int i=0; i < 32; i++) {
    if (std::rand() % 100 < sparsity)
      ds->decoder_1_0_bias[i] = 0;
    else
      ds->decoder_1_0_bias[i] = setval != -1 ? setval : std::rand() % 20000 / 10000.0 - 1;
  }

  for (int i=0; i < 16 * 8 * 8; i++) {
    if (std::rand() % 100 < sparsity)
      ds->decoder_1_2_weight[i] = 0;
    else
      ds->decoder_1_2_weight[i] = setval != -1 ? setval : std::rand() % 20000 / 10000.0 - 1;
  }

  for (int i=0; i < 8; i++) {
    if (std::rand() % 100 < sparsity)
      ds->decoder_1_2_bias[i] = 0;
    else
      ds->decoder_1_2_bias[i] = setval != -1 ? setval : std::rand() % 20000 / 10000.0 - 1;
  }

  for (int i=0; i < 16 * 8 * 1; i++) {
    if (std::rand() % 100 < sparsity)
      ds->decoder_2_0_weight[i] = 0;
    else
      ds->decoder_2_0_weight[i] = setval != -1 ? setval : std::rand() % 20000 / 10000.0 - 1;
  }

  for (int i=0; i < 16; i++) {
    if (std::rand() % 100 < sparsity)
      ds->decoder_2_0_bias[i] = 0;
    else
      ds->decoder_2_0_bias[i] = setval != -1 ? setval : std::rand() % 20000 / 10000.0 - 1;
  }

  for (int i=0; i < 8 * 4 * 8; i++) {
    if (std::rand() % 100 < sparsity)
      ds->decoder_2_2_weight[i] = 0;
    else
      ds->decoder_2_2_weight[i] = setval != -1 ? setval : std::rand() % 20000 / 10000.0 - 1;
  }

  for (int i=0; i < 4; i++) {
    if (std::rand() % 100 < sparsity)
      ds->decoder_2_2_bias[i] = 0;
    else
      ds->decoder_2_2_bias[i] = setval != -1 ? setval : std::rand() % 20000 / 10000.0 - 1;
  }

  for (int i=0; i < 8 * 4 * 1; i++) {
    if (std::rand() % 100 < sparsity)
      ds->decoder_3_0_weight[i] = 0;
    else
      ds->decoder_3_0_weight[i] = setval != -1 ? setval : std::rand() % 20000 / 10000.0 - 1;
  }

  for (int i=0; i < 8; i++) {
    if (std::rand() % 100 < sparsity)
      ds->decoder_3_0_bias[i] = 0;
    else
      ds->decoder_3_0_bias[i] = setval != -1 ? setval : std::rand() % 20000 / 10000.0 - 1;
  }

  for (int i=0; i < 4 * 1 * 8; i++) {
    if (std::rand() % 100 < sparsity)
      ds->decoder_3_2_weight[i] = 0;
    else
      ds->decoder_3_2_weight[i] = setval != -1 ? setval : std::rand() % 20000 / 10000.0 - 1;
  }

  for (int i=0; i < 1; i++) {
    if (std::rand() % 100 < sparsity)
      ds->decoder_3_2_bias[i] = 0;
    else
      ds->decoder_3_2_bias[i] = setval != -1 ? setval : std::rand() % 20000 / 10000.0 - 1;
  }
}

// Fills weight and bias of layer, turns off gradients.
void fillWeights(torch::nn::Conv1d *layer, double value, double sparsity)
{
  (*layer)->bias.requires_grad_(false);
  (*layer)->weight.requires_grad_(false);
  (*layer)->weight = (*layer)->weight.to(torch::kFloat64);
  (*layer)->bias = (*layer)->bias.to(torch::kFloat64);

  if (value != -1)
  {
    (*layer)->weight.fill_(value);
    (*layer)->bias.fill_(value);
  }
  else
  {
    for (int i = 0; i < (*layer)->weight.size(0); i++)
    {
      for (int j = 0; j < (*layer)->weight.size(1); j++)
      {
        for (int k = 0; k < (*layer)->weight.size(2); k++)
        {
          if (std::rand() % 100 < sparsity)
          {
            (*layer)->weight[i][j][k] = 0;
          }
          else
          {
            (*layer)->weight[i][j][k] = std::rand() % 20000 / 10000.0 - 1;
          }
        }
      }
    }
    //std::cout << "bias shape: " << (*layer)->bias.sizes() << std::endl;
    for (int i = 0; i < (*layer)->bias.size(0); i++)
    {
      if (std::rand() % 100 < sparsity)
      {
        (*layer)->bias[i] = 0;
      }
      else
      {
        (*layer)->bias[i] = std::rand() % 20000 / 10000.0 - 1;
      }
    }
  }
  
}

void fillWeights(torch::nn::LSTM *layer, double value, double sparsity)
{
  auto params = (*layer)->named_parameters();
  
  for (auto& param : params) {
    if (value == -1)
    {
      if (param.key().substr(0, 4) != "bias")
      {
        for (int i = 0; i < param.value().data().size(0); i++)
        {
          for (int j = 0; j < param.value().data().size(1); j++)
          {        
            if (std::rand() % 100 < sparsity)
            {
              param.value().data()[i][j] = 0;
            }
            else
            {
              param.value().data()[i][j] = std::rand() % 20000 / 10000.0 - 1;
            }
          }
        }
      }
      else
      {
        for (int i = 0; i < param.value().data().size(0); i++)
        {   
            if (std::rand() % 100 < sparsity)
            {
              param.value().data()[i] = 0;
            }
            else
            {
              param.value().data()[i] = std::rand() % 20000 / 10000.0 - 1;
            }
        }
      }
      

    }
    else
    {
      param.value().data().fill_(value);
    }
    param.value().data().requires_grad_(false);
  }
}
 
void fillWeights(torch::nn::ConvTranspose1d *layer, double value, double sparsity)
{
  (*layer)->bias.requires_grad_(false);
  (*layer)->weight.requires_grad_(false);
  (*layer)->weight = (*layer)->weight.to(torch::kFloat64);
  (*layer)->bias = (*layer)->bias.to(torch::kFloat64);

  if (value != -1)
  {
    (*layer)->weight.fill_(value);
    (*layer)->bias.fill_(value);
  }
  else
  {
    for (int i = 0; i < (*layer)->weight.size(0); i++)
    {
      for (int j = 0; j < (*layer)->weight.size(1); j++)
      {
        for (int k = 0; k < (*layer)->weight.size(2); k++)
        {
          if (std::rand() % 100 < sparsity)
          {
            (*layer)->weight[i][j][k] = 0;
          }
          else
          {
            //std::cout << "Filling with " << (*layer)->weight[i][j][k] << std::endl;
            (*layer)->weight[i][j][k] = std::rand() % 20000 / 10000.0 - 1;
            //std::cout << "Filling with " << (*layer)->weight[i][j][k] << std::endl;
          }
        }
      }
    }
    for (int i = 0; i < (*layer)->bias.size(0); i++)
    {
      if (std::rand() % 100 < sparsity)
      {
        (*layer)->bias[i] = 0;
      }
      else
      {
        (*layer)->bias[i] = std::rand() % 20000 / 10000.0 - 1;
      }
    }
  }
}


void randomizeWeights(double sparsity, DenoiserState* ds, DenoiserStatePyTorch* dspt)
{
  // Fill weights randomly.
  std::cout << "Sparsity is " << sparsity << std::endl;
  fillWeights(&dspt->decoder_0_0, -1, sparsity);
  fillWeights(&dspt->decoder_0_2, -1, sparsity);
  fillWeights(&dspt->decoder_1_0, -1, sparsity);
  fillWeights(&dspt->decoder_1_2, -1, sparsity);
  fillWeights(&dspt->decoder_2_0, -1, sparsity);
  fillWeights(&dspt->decoder_2_2, -1, sparsity);
  fillWeights(&dspt->decoder_3_0, -1, sparsity);
  fillWeights(&dspt->decoder_3_2, -1, sparsity);
  fillWeights(&dspt->lstm, -1, sparsity);
  initializeDenoiserState(ds, sparsity, -1);
}


void runDenoiserExperiments(DenoiserState* ds, WorkingMemory* wm, DenoiserStatePyTorch* dspt)
{
  int sparsity = 0;
  int step = 10;
  int iterations = 50;
  auto sparsity_percentages = torch::zeros({100 / step, 10}).to(torch::kFloat64);
  auto times = torch::zeros({100 / step, 12}).to(torch::kFloat64);
  std::srand(std::time(nullptr));

  for (sparsity = 0; sparsity < 100; sparsity += step)
  {
    for (int i=0; i < iterations; i++)
    {
      printf("Sparsity: %i\n", sparsity);
      printf("Iteration: %i\n", i);
      // Generate random input
      //double* inp = (double*)malloc(INPUT_SIZE * sizeof(double));
      double* output = (double*)malloc(INPUT_SIZE * sizeof(double));
      //for (int i=0; i < INPUT_SIZE; i++) {
        //inp[i] = (double)rand() / (double)RAND_MAX;
      //}
    char* filename = (char*) malloc(6 * sizeof(char));
    filename[0] = '0' + (i % 5 + 1);
    filename[1] = '.';
    filename[2] = 'w';
    filename[3] = 'a';
    filename[4] = 'v';
    filename[5] = '\0';
    std::vector<double> audioData;
    SF_INFO sfinfo = loadWavFile(filename, audioData);

    // Check if the vector is empty (indicating an error during loading)
    if (audioData.empty()) {
        std::cerr << "Failed to load .wav file." << std::endl;
        exit(1);
    }

      double* inp = (double*) malloc(10000 * sizeof(double));
     vectorToArray(audioData, inp, 10000);

      // Fill weights randomly.
      randomizeWeights(sparsity, ds, dspt);
      std::cout << "Weights randomized" << std::endl;

      // Run denoiser
      printf("Normalizing...\n");
      auto start = std::chrono::high_resolution_clock::now();
      double SD; // Standard deviation
      if (NORMALIZE)
      {
        double sum = 0;
        int i;
        for (i=0; i < INPUT_SIZE; i++) {
            sum += inp[i];
        }
        sum /= INPUT_SIZE; // This is the average.
        SD = 0;

        for (i = 0; i < INPUT_SIZE; ++i) {
            SD += pow(inp[i] - sum, 2);
        }
        // For some reason, pytorch uses N-1 instead of N.
        SD = sqrt(SD / (INPUT_SIZE - 1));

        // Also pad input
        for (i=0; i < VALID_LENGTH; i++) {
          wm->padded_input[i] = i < INPUT_SIZE ? (inp[i]) / (FLOOR + SD) : 0;
        }
      }
      else
      {
        int i;
        // Also pad input
        for (i=0; i < VALID_LENGTH; i++) {
          wm->padded_input[i] = i < INPUT_SIZE ? inp[i] : 0;
        }
        SD = 1; 
      }

      auto end = std::chrono::high_resolution_clock::now();
      double duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
      times[sparsity / step][0] += duration / iterations;

      if (RESAMPLE != 4)
      {
        printf("RESAMPLE: %i\n", RESAMPLE);
        printf("This value is not supported. It must be 4.\n");
        exit(1);
      }
      printf("Upsampling...\n");
      start = std::chrono::high_resolution_clock::now();
      double_upsample2_valid(wm->padded_input, wm->upsampled_input, wm);
      end = std::chrono::high_resolution_clock::now();
      duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
      times[sparsity / step][1] += duration / iterations;
      
      // Now we run each of the encoders in sequence.

      // encoder.0.0
      printf("Encoder.0.0\n");

      start = std::chrono::high_resolution_clock::now();
      int current_length = VALID_LENGTH - 1;
      conv1dChannels(wm->upsampled_input, ds->encoder_0_0_weight, ds->encoder_0_0_bias, wm->memory_grid, VALID_LENGTH * 4, current_length, KERNEL, 1, 4, STRIDE);
      // encoder.0.1
      //printf("Encoder.0.1\n");
      ReLU_(wm->memory_grid, current_length * 4);

      // encoder.0.2
      //printf("Encoder.0.2\n");
      conv1dChannels(wm->memory_grid, ds->encoder_0_2_weight, ds->encoder_0_2_bias, wm->memory_grid2, current_length, current_length, 1, 4, 8, 1);

      // encoder.0.3
      //printf("Encoder.0.3\n");
      GLU_split(wm->memory_grid2, current_length, 8, wm->memory_grid);
      
      // Copy to skips
      //printf("Copy to skips\n");
      for (int i=0; i < current_length * 4; i++) {
        wm->skip_1[i] = wm->memory_grid[i];
      }
      end = std::chrono::high_resolution_clock::now();
      duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
      times[sparsity / step][2] += duration / iterations;

      // Count sparsity
      int count = 0;
      for (int i=0; i < current_length * 4; i++) {
        if (wm->memory_grid[i] == 0)
          count++;
      }
      sparsity_percentages[sparsity / step][0] += (count / (double)(current_length * 4)) / iterations;

      // encoder.1.0
      printf("Encoder.1.0\n");
      
      start = std::chrono::high_resolution_clock::now();
      conv1dChannels(wm->memory_grid, ds->encoder_1_0_weight, ds->encoder_1_0_bias, wm->memory_grid2, current_length, current_length / 4 - 1, KERNEL, 4, 8, STRIDE);
      current_length = current_length / 4 - 1;

      // encoder.1.1
      //printf("Encoder.1.1\n");
      ReLU_(wm->memory_grid2, current_length * 8);

      // encoder.1.2
      //printf("Encoder.1.2\n");
      conv1dChannels(wm->memory_grid2, ds->encoder_1_2_weight, ds->encoder_1_2_bias, wm->memory_grid, current_length, current_length, 1, 8, 16, 1);

      // encoder.1.3
      //printf("Encoder.1.3\n");
      GLU_split(wm->memory_grid, current_length, 16, wm->memory_grid2);

      // Copy to skips
      //printf("Copy to skips\n");
      for (int i=0; i < current_length * 8; i++) {
        wm->skip_2[i] = wm->memory_grid2[i];
      }
      end = std::chrono::high_resolution_clock::now();
      duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
      times[sparsity / step][3] += duration / iterations;

      // Count sparsity
      count = 0;
      for (int i=0; i < current_length * 8; i++) {
        if (wm->memory_grid2[i] == 0)
          count++;
      }
      sparsity_percentages[sparsity / step][1] += count / (double)(current_length * 8) / iterations;

      // encoder.2.0
      printf("Encoder.2.0\n");
      start = std::chrono::high_resolution_clock::now();
      conv1dChannels(wm->memory_grid2, ds->encoder_2_0_weight, ds->encoder_2_0_bias, wm->memory_grid, current_length, current_length / 4 - 1, KERNEL, 8, 16, STRIDE);
      current_length = current_length / 4 - 1;

      // encoder.2.1
      //printf("Encoder.2.1\n");
      ReLU_(wm->memory_grid, current_length * 16);

      // encoder.2.2
      //printf("Encoder.2.2\n");
      conv1dChannels(wm->memory_grid, ds->encoder_2_2_weight, ds->encoder_2_2_bias, wm->memory_grid2, current_length, current_length, 1, 16, 32, 1);

      // encoder.2.3
      //printf("Encoder.2.3\n");
      GLU_split(wm->memory_grid2, current_length, 32, wm->memory_grid);

      // Copy to skips
      //printf("Copy to skips\n");
      for (int i=0; i < current_length * 16; i++) {
        wm->skip_3[i] = wm->memory_grid[i];
      }
      end = std::chrono::high_resolution_clock::now();
      duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
      times[sparsity / step][4] += duration / iterations;

      // Count sparsity
      count = 0;
      for (int i=0; i < current_length * 16; i++) {
        if (wm->memory_grid[i] == 0)
        {
            count++;
        }
      }
      sparsity_percentages[sparsity / step][2] += count / (double)(current_length * 16) / iterations;

      // encoder.3.0
      printf("Encoder.3.0\n");
      start = std::chrono::high_resolution_clock::now();
      conv1dChannels(wm->memory_grid, ds->encoder_3_0_weight, ds->encoder_3_0_bias, wm->memory_grid2, current_length, current_length / 4 - 1, KERNEL, 16, 32, STRIDE);
      current_length = current_length / 4 - 1;

      // encoder.3.1
      ReLU_(wm->memory_grid2, current_length * 32);

      // encoder.3.2
      conv1dChannels(wm->memory_grid2, ds->encoder_3_2_weight, ds->encoder_3_2_bias, wm->memory_grid, current_length, current_length, 1, 32, 64, 1);

        
      // encoder.3.3
      GLU_split(wm->memory_grid, current_length, 64, wm->memory_grid2);

      //print_array(wm->memory_grid2, 0, current_length * 32, "After");

      // Copy to skips
      for (int i=0; i < current_length * 32; i++) {
        wm->skip_4[i] = wm->memory_grid2[i];
      }
      end = std::chrono::high_resolution_clock::now();
      duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
      times[sparsity / step][5] += duration / iterations;

      // Count sparsity
      count = 0;
      for (int i=0; i < current_length * 32; i++) {
        if (wm->memory_grid2[i] == 0)
        {
          count++;
        }
      }
      sparsity_percentages[sparsity / step][3] += count / (double)(current_length * 32) / iterations;

      printf("Run LSTM\n");
      start = std::chrono::high_resolution_clock::now();
      at::Tensor input = torch::zeros({1, 32, current_length}).to(torch::kFloat64);
      for (int i=0; i < current_length * 32; i++) {
        input[0][i / current_length][i % current_length] = wm->memory_grid2[i];
      }
    //printf("Input shape: %i, %i, %i\n", input.size(0), input.size(1), input.size(2));
      
      // Run LSTM
      // Create hidden and cell state tensors

      printf("Run LSTM\n");
      
      auto hidden = torch::zeros({LAYERS, 1, HIDDEN}).to(torch::kFloat64);
      auto cell = torch::zeros({LAYERS, 1, HIDDEN}).to(torch::kFloat64);
      input = input.permute({2, 0, 1});
      input = std::get<0>(dspt->lstm->forward(input, std::make_tuple(hidden, cell)));
      input = input.permute({1, 2, 0});

      printf("Run LSTM\n");
      // Convert back to C array
      for (int i=0; i < input.size(1); i++) {
        for (int j = 0; j < input.size(2); j++)
        {
          wm->memory_grid2[i * input.size(2) + j] = input[0][i][j].item<double>();
        }
      }

      printf("Run LSTM\n");

      end = std::chrono::high_resolution_clock::now();
      duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
      times[sparsity / step][6] += duration / iterations;

      // Count sparsity
      count = 0;
      for (int i=0; i < current_length * 32; i++) {
        if (wm->memory_grid2[i] == 0)
        {
          count++;
        }
      }
      sparsity_percentages[sparsity / step][4] += count / (double)(current_length * 32) / iterations;

      printf("Decoder.0.0\n");
      start = std::chrono::high_resolution_clock::now();

      // Add skip4
      for (int i=0; i < current_length * 32; i++) {
        wm->memory_grid2[i] += wm->skip_4[i];
      }
      
      // decoder.0.0
      //printf("Decoder.0.0\n");
      conv1dChannels(wm->memory_grid2, ds->decoder_0_0_weight, ds->decoder_0_0_bias, wm->memory_grid, current_length, current_length, 1, 32, 65, 1);
      
      // encoder.0.1
      //printf("Decoder.0.1\n");
      GLU_split(wm->memory_grid, current_length, 64, wm->memory_grid2);  

      // decoder.0.2
      //printf("Decoder.0.2\n");
      conv1dTransposeChannels(wm->memory_grid2, ds->decoder_0_2_weight, ds->decoder_0_2_bias, wm->memory_grid, current_length, KERNEL, 32, 16, STRIDE);
      current_length = (current_length - 1) * STRIDE + KERNEL;
        
      // decoder.0.3
      //printf("Decoder.0.3\n");
      ReLU_(wm->memory_grid, current_length * 16);

      end = std::chrono::high_resolution_clock::now();
      duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
      times[sparsity / step][7] += duration / iterations;

      // Count sparsity
      count = 0;
      for (int i=0; i < current_length * 16; i++) {
        if (wm->memory_grid[i] == 0)
        {
          count++;
        }
      }
      sparsity_percentages[sparsity / step][5] += count / (double)(current_length * 16) / iterations;

      printf("Decoder.1.0\n");
      start = std::chrono::high_resolution_clock::now();

      // Add skip3
      for (int i=0; i < current_length * 16; i++) {
        wm->memory_grid[i] += wm->skip_3[i];
      }
      
      // decoder.1.0
      //printf("Decoder.1.0\n");
      conv1dChannels(wm->memory_grid, ds->decoder_1_0_weight, ds->decoder_1_0_bias, wm->memory_grid2, current_length, current_length, 1, 16, 32, 1);
      
      // encoder.1.1
      //printf("Decoder.1.1\n");
      GLU_split(wm->memory_grid2, current_length, 32, wm->memory_grid);

      // decoder.1.2
      //printf("Decoder.1.2\n");
      conv1dTransposeChannels(wm->memory_grid, ds->decoder_1_2_weight, ds->decoder_1_2_bias, wm->memory_grid2, current_length, KERNEL, 16, 8, STRIDE);
      current_length = (current_length - 1) * STRIDE + KERNEL;

      // decoder.1.3
      //printf("Decoder.1.3\n");
      ReLU_(wm->memory_grid2, current_length * 8);

      end = std::chrono::high_resolution_clock::now();
      duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
      times[sparsity / step][8] += duration / iterations;

      // Count sparsity
      count = 0;
      for (int i=0; i < current_length * 8; i++) {
        if (wm->memory_grid2[i] == 0)
        {
          count++;
        }
      }
      sparsity_percentages[sparsity / step][6] += count / (double)(current_length * 8) / iterations;

      printf("Decoder.2.0\n");
      start = std::chrono::high_resolution_clock::now();

      // Add skip2
      for (int i=0; i < current_length * 8; i++) {
        wm->memory_grid2[i] += wm->skip_2[i];
      }
    
      // decoder.2.0
      //printf("Decoder.2.0\n");
      conv1dChannels(wm->memory_grid2, ds->decoder_2_0_weight, ds->decoder_2_0_bias, wm->memory_grid, current_length, current_length, 1, 8, 16, 1);

      // decoder.2.1
      //printf("Decoder.2.1\n");
      GLU_split(wm->memory_grid, current_length, 16, wm->memory_grid2);
      
      // decoder.2.2
      //printf("Decoder.2.2\n");
      conv1dTransposeChannels(wm->memory_grid, ds->decoder_2_2_weight, ds->decoder_2_2_bias, wm->memory_grid2, current_length, KERNEL, 8, 4, STRIDE);
      current_length = (current_length - 1) * STRIDE + KERNEL;

      // decoder.2.3
      //printf("Decoder.2.3\n");
      ReLU_(wm->memory_grid2, current_length * 4);

      end = std::chrono::high_resolution_clock::now();
      duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
      times[sparsity / step][9] += duration / iterations;

      // Count sparsity
      count = 0;
      for (int i=0; i < current_length * 4; i++) {
        if (wm->memory_grid2[i] == 0)
        {
          count++;
        }
      }
      sparsity_percentages[sparsity / step][7] += count / (double)(current_length * 4) / iterations;

      printf("Decoder.3.0\n");
      start = std::chrono::high_resolution_clock::now();

      // Add skip1
      for (int i=0; i < current_length * 4; i++) {
        wm->memory_grid2[i] += wm->skip_1[i];
      }
    
      // decoder.3.0
      //printf("Decoder.3.0\n");
      conv1dChannels(wm->memory_grid2, ds->decoder_3_0_weight, ds->decoder_3_0_bias, wm->memory_grid, current_length, current_length, 1, 4, 8, 1);

      // decoder.3.1
      //printf("Decoder.2.1\n");
      GLU_split(wm->memory_grid, current_length, 8, wm->memory_grid2);

      // decoder.3.2
      //printf("Decoder.3.2\n");
      conv1dTransposeChannels(wm->memory_grid2, ds->decoder_3_2_weight, ds->decoder_3_2_bias, wm->memory_grid, current_length, KERNEL, 4, 1, STRIDE);
      current_length = (current_length - 1) * STRIDE + KERNEL;

      end = std::chrono::high_resolution_clock::now();

      duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
      times[sparsity / step][10] += duration / iterations;

      // Count sparsity
      count = 0;
      for (int i=0; i < current_length * 4; i++) {
        if (wm->memory_grid2[i] == 0)
        {
          count++;
        }
      }
      sparsity_percentages[sparsity / step][8] += count / (double)(current_length * 4) / iterations;

      printf("Downsampling\n");
      start = std::chrono::high_resolution_clock::now();

      if (RESAMPLE == 4)
      {
        //print_array(wm->memory_grid, 0, current_length * 1, "After");
        downsample2_consts(wm->memory_grid, wm->memory_grid2, wm, current_length);
        downsample2_consts(wm->memory_grid2, wm->memory_grid, wm, current_length / 2);
      }
      for (int i=0; i < INPUT_SIZE; i++) {
        output[i] = wm->memory_grid[i] * (SD);
      }

      end = std::chrono::high_resolution_clock::now();
      duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
      times[sparsity / step][11] += duration / iterations;

      // Count sparsity
      count = 0;
      for (int i=0; i < INPUT_SIZE; i++) {
        if (output[i] == 0)
        {
          count++;
        }
      }
      sparsity_percentages[sparsity / step][9] += count / (double)(INPUT_SIZE) / iterations;

      free(inp);
      free(output);
    }
  }

  std::cout << "Times: " << std::endl;
  std::cout << times << std::endl;

  std::cout << "Sparsity: " << std::endl;
  std::cout << sparsity_percentages << std::endl;
  
}

void runDenoiser(double* inp, DenoiserState* ds, WorkingMemory* wm, DenoiserStatePyTorch* dspt, double* output)
{
  // Run denoiser
  printf("Normalizing...\n");
  double SD; // Standard deviation
  if (NORMALIZE)
  {
    double sum = 0;
    int i;
    for (i=0; i < INPUT_SIZE; i++) {
        sum += inp[i];
    }
    std::cout << "Sum: " << sum << std::endl;
    sum /= INPUT_SIZE; // This is the average.
    SD = 0;

    for (i = 0; i < INPUT_SIZE; ++i) {
        SD += pow(inp[i] - sum, 2);
    }
    // For some reason, pytorch uses N-1 instead of N.
    SD = sqrt(SD / (INPUT_SIZE - 1));

    std::cout << "SD: " << SD << std::endl;

    // Also pad input
    for (i=0; i < VALID_LENGTH; i++) {
      wm->padded_input[i] = i < INPUT_SIZE ? (inp[i]) / (FLOOR + SD) : 0;
    }
  }
  else
  {
    int i;
    // Also pad input
    for (i=0; i < VALID_LENGTH; i++) {
      wm->padded_input[i] = i < INPUT_SIZE ? inp[i] : 0;
    }
    SD = 1; 
  }
  if (RESAMPLE != 4)
  {
    printf("RESAMPLE: %i\n", RESAMPLE);
    printf("This value is not supported. It must be 4.\n");
    exit(1);
  }
  printf("Upsampling data...\n");
  double_upsample2_valid(wm->padded_input, wm->upsampled_input, wm);

  // Now we run each of the encoders in sequence.

  // encoder.0.0
  printf("Encoder.0.0\n");

  int current_length = VALID_LENGTH - 1;
  conv1dChannels(wm->upsampled_input, ds->encoder_0_0_weight, ds->encoder_0_0_bias, wm->memory_grid, VALID_LENGTH * 4, current_length, KERNEL, 1, 4, STRIDE);
  
  // encoder.0.1
  //printf("Encoder.0.1\n");
  ReLU_(wm->memory_grid, current_length * 4);

  

  // encoder.0.2
  //printf("Encoder.0.2\n");
  conv1dChannels(wm->memory_grid, ds->encoder_0_2_weight, ds->encoder_0_2_bias, wm->memory_grid2, current_length, current_length, 1, 4, 8, 1);


  // encoder.0.3
  //printf("Encoder.0.3\n");
  GLU_split(wm->memory_grid2, current_length, 8, wm->memory_grid);

  
  // Copy to skips
  //printf("Copy to skips\n");
  for (int i=0; i < current_length * 4; i++) {
    wm->skip_1[i] = wm->memory_grid[i];
  }

  
  

  // encoder.1.0
  printf("Encoder.1.0\n");
  
  conv1dChannels(wm->memory_grid, ds->encoder_1_0_weight, ds->encoder_1_0_bias, wm->memory_grid2, current_length, current_length / 4 - 1, KERNEL, 4, 8, STRIDE);
  current_length = current_length / 4 - 1;



  // encoder.1.1
  //printf("Encoder.1.1\n");
  ReLU_(wm->memory_grid2, current_length * 8);



  // encoder.1.2
  //printf("Encoder.1.2\n");
  conv1dChannels(wm->memory_grid2, ds->encoder_1_2_weight, ds->encoder_1_2_bias, wm->memory_grid, current_length, current_length, 1, 8, 16, 1);



  // encoder.1.3
  //printf("Encoder.1.3\n");
  GLU_split(wm->memory_grid, current_length, 16, wm->memory_grid2);

  // Copy to skips
  //printf("Copy to skips\n");
  for (int i=0; i < current_length * 8; i++) {
    wm->skip_2[i] = wm->memory_grid2[i];
  }

  // encoder.2.0
  printf("Encoder.2.0\n");
  conv1dChannels(wm->memory_grid2, ds->encoder_2_0_weight, ds->encoder_2_0_bias, wm->memory_grid, current_length, current_length / 4 - 1, KERNEL, 8, 16, STRIDE);
  current_length = current_length / 4 - 1;

  

  // encoder.2.1
  //printf("Encoder.2.1\n");
  ReLU_(wm->memory_grid, current_length * 16);

  

  // encoder.2.2
  //printf("Encoder.2.2\n");
  conv1dChannels(wm->memory_grid, ds->encoder_2_2_weight, ds->encoder_2_2_bias, wm->memory_grid2, current_length, current_length, 1, 16, 32, 1);

  // encoder.2.3
  //printf("Encoder.2.3\n");
  GLU_split(wm->memory_grid2, current_length, 32, wm->memory_grid);

  // Copy to skips
  //printf("Copy to skips\n");
  for (int i=0; i < current_length * 16; i++) {
    wm->skip_3[i] = wm->memory_grid[i];
  }

  // encoder.3.0
  printf("Encoder.3.0\n");
  conv1dChannels(wm->memory_grid, ds->encoder_3_0_weight, ds->encoder_3_0_bias, wm->memory_grid2, current_length, current_length / 4 - 1, KERNEL, 16, 32, STRIDE);
  current_length = current_length / 4 - 1;

  // encoder.3.1
  ReLU_(wm->memory_grid2, current_length * 32);

  // encoder.3.2
  conv1dChannels(wm->memory_grid2, ds->encoder_3_2_weight, ds->encoder_3_2_bias, wm->memory_grid, current_length, current_length, 1, 32, 64, 1);

    
  // encoder.3.3
  GLU_split(wm->memory_grid, current_length, 64, wm->memory_grid2);


  // Copy to skips
  for (int i=0; i < current_length * 32; i++) {
    wm->skip_4[i] = wm->memory_grid2[i];
  }

  printf("Run LSTM\n");
  at::Tensor input = torch::zeros({1, 32, current_length}).to(torch::kFloat64);
  for (int i=0; i < current_length * 32; i++) {
    input[0][i / current_length][i % current_length] = wm->memory_grid2[i];
  }
  
  // Run LSTM
  // Create hidden and cell state tensors
  
  auto hidden = torch::zeros({LAYERS, 1, HIDDEN}).to(torch::kFloat64);
  auto cell = torch::zeros({LAYERS, 1, HIDDEN}).to(torch::kFloat64);
  input = input.permute({2, 0, 1});
  input = std::get<0>(dspt->lstm->forward(input, std::make_tuple(hidden, cell)));
  input = input.permute({1, 2, 0});

  // Convert back to C array
  for (int i=0; i < input.size(1); i++) {
    for (int j = 0; j < input.size(2); j++)
    {
      wm->memory_grid2[i * input.size(2) + j] = input[0][i][j].item<double>();
    }
  }
  

  // Add skip4
  for (int i=0; i < current_length * 32; i++) {
    wm->memory_grid2[i] += wm->skip_4[i];
  }
  
  // decoder.0.0
  printf("Decoder.0.0\n");
  conv1dChannels(wm->memory_grid2, ds->decoder_0_0_weight, ds->decoder_0_0_bias, wm->memory_grid, current_length, current_length, 1, 32, 64, 1);

  // encoder.0.1
  //printf("Decoder.0.1\n");
  GLU_split(wm->memory_grid, current_length, 64, wm->memory_grid2);  



  // decoder.0.2
  //printf("Decoder.0.2\n");
  conv1dTransposeChannels(wm->memory_grid2, ds->decoder_0_2_weight, ds->decoder_0_2_bias, wm->memory_grid, current_length, KERNEL, 32, 16, STRIDE);
  current_length = (current_length - 1) * STRIDE + KERNEL;

  // decoder.0.3
  //printf("Decoder.0.3\n");
  ReLU_(wm->memory_grid, current_length * 16);

  // Add skip3
  for (int i=0; i < current_length * 16; i++) {
    wm->memory_grid[i] += wm->skip_3[i];
  }

  // decoder.1.0
  printf("Decoder.1.0\n");
  conv1dChannels(wm->memory_grid, ds->decoder_1_0_weight, ds->decoder_1_0_bias, wm->memory_grid2, current_length, current_length, 1, 16, 32, 1);
  
  

  // encoder.1.1
  //printf("Decoder.1.1\n");
  GLU_split(wm->memory_grid2, current_length, 32, wm->memory_grid);

  

  // decoder.1.2
  //printf("Decoder.1.2\n");
  conv1dTransposeChannels(wm->memory_grid, ds->decoder_1_2_weight, ds->decoder_1_2_bias, wm->memory_grid2, current_length, KERNEL, 16, 8, STRIDE);
  current_length = (current_length - 1) * STRIDE + KERNEL;

  // decoder.1.3
  //printf("Decoder.1.3\n");
  ReLU_(wm->memory_grid2, current_length * 8);

  
  
  // Add skip2
  for (int i=0; i < current_length * 8; i++) {
    wm->memory_grid2[i] += wm->skip_2[i];
  }

  

  

  // decoder.2.0
  printf("Decoder.2.0\n");
  conv1dChannels(wm->memory_grid2, ds->decoder_2_0_weight, ds->decoder_2_0_bias, wm->memory_grid, current_length, current_length, 1, 8, 16, 1);

  // decoder.2.1
  //printf("Decoder.2.1\n");
  GLU_split(wm->memory_grid, current_length, 16, wm->memory_grid2);

  // decoder.2.2
  //printf("Decoder.2.2\n");
  conv1dTransposeChannels(wm->memory_grid2, ds->decoder_2_2_weight, ds->decoder_2_2_bias, wm->memory_grid, current_length, KERNEL, 8, 4, STRIDE);
  current_length = (current_length - 1) * STRIDE + KERNEL;

  // decoder.2.3
  //printf("Decoder.2.3\n");
  ReLU_(wm->memory_grid, current_length * 4);
  

  // Add skip1
  for (int i=0; i < current_length * 4; i++) {
    wm->memory_grid[i] += wm->skip_1[i];
  }

  // decoder.3.0
  printf("Decoder.3.0\n");
  conv1dChannels(wm->memory_grid, ds->decoder_3_0_weight, ds->decoder_3_0_bias, wm->memory_grid2, current_length, current_length, 1, 4, 8, 1);
    

  // decoder.3.1
  //printf("Decoder.2.1\n");
  GLU_split(wm->memory_grid2, current_length, 8, wm->memory_grid);

  
  // decoder.3.2
  //printf("Decoder.3.2\n");
  conv1dTransposeChannels(wm->memory_grid, ds->decoder_3_2_weight, ds->decoder_3_2_bias, wm->memory_grid2, current_length, KERNEL, 4, 1, STRIDE);
  current_length = (current_length - 1) * STRIDE + KERNEL;

  if (RESAMPLE == 4)
  {
    downsample2_consts(wm->memory_grid2, wm->memory_grid, wm, current_length);
    downsample2_consts(wm->memory_grid, wm->memory_grid2, wm, current_length / 2);
  }

  for (int i=0; i < INPUT_SIZE; i++) {
    output[i] = wm->memory_grid2[i] * (SD);
  }
  //print_array(output, 0, 100, "Memory Grid");
  //std::cout << "Current length: " << current_length << std::endl;
  //exit(1);
}

at::Tensor runDenoiserPytorch(at::Tensor input, DenoiserStatePyTorch* ds)
{
  std::cout << "Starting denoiser" << std::endl;
  if (input.dim() == 2)
  {
    input = input.unsqueeze(1);
  }

  double std = 1;

  if (NORMALIZE)
  {
    std = input.std(-1, true).item<double>();
    input = input / (FLOOR + std);
  }
  input = at::pad(input, {0, VALID_LENGTH - INPUT_SIZE}, "constant", 0);
  
  if (RESAMPLE == 2)
  {
    input = upsample2Pytorch(input);
  }
  else if (RESAMPLE == 4)
  {
    input = upsample2Pytorch(input);
    input = upsample2Pytorch(input);
  }
  
  // Encoder_0_0
  input = ds->encoder_0_0->forward(input);
  // Encoder_0_1
  input = F::relu(input);
  // Encoder_0_2
  input = ds->encoder_0_2->forward(input);
  // Encoder_0_3
  auto skip_0 = F::glu(input, 1);

  // Encoder_1_0
  input = ds->encoder_1_0->forward(skip_0);
  // Encoder_1_1
  input = F::relu(input);
  // Encoder_1_2
  input = ds->encoder_1_2->forward(input);
  // Encoder_1_3
  auto skip_1 = F::glu(input, 1);

  // Encoder_2_0
  input = ds->encoder_2_0->forward(skip_1);
  // Encoder_2_1
  input = F::relu(input);
  // Encoder_2_2
  input = ds->encoder_2_2->forward(input);
  // Encoder_2_3
  auto skip_2 = F::glu(input, 1);

  // Encoder_3_0
  input = ds->encoder_3_0->forward(skip_2);
  // Encoder_3_1
  input = F::relu(input);
  // Encoder_3_2
  input = ds->encoder_3_2->forward(input);
  // Encoder_3_3
  auto skip_3 = F::glu(input, 1);

  // Create hidden and cell state tensors
  auto hidden = torch::zeros({LAYERS, 1, HIDDEN}).to(torch::kFloat64);
  auto cell = torch::zeros({LAYERS, 1, HIDDEN}).to(torch::kFloat64);
  input = skip_3.permute({2, 0, 1});

  // Run LSTM
  input = std::get<0>(ds->lstm->forward(input, std::make_tuple(hidden, cell)));
  input = input.permute({1, 2, 0});
  // Pre 0. Use skip_3

  input = input + skip_3;
  // Decoder_0_0
  input = ds->decoder_0_0->forward(input);
  // Decoder_0_1
  input = F::glu(input, 1);
  // Decoder_0_2
  input = ds->decoder_0_2->forward(input);
  // Decoder_0_3
  input = F::relu(input);

  input = input + skip_2;
  // Decoder_1_0
  input = ds->decoder_1_0->forward(input);
  // Decoder_1_1
  input = F::glu(input, 1);
  // Decoder_1_2
  input = ds->decoder_1_2->forward(input);
  // Decoder_1_3
  input = F::relu(input);

  input = input + skip_1;
  // Decoder_2_0
  input = ds->decoder_2_0->forward(input);
  // Decoder_2_1
  input = F::glu(input, 1);
  // Decoder_2_2
  input = ds->decoder_2_2->forward(input);
  // Decoder_2_3
  input = F::relu(input);

  input = input + skip_0;
  // Decoder_3_0
  input = ds->decoder_3_0->forward(input);
  // Decoder_3_1
  input = F::glu(input, 1);
  // Decoder_3_2
  input = ds->decoder_3_2->forward(input);


  if (RESAMPLE == 2)
  {
    input = downsample2Pytorch(input);
  }
  else if (RESAMPLE == 4)
  {
    input = downsample2Pytorch(input);
    input = downsample2Pytorch(input);
  }
  input = input.slice(-1, 0, INPUT_SIZE);
  input = input * (std);
  return input;
}

// Creates a workingMemory struct and allocates memory for all of its members.
void initializeWorkingMemory(WorkingMemory* wm)
{  
    int i;
    std::cout << "Initializing working memory" << std::endl;

    for (i=0; i < VALID_LENGTH; i++) {
      wm->padded_input[i] = 0;
      wm->upsample_working[i] = 0;
    }

    for (i=0; i < 2 * VALID_LENGTH; i++) {
      wm->upsample_working_double[i] = 0;
      wm->half_input_one[i] = 0;
      wm->half_input_two[i] = 0;
    }

    for (i = 0; i < VALID_LENGTH + 2*ZEROS; i++)
    {
      wm->padded_upsample_input[i] = 0;
    }

    for (i = 0; i < 2 * VALID_LENGTH + 2*ZEROS; i++)
    {
      wm->padded_upsample_double[i] = 0;
      wm->padded_half_input[i] = 0;
    }

    for (int i = 0; i < VALID_LENGTH * 4; i++)
    {
      wm->skip_1[i] = 0;
      wm->skip_2[i] = 0;
      wm->skip_3[i] = 0;
      wm->skip_4[i] = 0;
      wm->upsampled_input[i] = 0;
    }
        
    for (int i = 0; i < VALID_LENGTH * 8; i++)
    {
      wm->memory_grid[i] = 0;
      wm->memory_grid2[i] = 0;
    }
}

void runSparsityExperiment()
{
  int iterations = 10;
  int step = 5;

  float sum = 0;
  int stride = 4;
  int kernel_size = 64;
  int input_size = 1000000;
  int output_length = (input_size - kernel_size) / stride + 1;
  auto times = torch::zeros({iterations, 10});
  for (int percentage = 0; percentage < 100; percentage += step)
  {
    for (int i = 0; i < iterations; i++)
    {
      // Create input tensor
      auto input = torch::rand({1, 1, input_size});
      input.to_sparse();

      ///int zero = 0;

      for (int j = 0; j < input_size; j++)
      {
          input[0][0][j] = std::rand() % 1000 / 1000.0;
      }

      torch::nn::Conv1d conv(torch::nn::Conv1dOptions(1, 1, kernel_size).stride(stride).padding(0).bias(false));
      conv->weight.requires_grad_(false);
      for (int j = 0; j < kernel_size; j++)
      {
        if (std::rand() % 100 < percentage)
        {
          conv->weight[0][0][j] = 0;
        }
        else
        {
          conv->weight[0][0][j] = std::rand() % 1000 / 1000.0;
        }
      }
      
      auto start = high_resolution_clock::now();

      auto out = conv->forward(input);
      auto end = high_resolution_clock::now();

      auto duration = duration_cast<microseconds>(end - start);
      
      sum += duration.count();
      times[i][percentage / 10] = duration.count();
    }
    sum /= iterations;
    std::cout << "Time taken to run convolution with " << percentage << " percent zeros is = " << sum << " microseconds." << std::endl;
  }

  auto averages = torch::zeros({10});
  for (int i = 0; i < 10; i++)
  {
    averages[i] = times.select(1, i).mean();
  }
  std::cout << averages << std::endl;
  
}

// Compares the speeds of two different conv1d functions.
void compareSpeed()
{
  int iterations = 20;
  int step = 5;

  float sum = 0;
  int stride = 4;
  int kernel_size = 64;
  int input_size = 1000000;
  int output_length = (input_size - kernel_size) / stride + 1;

  double* alg_one_times = (double*) malloc((100 / step) * sizeof(double));
  double* alg_two_times = (double*) malloc((100 / step) * sizeof(double));

  for (int percentage = 0; percentage < 100; percentage += step)
  {
    for (int i = 0; i < iterations; i++)
    {
      // Create input tensor
      double* input = (double*) malloc(input_size * sizeof(double));

      for (int j = 0; j < input_size; j++)
      {
          input[j] = std::rand() % 10000 / 10000.0;
      }
      
      double* kernel = (double*) malloc(kernel_size * sizeof(double));
      double* output = (double*) malloc(output_length * sizeof(double));

      for (int j = 0; j < kernel_size; j++)
      {
          if (std::rand() % 100 < percentage)
          {
            kernel[j] = 0;
          }
          else
          {
            kernel[j] = std::rand() % 10000 / 10000.0;
          }
      }
      
      auto start = high_resolution_clock::now();

      //conv1d_loops_flipped(input, kernel, 0, output, input_size, output_length, kernel_size, stride);
      auto end = high_resolution_clock::now();

      auto duration = duration_cast<microseconds>(end - start);
      
      sum += duration.count();

      free(input);
      free(kernel);
      free(output);
    }
    sum /= iterations;
    alg_one_times[percentage / step] = sum;
    std::cout << "Time taken to run convolution with " << percentage << " percent zeros is = " << sum << " microseconds." << std::endl;
  }

  for (int percentage = 0; percentage < 100; percentage += step)
  {
    for (int i = 0; i < iterations; i++)
    {
      double* input = (double*) malloc(input_size * sizeof(double));

      for (int j = 0; j < input_size; j++)
      {
          input[j] = std::rand() % 10000 / 10000.0;
      }
      
      double* kernel = (double*) malloc(kernel_size * sizeof(double));
      double* output = (double*) malloc(output_length * sizeof(double));

      for (int j = 0; j < kernel_size; j++)
      {
          if (std::rand() % 100 < percentage)
          {
            kernel[j] = 0;
          }
          else
          {
            kernel[j] = std::rand() % 10000 / 10000.0;
          }
      }
      
      auto start = high_resolution_clock::now();

      conv1d_unoptimized(input, kernel, 0, output, input_size, output_length, kernel_size, stride);
      auto end = high_resolution_clock::now();

      auto duration = duration_cast<microseconds>(end - start);
      
      sum += duration.count();

      free(input);
      free(kernel);
      free(output);
    }
    sum /= iterations;
    alg_two_times[percentage / step] = sum;
    std::cout << "Time taken to run convolution unoptimized with " << percentage << " percent zeros is = " << sum << " microseconds." << std::endl;
  }

  std::cout << "Algorithm 1: " << std::endl;
  for (int i = 0; i < 100 / step; i++)
  {
    std::cout << (i*step) << "%: " << alg_one_times[i] << std::endl;
  }
  std::cout << "Base unoptimized algorithm: " << std::endl;
  for (int i = 0; i < 100 / step; i++)
  {
    std::cout << (i*step) << "%: " << alg_two_times[i] << std::endl;
  }
}

// Loads weights to fill the denoiser state.
void loadWeightsFromDisk(char* filename, DenoiserState* ds, DenoiserStatePyTorch* dspt)
{
  std::cout << "Loading weight file" << std::endl;
  FILE* file = fopen(filename, "r");

  for (int i = 0; i < 4 * 8; i++)
  {
    fscanf(file, "%lf", &ds->encoder_0_0_weight[i]);
  }
    
  for (int i = 0; i < 4; i++)
    fscanf(file, "%lf", &ds->encoder_0_0_bias[i]);

  for (int i = 0; i < 4 * 8; i++)
    fscanf(file, "%lf", &ds->encoder_0_2_weight[i]);
  for (int i = 0; i < 8; i++)
    fscanf(file, "%lf", &ds->encoder_0_2_bias[i]);


  for (int i = 0; i < 8 * 4 * 8; i++)
    fscanf(file, "%lf", &ds->encoder_1_0_weight[i]);
  for (int i = 0; i < 8; i++)
    fscanf(file, "%lf", &ds->encoder_1_0_bias[i]);

  for (int i = 0; i < 16 * 8; i++)
    fscanf(file, "%lf", &ds->encoder_1_2_weight[i]);
  for (int i = 0; i < 16; i++)
    fscanf(file, "%lf", &ds->encoder_1_2_bias[i]);


  for (int i = 0; i < 16 * 8 * 8; i++)
    fscanf(file, "%lf", &ds->encoder_2_0_weight[i]);
  for (int i = 0; i < 16; i++)
    fscanf(file, "%lf", &ds->encoder_2_0_bias[i]);

  for (int i = 0; i < 32 * 16; i++)
    fscanf(file, "%lf", &ds->encoder_2_2_weight[i]);
  for (int i = 0; i < 32; i++)
    fscanf(file, "%lf", &ds->encoder_2_2_bias[i]);


  for (int i = 0; i < 32 * 16 * 8; i++)
    fscanf(file, "%lf", &ds->encoder_3_0_weight[i]);
  for (int i = 0; i < 32; i++)
    fscanf(file, "%lf", &ds->encoder_3_0_bias[i]);

  for (int i = 0; i < 64 * 32; i++)
    fscanf(file, "%lf", &ds->encoder_3_2_weight[i]);
  for (int i = 0; i < 64; i++)
    fscanf(file, "%lf", &ds->encoder_3_2_bias[i]);

  
  for (int i = 0; i < 64 * 32 * 1; i++)
    fscanf(file, "%lf", &ds->decoder_0_0_weight[i]);
  for (int i = 0; i < 64; i++)
    fscanf(file, "%lf", &ds->decoder_0_0_bias[i]);

  for (int i = 0; i < 32 * 16 * 8; i++)
    fscanf(file, "%lf", &ds->decoder_0_2_weight[i]);
  for (int i = 0; i < 16; i++)
    fscanf(file, "%lf", &ds->decoder_0_2_bias[i]);


  for (int i = 0; i < 32 * 16 * 1; i++)
    fscanf(file, "%lf", &ds->decoder_1_0_weight[i]);
  for (int i = 0; i < 32; i++)
    fscanf(file, "%lf", &ds->decoder_1_0_bias[i]);

  for (int i = 0; i < 16 * 8 * 8; i++)
    fscanf(file, "%lf", &ds->decoder_1_2_weight[i]);
  for (int i = 0; i < 8; i++)
    fscanf(file, "%lf", &ds->decoder_1_2_bias[i]);


  for (int i = 0; i < 16 * 8 * 1; i++)
    fscanf(file, "%lf", &ds->decoder_2_0_weight[i]);
  for (int i = 0; i < 16; i++)
    fscanf(file, "%lf", &ds->decoder_2_0_bias[i]);

  for (int i = 0; i < 8 * 4 * 8; i++)
    fscanf(file, "%lf", &ds->decoder_2_2_weight[i]);
  for (int i = 0; i < 4; i++)
    fscanf(file, "%lf", &ds->decoder_2_2_bias[i]);

  
  for (int i = 0; i < 8 * 4 * 1; i++)
    fscanf(file, "%lf", &ds->decoder_3_0_weight[i]);
  for (int i = 0; i < 8; i++)
    fscanf(file, "%lf", &ds->decoder_3_0_bias[i]);

  for (int i = 0; i < 4 * 1 * 8; i++)
    fscanf(file, "%lf", &ds->decoder_3_2_weight[i]);
  for (int i = 0; i < 1; i++)
    fscanf(file, "%lf", &ds->decoder_3_2_bias[i]);
  
  // Load LSTM
  std::cout << "Loading LSTM" << std::endl;

  auto elements = dspt->lstm->named_parameters();
  double value;

  // lstm.lstm.weight_ih_l0
  auto element = elements[0];
  element.value().to(torch::kFloat64);
  element.value().requires_grad_(false);

  for (int i = 0; i < 128; i++)
  {
    for (int j = 0; j < 32; j++)
    {
      fscanf(file, "%lf", &value);
      element.value()[i][j] = value;
    }
  }

  // lstm.lstm.weight_hh_l0
  element = elements[1];
  element.value().to(torch::kFloat64);
  element.value().requires_grad_(false);

  for (int i = 0; i < 128; i++)
  {
    for (int j = 0; j < 32; j++)
    {
      fscanf(file, "%lf", &value);
      element.value()[i][j] = value;
    }
  }

  // lstm.lstm.bias_ih_l0
  element = elements[2];
  element.value().to(torch::kFloat64);
  element.value().requires_grad_(false);

  for (int i = 0; i < 128; i++)
  {
    fscanf(file, "%lf", &value);
    element.value()[i] = value;
  }

  // lstm.lstm.bias_hh_l0
  element = elements[3];
  element.value().to(torch::kFloat64);
  element.value().requires_grad_(false);

  for (int i = 0; i < 128; i++)
  {
    fscanf(file, "%lf", &value);
    element.value()[i] = value;
  }

  // lstm.lstm.weight_ih_l1
  element = elements[4];
  element.value().to(torch::kFloat64);
  element.value().requires_grad_(false);

  for (int i = 0; i < 128; i++)
  {
    for (int j = 0; j < 32; j++)
    {
      fscanf(file, "%lf", &value);
      element.value()[i][j] = value;
    }
  }

  // lstm.lstm.weight_hh_l1
  element = elements[5];
  element.value().to(torch::kFloat64);
  element.value().requires_grad_(false);

  for (int i = 0; i < 128; i++)
  {
    for (int j = 0; j < 32; j++)
    {
      fscanf(file, "%lf", &value);
      element.value()[i][j] = value;
    }
  }

  // lstm.lstm.bias_ih_l1
  element = elements[6];
  element.value().to(torch::kFloat64);
  element.value().requires_grad_(false);

  for (int i = 0; i < 128; i++)
  {
    fscanf(file, "%lf", &value);
    element.value()[i] = value;
  }

  // lstm.lstm.bias_hh_l1
  element = elements[7];
  element.value().to(torch::kFloat64);
  element.value().requires_grad_(false);

  for (int i = 0; i < 128; i++)
  {
    fscanf(file, "%lf", &value);
    element.value()[i] = value;
  }

  dspt->lstm->to(torch::kFloat64);

  std::cout << "Loaded LSTM" << std::endl;

  fclose(file);
  
}

void initalizeDenoiserState(DenoiserStatePyTorch* ds)
{
  fillWeights(&ds->encoder_0_0, 0.1, 0);
  fillWeights(&ds->encoder_0_2, 0.1, 0);

  fillWeights(&ds->encoder_1_0, 0.1, 0);
  fillWeights(&ds->encoder_1_2, 0.1, 0);

  fillWeights(&ds->encoder_2_0, 0.1, 0);
  fillWeights(&ds->encoder_2_2, 0.1, 0);

  fillWeights(&ds->encoder_3_0, 0.1, 0);
  fillWeights(&ds->encoder_3_2, 0.1, 0);

  fillWeights(&ds->lstm, 0.1, 0);
  ds->lstm->to(torch::kFloat64);

  fillWeights(&ds->decoder_0_0, 0.1, 0);
  fillWeights(&ds->decoder_0_2, 0.1, 0);

  fillWeights(&ds->decoder_1_0, 0.1, 0);
  fillWeights(&ds->decoder_1_2, 0.1, 0);

  fillWeights(&ds->decoder_2_0, 0.1, 0);
  fillWeights(&ds->decoder_2_2, 0.1, 0);

  fillWeights(&ds->decoder_3_0, 0.1, 0);
  fillWeights(&ds->decoder_3_2, 0.1, 0);

}


int main(int argc, char *argv[]) {

  int operation = 5; // 0 = denoiser, 1 = convolution tests, 2/3 = speed/sparsity tests, 4 = load wav file

  if (argc > 1)
  {
    operation = atoi(argv[1]);
  }
  else
  {
    printf("No operation specified. Defaulting to denoiser.\n");
    operation = 0;
  }

  // Run Denoiser
  if (operation == 0)
  {
    double* input = (double*) malloc(INPUT_SIZE * sizeof(double));
    double* output = (double*) malloc(INPUT_SIZE * sizeof(double));
    for (int i=0; i < INPUT_SIZE; i++) 
    {
        input[i] = 0;
        output[i] = 0;
    }

     SF_INFO sfinfo;

    if (argc > 2)
    {
      std::cout << "Loading wav file" << std::endl;
      std::vector<double> audioData;
      sfinfo = loadWavFile(argv[2], audioData);
      vectorToArray(audioData, input, INPUT_SIZE);
      std::cout << "Loaded wav file" << std::endl;
    }
    else
    {
      std::cout << "No wav file specified. Defaulting to sine wave." << std::endl;
      for (int i=0; i < INPUT_SIZE; i++) 
      {
          input[i] = sin(i * .1);
      }
    }

    DenoiserState *ds = (DenoiserState*) malloc(sizeof(DenoiserState));
    WorkingMemory *wm = (WorkingMemory*) malloc(sizeof(WorkingMemory));
    DenoiserStatePyTorch dspt = DenoiserStatePyTorch();
    mallocDenoiserState(ds);
    mallocWorkingMemory(wm);

    std::cout << "Allocated memory" << std::endl;

    if (argc > 3)
    {
      std::cout << "Loading weights from disk" << std::endl;
      loadWeightsFromDisk(argv[3], ds, &dspt);
    }
    else
    {
      std::cout << "No weight file specified. Defaulting to random weights." << std::endl;
      initializeDenoiserState(ds, 0, 0.1);
      initalizeDenoiserState(&dspt);
      randomizeWeights(0, ds, &dspt);
    }

    initializeWorkingMemory(wm);

    std::cout << "Starting" << std::endl;
    
    runDenoiser(input, ds, wm, &dspt, output);

    std::cout << "End" << std::endl;
    
    freeDenoiserState(ds);
    freeWorkingMemory(wm);

    // Save output as .wav file.
    // Add a readme.md

    if (argc > 4 )
    {
      std::cout << "Saving file to " << argv[4] << std::endl;

      SNDFILE * outfile = sf_open(argv[4], SFM_WRITE, &sfinfo);
      sf_count_t count = sf_write_double(outfile, &output[0], INPUT_SIZE);
      sf_write_sync(outfile);
      sf_close(outfile);
    }

    free(input);
    free(output);
    std::cout << "Done" << std::endl;
    return 0;
  }
  else if (operation == 1) // Run convolution tests
  {

    std::cout << "Starting convolution tests" << std::endl;
    const int KERNEL_SIZE = 64;
    const int OUTPUT_LENGTH = (INPUT_SIZE - KERNEL_SIZE) / STRIDE + 1;
    double* input = (double*) malloc(INPUT_SIZE * sizeof(double));
    double* kernel = (double*) malloc(KERNEL_SIZE * sizeof(double));
    double* output = (double*) malloc(OUTPUT_LENGTH * sizeof(double));

    std::cout << "Allocated memory" << std::endl;
    for (int i=0; i < INPUT_SIZE; i++) {
        input[i] = sin(i * .1);
    }

    for (int i=0; i < KERNEL_SIZE; i++) {
        kernel[i] = sin(i * -.1);
    }

    //conv1d_unrolled(input, kernel, 0, output, INPUT_SIZE, OUTPUT_LENGTH, KERNEL_SIZE, STRIDE);

    for (int i=0; i < OUTPUT_LENGTH; i++) {
        printf("%f\n", output[i]);
    }

    free(input);
    free(kernel);
    free(output);
  }
  else if (operation == 2)
  {
    compareSpeed();
  }
  else if (operation == 3)
  {
    int outchannels = 1;
    int inchannels = 4;
    double* input = (double*) malloc(INPUT_SIZE * inchannels * sizeof(double));
    //double* output = (double*) malloc(INPUT_SIZE * 4 * sizeof(double));

    for (int i=0; i < INPUT_SIZE; i++) 
    {
        input[i] = 0;
        //output[i] = 0;
    }

     SF_INFO sfinfo;

    if (argc > 2)
    {
      std::cout << "Loading wav file" << std::endl;
      std::vector<double> audioData;
      sfinfo = loadWavFile(argv[2], audioData);
      vectorToArray(audioData, input, INPUT_SIZE * inchannels);
      std::cout << "Loaded wav file" << std::endl;
      //print_array(input, 0, 100, "input");
    }
    else
    {
      std::cout << "No wav file specified. Defaulting to sine wave." << std::endl;
      for (int i=0; i < INPUT_SIZE; i++) 
      {
          input[i] = sin(i * .1);
      }
    }

    DenoiserState *ds = (DenoiserState*) malloc(sizeof(DenoiserState));
    WorkingMemory *wm = (WorkingMemory*) malloc(sizeof(WorkingMemory));
    DenoiserStatePyTorch dspt = DenoiserStatePyTorch();
    mallocDenoiserState(ds);
    mallocWorkingMemory(wm);

    std::cout << "Allocated memory" << std::endl;

    if (argc > 3)
    {
      std::cout << "Loading weights from disk" << std::endl;
      loadWeightsFromDisk(argv[3], ds, &dspt);
    }
    else
    {
      std::cout << "No weight file specified. Defaulting to random weights." << std::endl;
      initializeDenoiserState(ds, 0, 0.1);
      initalizeDenoiserState(&dspt);
      randomizeWeights(0, ds, &dspt);
    }

    initializeWorkingMemory(wm);
    
    int kernel_size = 8;
    double* kernel = (double*) malloc(inchannels * outchannels * kernel_size * sizeof(double));
    int stride = 4;
    double* bias = (double*) malloc(outchannels * sizeof(double));

    for (int k = 0; k < inchannels; k++)
    {
       for (int j = 0; j < outchannels; j++)
        {
          for (int i = 0; i < kernel_size; i++)
          {
            kernel[k*outchannels*kernel_size + j*kernel_size + i] = sin((i+1)*(j+1)*(k+1));
          }
        }
    }


   
    
    for (int i = 0; i < outchannels; i++)
    {
      bias[i] = sin(i+1);
    }

    std::cout << "Starting" << std::endl;

    int output_size = (INPUT_SIZE - 1) * stride + kernel_size;
    double* output = (double*) malloc(output_size * outchannels * sizeof(double));
    for (int i = 0; i < output_size * outchannels; i++)
    {
      output[i] = 0;
    }
    //conv1d_gpt(input, ds->encoder_0_0_weight, ds->encoder_0_0_bias, output, INPUT_SIZE, 1, 1, 8, 1);
    //conv1dTranspose(input, kernel, output, INPUT_SIZE, kernel_size, output_size, stride);
    conv1dTransposeChannels(input, kernel, bias, output,
            INPUT_SIZE, kernel_size, inchannels, outchannels, stride);
    print_array(kernel, 0, inchannels*outchannels*kernel_size, "kernel");
    print_array(bias, 0, outchannels, "bias");
    print_array(output, 0, 100, "output");

    std::cout << "End" << std::endl;
    
    freeDenoiserState(ds);
    freeWorkingMemory(wm);

    free(input);
    free(output);
    std::cout << "Done" << std::endl;
    return 0;
  }
  else if (operation == 4)
  {
    std::cout << "Loading wav file" << std::endl;
    SF_INFO sfinfo;
    std::vector<double> audioData;
    sfinfo = loadWavFile(argv[2], audioData);
    std::cout << "Loaded wav file" << std::endl;
    std::cout << audioData.size() << std::endl;
    std::cout << audioData[0] << std::endl;
    std::cout << audioData[1] << std::endl;
    std::cout << audioData[2] << std::endl;
    //std::cout << "Saving wav file" << std::endl;
    //saveWavFile(audioData, argv[3]);
    //std::cout << "Saved wav file" << std::endl;
  }
  return 0;
}


