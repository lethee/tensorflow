/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <cstdarg>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>
#include <queue>
#include <sstream>

#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/model.h"
#include "tensorflow/contrib/lite/string_util.h"
#include "tensorflow/contrib/lite/tools/mutable_op_resolver.h"

using namespace std;

#ifdef TFLITE_CUSTOM_OPS_HEADER
void RegisterSelectedOps(::tflite::MutableOpResolver* resolver);
#endif

#define LOG(x) std::cerr
#define CHECK(x) if (!(x)) { LOG(ERROR) << #x << "failed"; exit(1); }

namespace tensorflow {
namespace benchmark_tflite_model {

// Returns the top N confidence values over threshold in the provided vector,
// sorted by confidence in descending order.
static void GetTopN(
    const float* prediction,
    const int prediction_size,
    const int num_results, const float threshold,
    std::vector<std::pair<float, int> >* top_results) {
  // Will contain top N results in ascending order.
  std::priority_queue<std::pair<float, int>,
      std::vector<std::pair<float, int> >,
      std::greater<std::pair<float, int> > > top_result_pq;

  const long count = prediction_size;
  for (int i = 0; i < count; ++i) {
    const float value = prediction[i];

    // Only add it if it beats the threshold and has a chance at being in
    // the top N.
    if (value < threshold) {
      continue;
    }

    top_result_pq.push(std::pair<float, int>(value, i));

    // If at capacity, kick the smallest value out.
    if (top_result_pq.size() > num_results) {
      top_result_pq.pop();
    }
  }

  // Copy to output vector and reverse into descending order.
  while (!top_result_pq.empty()) {
    top_results->push_back(top_result_pq.top());
    top_result_pq.pop();
  }
  std::reverse(top_results->begin(), top_results->end());
}

std::unique_ptr<tflite::FlatBufferModel> model;
std::unique_ptr<tflite::Interpreter> interpreter;

void InitImpl(const std::string& graph, const std::vector<int>& sizes,
              const std::string& input_layer_type, int num_threads) {
  CHECK(graph.c_str());

  // Read the label list
  std::vector<std::string> label_strings;
  std::ifstream t;
  t.open("labels.txt");
  std::string line;
  while(t){
    std::getline(t, line);
    label_strings.push_back(line);
  }
  t.close();


  model = tflite::FlatBufferModel::BuildFromFile(graph.c_str());
  if (!model) {
    LOG(FATAL) << "Failed to mmap model " << graph << endl;
  }
  LOG(INFO) << "Loaded model " << graph << endl;
  model->error_reporter();
  LOG(INFO) << "resolved reporter" << endl;

#ifdef TFLITE_CUSTOM_OPS_HEADER
  tflite::MutableOpResolver resolver;
  RegisterSelectedOps(&resolver);
#else
  tflite::ops::builtin::BuiltinOpResolver resolver;
#endif

  tflite::InterpreterBuilder(*model, resolver)(&interpreter);
  if (!interpreter) {
    LOG(FATAL) << "Failed to construct interpreter" << endl;
  }

  if (num_threads != -1) {
    interpreter->SetNumThreads(num_threads);
  }

  int input = interpreter->inputs()[0];

  if (input_layer_type != "string") {
    interpreter->ResizeInputTensor(input, sizes); // allocate 224 x 224 x 3
  }

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    LOG(FATAL) << "Failed to allocate tensors!" << endl;
  }


  // read input image

  unsigned char input_buffer[224 * 224 * 3];
  memset(input_buffer, 0, sizeof(input_buffer));
  printf("input_buffer size: %zu\n", sizeof(input_buffer));

  FILE *fp = fopen("grace_hopper.rgb", "rb");
  if (fp == nullptr) {
    std::cerr << "input file open error" << endl;
    return;
  }
  size_t read_bytes = fread(input_buffer, sizeof(input_buffer), 1, fp);
  printf("read_bytes size: %zu\n", sizeof(input_buffer));
  fclose(fp);
  printf("%u, %u, %u\n", input_buffer[150525], input_buffer[150526], input_buffer[150527]);

  // feed input data

  const float input_mean = 127.5f;
  const float input_std = 127.5f;

  float* out = interpreter->typed_tensor<float>(input);
  for (int y = 0; y < 224; ++y) {
    unsigned char* in_row = input_buffer + 3 * 224 * y;
    float* out_row = out + 3 * 224 * y;

    for (int x = 0; x < 224; ++x) {
       float* out_pixel = out_row + 3 * x;
       unsigned char* in_pixel = in_row + 3 * x;

       for (int c = 0; c < 3; ++c) {
         out_pixel[c] = (in_pixel[c] - input_mean) / input_std;
       }
    }
  }
  printf("%f, %f, %f\n", out[150525], out[150526], out[150527]);

  // Invoke

  std::cerr << "Invoke" << std::endl;
  if (interpreter->Invoke() != kTfLiteOk) {
    std::cerr << "Invoke Error" << endl;
    return;
  }
  std::cerr << "Invoke End" << std::endl;

  // output

  float* output = interpreter->typed_output_tensor<float>(0);
  const int output_size = 1000;
  const int kNumResults = 5;
  const float kThreshold = 0.01f;
  std::vector<std::pair<float, int> > top_results;
  GetTopN(output, output_size, kNumResults, kThreshold, &top_results);

  // print output with label

  std::stringstream ss;
  ss.precision(3);
  for (const auto& result : top_results) {
    const float confidence = result.first;
    const int index = result.second;

    ss << index << ", " << confidence << " - " << label_strings[index] << endl;
  }
  std::cout << ss.str();

  // end
  
  std::cout << "--- end ---" << std::endl;
}

int Main(int argc, char** argv) {
  InitImpl("mobilenet_v1_1.0_224.tflite",
           {1, 224, 224, 3}, "float", 1);
  return 0;
}

}  // namespace benchmark_tflite_model
}  // namespace tensorflow

int main(int argc, char** argv) {
  return tensorflow::benchmark_tflite_model::Main(argc, argv);
}
