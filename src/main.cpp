#include <algorithm> 
#include <cctype>
#include <locale>
#include <iostream>
#include <cstring>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>
#include "cudnn_conv.h"
#include "profile.h"
using namespace std;

void* input_data_g = nullptr;
void* output_data_g = nullptr;
void* weight_data_g = nullptr;
void* cudnn_workspace_g = nullptr;
void* input_data_host_g = nullptr;
void* output_data_host_g = nullptr;
void* weight_data_host_g = nullptr;

extern cudnnMathType_t math_type;
extern bool int8_ext;

// trim from start (in place)
static inline void ltrim(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int ch) {
        return !std::isspace(ch);
    }));
}

// trim from end (in place)
static inline void rtrim(std::string &s) {
    s.erase(std::find_if(s.rbegin(), s.rend(), [](int ch) {
        return !std::isspace(ch);
    }).base(), s.end());
}

// trim from both ends (in place)
static inline void trim(std::string &s) {
    ltrim(s);
    rtrim(s);
}

// trim from start (copying)
static inline std::string ltrim_copy(std::string s) {
    ltrim(s);
    return s;
}

// trim from end (copying)
static inline std::string rtrim_copy(std::string s) {
    rtrim(s);
    return s;
}

// trim from both ends (copying)
static inline std::string trim_copy(std::string s) {
    trim(s);
    return s;
}

static cudnnDataType_t GetType(int type, bool use_tc) {
    if (type == 0) {
        return CUDNN_DATA_FLOAT;
    } else if (type == 1){
        return CUDNN_DATA_HALF;
    } else if (type == 2) {
        if (int8_ext) {
            if (use_tc) {
                return CUDNN_DATA_INT8x32;
            } else {
                return CUDNN_DATA_INT8x4;
            }
        } else {
            return CUDNN_DATA_INT8;
        }
    } else {
        CHECK_EXIT(true, "type error");
    }
}
static cudnnTensorFormat_t GetFormat(int format) {
    if (format == 0) {
        return CUDNN_TENSOR_NCHW;
    } else if (format == 1) {
        if (int8_ext) {
            return CUDNN_TENSOR_NCHW_VECT_C;
        } else {
            return CUDNN_TENSOR_NHWC;
        }
    } else {
        CHECK_EXIT(true, "format error");
    }
}
class ConvConfig {
public:
    ConvConfig (string line) {
        int i_type, w_type, o_type, i_format, w_format, o_format, v, use_tc;
        sscanf(line.data(),
               "%d %d %d %d\
               %d %d %d\
               %d %d\
               %d %d\
               %d %d\
               %d\
               %d %d %d\
               %d %d %d\
               %d\
               %d",
               &input_n, &input_c, &input_h, &input_w,
               &k_n, &k_h, &k_w,
               &p_h, &p_w,
               &s_h, &s_w,
               &d_h, &d_w,
               &group,
               &i_type, &w_type, &o_type,
               &i_format, &w_format, &o_format,
               &v,
               &use_tc);

         use_tensor_core = (use_tc == 1) ? true : false;

         k_c = input_c;
         input_type = GetType(i_type, use_tensor_core);
         weight_type = GetType(w_type, use_tensor_core);
         output_type = GetType(o_type, use_tensor_core);

         input_format = GetFormat(i_format);
         weight_format = GetFormat(w_format);
         output_format = GetFormat(o_format);

         val = (v == 1) ? true : false;
    }

    friend ostream& operator <<(ostream& os, ConvConfig& config) {
        os << "Config: " << 
            config.input_n << ", " << config.input_c << ", " << config.input_h << ", " << config.input_w << ", " << 
            config.k_n << ", " << config.k_c << ", " << config.k_h << ", " << config.k_w << ", " << 
            config.p_h << ", " << config.p_w << ", " << 
            config.s_h << ", " << config.s_w << ", " <<
            config.d_h << ", " << config.d_w << ", " << 
            config.group << ", " << 
            config.input_type << ", " << config.weight_type << ", " << config.output_type << ", " << 
            config.input_format << ", " << config.weight_format << ", " << config.output_format << ", " <<
            config.val << ", " << config.use_tensor_core << endl;
        return os;
    }

	int input_n;
    int input_c;
    int input_h;
    int input_w;
    int k_n;
    int k_c;
    int k_h;
    int k_w;
    int p_h;
    int p_w;
    int s_h;
    int s_w;
    int d_h;
    int d_w;
    int group;
    bool use_tensor_core;
    cudnnDataType_t input_type;
    cudnnDataType_t weight_type;
    cudnnDataType_t output_type;
    cudnnTensorFormat_t input_format;
    cudnnTensorFormat_t weight_format;
    cudnnTensorFormat_t output_format;
    bool val;
};

vector<ConvConfig> ReadConvConfig(string config_path) {
    vector<ConvConfig> configs;
    string line;
    ifstream infile; 
    infile.open(config_path.data());   
    CHECK_EXIT(!infile.is_open(), "file not open")

    while (getline(infile, line)) {
	    trim(line);
        if (line.data()[0] == '#') continue;
        if (line.size() == 0) continue;
        cout << line << endl;
        ConvConfig config(line);
        configs.push_back(config);
    }
    return configs;
}

#define THRESHOLD (0.001)

/*
void BasicConv(float* output, float* input, float* weight,
               int input_n, int input_c, int input_h, int input_w,
               int k_n, int k_c, int k_h, int k_w, 
               int p_h, int p_w,
               int s_h, int s_w,
               int d_h, int d_w,
               int group) {
     
    CudnnConv conv(input_n, input_c, input_h, input_w,
                   k_n, k_c, k_h, k_w,
                   p_h, p_w, s_h, s_w, d_h, d_w,
                   group);
    //int out_n = conv.output_n();
    int out_c = conv.output_c();
    int out_h = conv.output_h();
    int out_w = conv.output_w();

    for (int c = 0; c < out_c; c++) {
        for (int h = 0; h < out_h; h++) {
            for (int w = 0; w < out_w; w++) {
                float sum = 0;
                for (int kc = 0; kc < k_c; kc++ ) {
                    for (int kh = 0; kh < k_h; kh++) {
                        int ih = h + kh - p_h;
                        for (int kw = 0; kw < k_w; kw++) {
                            int iw = w + kw - p_w;
                            int src_index = kc * input_h * input_w + ih * input_w + iw;
                            int weight_index = c * k_c * k_h * k_w + kc * k_h * k_w + kh * k_w + kw;
                            float src_value;
                            if ((ih >= input_h) || (ih < 0) || (iw >= input_w) || (iw < 0)) {
                                src_value = 0;
                            } else {
                                src_value = input[src_index];
                            }
                            float weight_value = weight[weight_index];
                            sum += src_value * weight_value;
                        }
                    }
                }
                int out_index = c * out_h * out_w + h * out_w + w;
                output[out_index] = sum;
            }
        }
    }
}
*/


void TestCudnnConv(int input_n, int input_c, int input_d, int input_h, int input_w, 
                   int k_n, int k_c, int k_d, int k_h, int k_w, 
                   int p_d, int p_h, int p_w, 
                   int s_d, int s_h, int s_w,
                   int d_d, int d_h, int d_w,
                   int group,
                   cudnnDataType_t in_type = CUDNN_DATA_FLOAT,
                   cudnnDataType_t weight_type = CUDNN_DATA_FLOAT,
                   cudnnDataType_t out_type = CUDNN_DATA_FLOAT,
                   cudnnTensorFormat_t in_format = CUDNN_TENSOR_NCHW,
                   cudnnTensorFormat_t weight_format = CUDNN_TENSOR_NCHW,
                   cudnnTensorFormat_t output_format = CUDNN_TENSOR_NCHW,
                   bool validate = true) {
    int input_size = input_n * input_c * input_h * input_w;
    CHECK_EXIT(input_size > kMaxDataSize, "input_size > kMaxDataSize");
    int weight_size = k_n * k_c * k_h * k_w;
    CHECK_EXIT(weight_size > kMaxWeightSize, "weight_size > kMaxWeightSize");

    CudnnConv conv(input_n, input_c, input_d, input_h, input_w,
                   k_n, k_c, k_d, k_h, k_w,
                   p_d, p_h, p_w, 
                   s_d, s_h, s_w, 
                   d_d, d_h, d_w,
                   group,
                   in_type, weight_type, out_type,
                   in_format, weight_format, output_format);

    int output_size = conv.output_size();
    CHECK_EXIT(output_size > kMaxDataSize, "output_size > kMaxDataSize");

    conv.InitAlgo(cudnn_handle_g);

    OPT_PROFILE_TIME_RESET(0);
#ifdef NVPROFILE
    int profile_count = 1;
#else
    int profile_count = 1;
#endif
    OPT_PROFILE_TIME_START(0);
    for (int i = 0; i < profile_count; i++) {
        conv.Run(input_data_g, weight_data_g, output_data_g, cudnn_workspace_g, cudnn_handle_g);
    }
    OPT_PROFILE_TIME_STOP(0, "Run", profile_count, 1);

    float* output_host = new float[output_size];
    cudaMemcpy(output_host, output_data_g, output_size * sizeof(float), cudaMemcpyDeviceToHost);
    int print_count = output_size <= 100 ? output_size : 100;
    for (int i = 0; i < print_count; i++) {
        printf("%f, ", output_host[i]);
    }
    printf("\n");

    /*
    if (validate) {
        float diff = 0.0;
        int diff_count = 0;
        float* output_host = new float[output_size];
        float* output_host_ref = new float[output_size];
        cudaMemcpy(output_host, output_data_g, output_size * sizeof(float), cudaMemcpyDeviceToHost);
        int print_count = output_size <= 100 ? output_size : 100;
        for (int i = 0; i < print_count; i++) {
            printf("%f, ", output_host[i]);
        }
        printf("\n");

#if 1
        BasicConv((float*)output_host_ref, (float*)input_data_host_g, (float*)weight_data_host_g, 
                 input_n, input_c, input_h, input_w,
                   k_n, k_c, k_h, k_w,
                   p_h, p_w, s_h, s_w, d_h, d_w,
                   group);
        for (int i = 0; i < print_count; i++) {
            printf("%f, ", output_host_ref[i]);
        }
        printf("\n");
#endif

        for (int i = 0; i < output_size; i++) {
            diff = output_host_ref[i] - output_host[i];
            if (abs(diff) > THRESHOLD) {
                diff_count++;
                //printf("diff: %f, ", diff);
            }
        }
        printf("\n");
        printf("diff_count / total, %d / %d\n", diff_count, output_size);
        delete[] output_host;
        delete[] output_host_ref;
    }
    */
}

void Allocdata(cudnnDataType_t type_input = CUDNN_DATA_FLOAT,
               cudnnDataType_t type_weight = CUDNN_DATA_FLOAT,
               cudnnDataType_t type_output = CUDNN_DATA_FLOAT) {
    cout << "Initing Data ..." << endl;
    CHECK_EXIT(type_input != CUDNN_DATA_FLOAT, "only support float");
    CHECK_EXIT(type_weight != CUDNN_DATA_FLOAT, "only support float");
    CHECK_EXIT(type_output != CUDNN_DATA_FLOAT, "only support float");

    if (!input_data_g) cudaMalloc(&input_data_g, kMaxDataSize * sizeof(float));
    if (!weight_data_g) cudaMalloc(&weight_data_g, kMaxWeightSize * sizeof(float));
    if (!output_data_g) cudaMalloc(&output_data_g, kMaxDataSize * sizeof(float));
    if (!cudnn_workspace_g) cudaMalloc(&cudnn_workspace_g, kMaxCudnnWorkspace);
    if (!input_data_host_g) input_data_host_g = malloc(kMaxDataSize * sizeof(float));
    if (!weight_data_host_g) weight_data_host_g = malloc(kMaxWeightSize * sizeof(float));
    if (!output_data_host_g) output_data_host_g = malloc(kMaxDataSize * sizeof(float));

    for (int i = 0; i < kMaxDataSize; i++) {
        //((float*)input_data_host_g)[i] = rand() % 10;
        ((float*)input_data_host_g)[i] = 0.5;
    }
    for (int i = 0; i < kMaxWeightSize; i++) {
        //((float*)weight_data_host_g)[i] = rand() % 10 / 10.;
        ((float*)weight_data_host_g)[i] = 1;
    }

    cudaMemcpy(input_data_g, input_data_host_g, kMaxDataSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(weight_data_g, weight_data_host_g, kMaxWeightSize * sizeof(float), cudaMemcpyHostToDevice);
    cudnnStatus_t sts = cudnnCreate(&cudnn_handle_g);
    CHECK_EXIT(sts != CUDNN_STATUS_SUCCESS, "cudnnCreate");
}

void ReleaseData() {
    cout << "Release Data ..........." << endl;
    if (input_data_g) cudaFree(input_data_g);
    if (weight_data_g) cudaFree(weight_data_g);
    if (output_data_g) cudaFree(output_data_g);
    if (cudnn_workspace_g) cudaFree(cudnn_workspace_g);
    if (input_data_host_g) free(input_data_host_g);
    if (weight_data_host_g) free(weight_data_host_g);
    if (output_data_host_g) free(output_data_host_g);
    cudnnDestroy(cudnn_handle_g);
}

int main() {
    cout << "Allocating Data ..........." << endl;
    Allocdata();

    TestCudnnConv(1, 3, 8, 350, 640,
                  64, 3, 1, 7, 7,
                  0, 3, 3,
                  1, 2, 2,
                  1, 1, 1,
                  1,
                  CUDNN_DATA_FLOAT,
                  CUDNN_DATA_FLOAT,
                  CUDNN_DATA_FLOAT,
                  CUDNN_TENSOR_NCHW,
                  CUDNN_TENSOR_NCHW,
                  CUDNN_TENSOR_NCHW,
                  false);

    ReleaseData();
    return 0;
}
