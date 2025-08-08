#include "ReID.h"

#include "INIReader.h"

ReIDModel::ReIDModel(const ReIDParams &params,
                     std::shared_ptr<luna::AIEngine::IAiAbstraction> &model)
{
    std::cout << "Initializing ReID model" << std::endl;
    _load_params_from_config(params);

    _model = model;
}


FeatureVector ReIDModel::extract_features(cv::Mat &image_patch)
{
    pre_process(image_patch);
    // std::vector<std::vector<float>> output = _trt_inference_engine->forward(image_patch);
    uint n_values =
            image_patch.total() * image_patch.elemSize() * sizeof(float);
    std::vector<uchar> float_img(n_values, 0);
    float *float_data_ptr = (float *) float_img.data();
    for (uint i = 0; i < image_patch.total() * image_patch.elemSize(); i++)
    {
        float_data_ptr[i] =
                static_cast<float>(image_patch.data[i]) /
                static_cast<float>(std::numeric_limits<uint8_t>::max());
    }
    std::map<int, std::vector<uchar>> model_output;
    _model->RunInference(float_img, model_output);
    // TODO: Clean this up
    FeatureVector feature_vector = FeatureVector::Zero(1, FEATURE_DIM);
    for (uint i = 0; i < FEATURE_DIM; i++)
    {
        feature_vector(0, i) = model_output[0][i];
    }

    return feature_vector;
}


void ReIDModel::pre_process(cv::Mat &image)
{
    cv::resize(image, image, _input_size);
    // if (_model_optimization_params.swapRB)
    //     cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
}

void ReIDModel::_load_params_from_config(const ReIDParams &params)
{
    _distance_metric = params.distance_metric;
    _trt_logging_level = params.trt_logging_level;
    _input_size = cv::Size(128, 256);
    // _model_optimization_params.batch_size = static_cast<int>(params.batch_size);
    // _model_optimization_params.fp16 = params.enable_fp16;
    // _model_optimization_params.tf32 = params.enable_tf32;
    // _model_optimization_params.input_layer_name = params.input_layer_name;

    // std::cout << "Trying to get input dims" << std::endl;
    // const auto &input_dims = params.input_layer_dimensions;
    // _input_size = cv::Size(input_dims[3], input_dims[2]);

    // std::cout << "Read input dims" << std::endl;
    // std::cout << "Input dims: " << input_dims[0] << " " << input_dims[1] << " "
    //           << input_dims[2] << " " << input_dims[3] << std::endl;

    // _model_optimization_params.input_dims = nvinfer1::Dims4{
    //         input_dims[0], input_dims[1], input_dims[2], input_dims[3]};
    // _model_optimization_params.swapRB = params.swap_rb;

    // _model_optimization_params.output_layer_names = params.output_layer_names;
}