/*
 * @Descripttion: 
 * @version: 
 * @Author: Gager
 * @Date: 2020-12-08 15:55:35
 * @LastEditors: sueRimn
 * @LastEditTime: 2020-12-14 14:02:22
 */
#include "FaceSpoof.hpp"


using namespace std;
using namespace cv;


FaceSpoof::FaceSpoof(const std::string &mnn_path, int num_thread_){

    //加载模型
    model_interpreter = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(mnn_path.c_str()));
    // 配置调度
    MNN::ScheduleConfig config;
    config.numThread = num_thread_;
    // config.type      = static_cast<MNNForwardType>(MNN_FORWARD_OPENCL);
    config.type = static_cast<MNNForwardType>(MNN_FORWARD_CPU);
    // 配置后端
    MNN::BackendConfig backendConfig;
    // backendConfig.precision = (MNN::BackendConfig::PrecisionMode) 2;
    backendConfig.precision = MNN::BackendConfig::Precision_Low;
    backendConfig.power = MNN::BackendConfig::Power_High;
    config.backendConfig = &backendConfig;
    // 创建会话
    model_session = model_interpreter->createSession(config);
    input_tensor = model_interpreter->getSessionInput(model_session, nullptr);

    vector<int> dims{1, INPUT_SIZE, INPUT_SIZE, 3};
    nchw_Tensor = MNN::Tensor::create<float>(dims, NULL, MNN::Tensor::TENSORFLOW);

}

FaceSpoof::~FaceSpoof(){

    model_interpreter->releaseModel();
    model_interpreter->releaseSession(model_session);
}   


cv::Mat FaceSpoof::Get_Resize_Croped_Img(cv::Mat frame, cv::Point pt1, cv::Point pt2, cv::Point &s_point, cv::Size &croped_wh){

    float cx, cy, halfw;
    cv::Mat resize_img, croped_img;
    cv::Point_<float> center_point;

    try{
        //cout << "[INFO]>>> pt1:" << pt1 << "\t pt2:" << pt2 << endl;
        center_point = (pt1 + pt2) / 2;
        //cout << "[INFO]>>> center_point:" << center_point << endl;
        cx = center_point.x;
        cy = center_point.y;
        //cout << "[INFO]>>> cx:" << cx << "\t cy:" << cy << endl;
        halfw = max((pt2.x - pt1.x)/2, (pt2.y - pt1.y)/2);
        //cout << "[INFO]>>> halfw:" << halfw << endl;
        float min_x = (cx-halfw) > 0 ? cx-halfw:0;
        float min_y = (cy-halfw) > 0 ? cy-halfw:0;
        s_point = cv::Point(min_x, min_y);
        //cout << "[INFO]>>> s_point:" << s_point << endl;
        croped_wh = cv::Size(2*halfw, 2*halfw);
        //cout << "[INFO]>>> croped_wh:" << croped_wh << endl;

        croped_img = frame(cv::Rect(min_x, min_y, 2*halfw, 2*halfw));
        string croped_name = "croped_img.jpg";
        cv::imwrite(croped_name, croped_img);


        if(halfw > 20){
            cv::resize(croped_img, resize_img, cv::Size(INPUT_SIZE, INPUT_SIZE));
            resize_img.convertTo(resize_img, CV_32FC3);
            resize_img = (resize_img - 127.5) / 128.0;
        } 
    }
    catch(exception e){
        // cout << "[INFO]>>> No face was detected!!!" << endl;
        ;
    }

    return resize_img;
}


float FaceSpoof::GetScore(const cv::Mat &frame){


    auto nchw_data   = nchw_Tensor->host<float>();
    auto nchw_size   = nchw_Tensor->size();
    ::memcpy(nchw_data, frame.data, nchw_size);

    auto input_tensor  = model_interpreter->getSessionInput(model_session, nullptr);
    input_tensor->copyFromHostTensor(nchw_Tensor);

    // run network
    model_interpreter->runSession(model_session);
    // get output data
    string output_tensor_name = "Identity:0"; 
    MNN::Tensor *tensor_feature = model_interpreter->getSessionOutput(model_session, output_tensor_name.c_str());
    MNN::Tensor tensor_feature_host(tensor_feature, tensor_feature->getDimensionType());
    tensor_feature->copyToHostTensor(&tensor_feature_host);
    auto out_features = tensor_feature_host.host<float>();

    // cout << "[INFO]>>> out:" << out_features[0] << endl;
    return out_features[0];
    
    // int batch   = tensor_feature->batch();
    // int channel = tensor_feature->channel();
    // int height  = tensor_feature->height();
    // int width   = tensor_feature->width();
    // int type    = tensor_feature->getDimensionType();
    // printf("%d, %d, %d, %d, %d\n", batch, channel, height, width, type);

    // for(int i=0; i< 128; i++){
    //     feature.push_back(out_features[i]);
    //     // printf("%f \t", out_features[i]);
    // }
    
    // feature.insert(feature.end(), out_features.begin(), out_features.end())
}