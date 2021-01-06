/*
 * @Descripttion: 
 * @version: 
 * @Author: Gager
 * @Date: 2020-12-08 15:55:42
 * @LastEditors: sueRimn
 * @LastEditTime: 2020-12-14 11:06:53
 */
#ifndef FaceSpoof_hpp
#define FaceSpoof_hpp

#pragma once

#include <opencv2/opencv.hpp>
#include "ImageProcess.hpp"
#include "Interpreter.hpp"
#include "MNNDefine.h"
#include "Tensor.hpp"
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <chrono>


using namespace std;



class FaceSpoof {
public:
    FaceSpoof(const std::string &mnn_path, int num_thread_ = 4);

    ~FaceSpoof();
    float GetScore(const cv::Mat &frame);
    cv::Mat Get_Resize_Croped_Img(cv::Mat frame, cv::Point pt1, cv::Point pt2, cv::Point &s_point, cv::Size &croped_wh);

private:

    std::shared_ptr<MNN::Interpreter> model_interpreter;
    MNN::Session *model_session = nullptr;
    MNN::Tensor *input_tensor = nullptr;
    MNN::Tensor *nchw_Tensor = nullptr;
    const int INPUT_SIZE = 224;

};

#endif /* mobileface_hpp */
