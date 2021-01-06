/*
 * @Descripttion: 
 * @version: 
 * @Author: Gager
 * @Date: 2020-12-08 15:55:24
 * @LastEditors: sueRimn
 * @LastEditTime: 2020-12-14 11:12:39
 */
#include <opencv2/tracking.hpp>
#include <opencv2/opencv.hpp>
#include "FaceAligner.hpp"
#include "FaceDetect.h"
#include "FaceSpoof.hpp"
#include <iostream>
#include <thread>

using namespace cv;
using namespace std;



int main(){

    //人脸检测。
    std::string model_path = "./models/";
    TIEVD::FaceDetect face_detect(model_path, 1, 0.7f, 0.8f, 0.9f);

    //人脸特征提取
    string madel_path = "./models/face_spoof.mnn";
    FaceSpoof facenet = FaceSpoof(madel_path, 4);

    cv::Mat resize_img;
    cv::Mat img = cv::imread("./imgs/iu_spoof.jpg");
    cout << img.size() << endl;

    cv::resize(img, resize_img, cv::Size(224, 224));
    resize_img.convertTo(resize_img, CV_32FC3);
    resize_img = (resize_img - 127.5) / 128.0;
    auto start = chrono::steady_clock::now();
    float score = facenet.GetScore(resize_img);
    auto end = chrono::steady_clock::now();
    chrono::duration<double> elapsed = end - start;
    cout << "[INFO]>>> 特征提取耗时:" << elapsed.count() << endl;
    
}