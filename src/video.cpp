/*
 * @Descripttion: 
 * @version: 
 * @Author: Gager
 * @Date: 2020-12-14 11:12:15
 * @LastEditors: sueRimn
 * @LastEditTime: 2020-12-14 16:05:21
 */
#include <opencv2/tracking.hpp>
#include <opencv2/opencv.hpp>
#include "FaceAligner.hpp"
#include "FaceDetect.h"
#include "FaceSpoof.hpp"
#include "UltraFace.hpp"
#include <iostream>
#include <thread>

using namespace cv;
using namespace std;

string Convert(float Num)
{
	ostringstream oss;
	oss<<Num;
	string str(oss.str());
	return str;
}

int main(int argc, char **argv){


    //人脸检测。
    // std::string model_path = "./models/";
    // TIEVD::FaceDetect face_detect(model_path, 1, 0.7f, 0.8f, 0.9f);
    UltraFace ultraface("./models/slim-320.mnn", 320, 240, 4, 0.65); // config model input

    //人脸特征提取
    string spoof_model_path = "./models/face_spoof.mnn";
    FaceSpoof facenet = FaceSpoof(spoof_model_path, 4);


    VideoCapture capture(-1);
    if(!capture.isOpened()){
        cout << "[INFO]>>> 摄像头开启失败" << endl;
    }
    cv::Mat frame, resize_img;
    while(true){

        capture >> frame;
        // std::vector<TIEVD::FaceInfo> face_info1 = face_detect.Detect_MaxFace(frame, 32, 3);
        vector<FaceInfo> face_info1;
        ultraface.detect(frame, face_info1);
        if(face_info1.size() > 0){

            // cv::Point p1 = cv::Point(face_info1[0].bbox.xmin, face_info1[0].bbox.ymin);
            // cv::Point p2 = cv::Point(face_info1[0].bbox.xmax, face_info1[0].bbox.ymax);
            cv::Point p1 = cv::Point(face_info1[0].x1, face_info1[0].y1);
            cv::Point p2 = cv::Point(face_info1[0].x2, face_info1[0].y2);
            cv::Point face_center = (p1 + p2) / 2; 
            cv::rectangle(frame, p1, p2, cv::Scalar(0, 255, 0), 1);
            // cout << face_center.x << "," << face_center.y << endl;
            int w = p2.x - p1.x;
            int h = p2.y - p1.y;
            cout << w << "," << h << endl;
            if(w < 120 || h < 120){
                ;
            }
            else{
                int min_x = (face_center.x - 112) > 0 ? (face_center.x - 112):0;
                int min_y = (face_center.y - 112) > 0 ? (face_center.y - 112):0;
                int max_x = (face_center.x + 112) < 800 ? (face_center.x + 112):800;
                int max_y = (face_center.y + 112) < 600 ? (face_center.y + 112):600;
                p1 = cv::Point(min_x, min_y);
                p2 = cv::Point(max_x, max_y);
                cv::rectangle(frame, p1, p2, cv::Scalar(255, 0, 0), 1);
                cv::Mat crop_img = frame(cv::Rect(p1, p2));
                // cout << crop_img.size() << endl;

                cv::resize(crop_img, resize_img, cv::Size(224, 224));
                resize_img.convertTo(resize_img, CV_32FC3);
                resize_img = (resize_img - 127.5) / 128.0;
                float score = facenet.GetScore(resize_img);
                cv::putText(frame, Convert(score), cv::Point(16, 32),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255));
                
            }


            // box0 = cv::Rect2d(p1.x, p1.y, p2.x-p1.x, p2.y-p1.y);
        }

        cv::namedWindow("视频", 0);
        cv::resizeWindow("视频", 800, 600);
        cv::imshow("视频", frame);
        if(cv::waitKey(1) >= 0) {
            break;
        }
    }
}