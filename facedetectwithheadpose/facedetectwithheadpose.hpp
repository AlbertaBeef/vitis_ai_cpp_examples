/*
 * Copyright 2022 Avnet Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once
#include <vitis/ai/demo.hpp>
#include <vitis/ai/facedetect.hpp>
#include <vitis/ai/facelandmark.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;


cv::Point2d headpose(const cv::Mat &im, std::vector<cv::Point2d> image_points )
{
    
    // Read input image
    //cv::Mat im = cv::imread("headPose.jpg");
    
    // 2D image points. If you change the image, you need to change vector
    //std::vector<cv::Point2d> image_points;
    //image_points.push_back( cv::Point2d(359, 391) );    // Nose tip
    //image_points.push_back( cv::Point2d(399, 561) );    // Chin
    //image_points.push_back( cv::Point2d(337, 297) );     // Left eye left corner
    //image_points.push_back( cv::Point2d(513, 301) );    // Right eye right corner
    //image_points.push_back( cv::Point2d(345, 465) );    // Left Mouth corner
    //image_points.push_back( cv::Point2d(453, 469) );    // Right mouth corner
    
    // 3D model points.
    std::vector<cv::Point3d> model_points;
    model_points.push_back(cv::Point3d(0.0f, 0.0f, 0.0f));               // Nose tip
    model_points.push_back(cv::Point3d(0.0f, -330.0f, -65.0f));          // Chin
    model_points.push_back(cv::Point3d(-225.0f, 170.0f, -135.0f));       // Left eye left corner
    model_points.push_back(cv::Point3d(225.0f, 170.0f, -135.0f));        // Right eye right corner
    model_points.push_back(cv::Point3d(-150.0f, -150.0f, -125.0f));      // Left Mouth corner
    model_points.push_back(cv::Point3d(150.0f, -150.0f, -125.0f));       // Right mouth corner
    
    // Camera internals
    double focal_length = im.cols; // Approximate focal length.
    Point2d center = cv::Point2d(im.cols/2,im.rows/2);
    cv::Mat camera_matrix = (cv::Mat_<double>(3,3) << focal_length, 0, center.x, 0 , focal_length, center.y, 0, 0, 1);
    cv::Mat dist_coeffs = cv::Mat::zeros(4,1,cv::DataType<double>::type); // Assuming no lens distortion
    
    //cout << "Camera Matrix " << endl << camera_matrix << endl ;

    // Output rotation and translation
    cv::Mat rotation_vector; // Rotation in axis-angle form
    cv::Mat translation_vector;
    
    // Solve for pose
    cv::solvePnP(model_points, image_points, camera_matrix, dist_coeffs, rotation_vector, translation_vector);

    
    // Project a 3D point (0, 0, 1000.0) onto the image plane.
    // We use this to draw a line sticking out of the nose
    
    vector<Point3d> nose_end_point3D;
    vector<Point2d> nose_end_point2D;
    nose_end_point3D.push_back(Point3d(0,0,1000.0));
    
    projectPoints(nose_end_point3D, rotation_vector, translation_vector, camera_matrix, dist_coeffs, nose_end_point2D);
    
    
    for(int i=0; i < image_points.size(); i++)
    {
        //circle(im, image_points[i], 3, Scalar(0,0,255), -1);
        circle(im, image_points[i], 3, Scalar(255,255,255), 2);
    }
    
    cv::line(im,image_points[0], nose_end_point2D[0], cv::Scalar(255,0,0), 2);
    
    //cout << "Rotation Vector " << endl << rotation_vector << endl;
    //cout << "Translation Vector" << endl << translation_vector << endl;
    
    //cout <<  nose_end_point2D << endl;
    
    // Display image.
    //cv::imshow("Output", im);
    //cv::waitKey(0);

    //return 0;
    return nose_end_point2D[0];
}

namespace vitis {
namespace ai {

struct FaceDetectWithHeadPose {
  static std::unique_ptr<FaceDetectWithHeadPose> create();
  FaceDetectWithHeadPose();
  std::vector<vitis::ai::FaceLandmarkResult> run(const cv::Mat &input_image);
  int getInputWidth();
  int getInputHeight();
  size_t get_input_batch();

private:
  std::unique_ptr<vitis::ai::FaceDetect> face_detect_;
  std::unique_ptr<vitis::ai::FaceLandmark> face_landmark_;
};

std::unique_ptr<FaceDetectWithHeadPose> FaceDetectWithHeadPose::create() {
  return std::unique_ptr<FaceDetectWithHeadPose>(new FaceDetectWithHeadPose());
}
int FaceDetectWithHeadPose::getInputWidth() { return face_detect_->getInputWidth(); }
int FaceDetectWithHeadPose::getInputHeight() { return face_detect_->getInputHeight(); }
size_t FaceDetectWithHeadPose::get_input_batch() { return face_detect_->get_input_batch(); }

FaceDetectWithHeadPose::FaceDetectWithHeadPose()
    : face_detect_{vitis::ai::FaceDetect::create("densebox_640_360")},
      face_landmark_{vitis::ai::FaceLandmark::create("face_landmark")} 
      {}


std::vector<vitis::ai::FaceLandmarkResult>
FaceDetectWithHeadPose::run(const cv::Mat &input_image) {
  std::vector<vitis::ai::FaceLandmarkResult> mt_results;
  cv::Mat image;
  image = input_image;

  // run facedetect (densebox)
  auto face_detect_results = face_detect_->run(image);

  for (const auto &r : face_detect_results.rects) {
    cv::rectangle(image, cv::Rect{cv::Point(r.x * image.cols, r.y * image.rows),
                                  cv::Size{(int)(r.width * image.cols),
                                           (int)(r.height * image.rows)}},
                         cv::Scalar(0,255,0),2);
    cv::Rect roi = (cv::Rect{cv::Point(r.x * image.cols, r.y * image.rows),
                                 cv::Size{(int)(r.width * image.cols),
                                          (int)(r.height * image.rows)}});
    if (0 <= roi.x && 0 <= roi.width && roi.x + roi.width <= image.cols && 0 <= roi.y && 0 <= roi.height && roi.y + roi.height <= image.rows) {
      cv::Mat sub_img = image(roi);
      auto face_landmark_results = face_landmark_->run(sub_img);
      //process_result(sub_img, face_landmark_results, false);
      auto points = face_landmark_results.points;
      //for (int i = 0; i < 5; ++i) {
        //LOG_IF(INFO, false) << points[i].first << " " << points[i].second << " ";
        //auto point = cv::Point{static_cast<int>(points[i].first * sub_img.cols),
        //                       static_cast<int>(points[i].second * sub_img.rows)};
        //cv::circle(sub_img, point, 3, cv::Scalar(255, 255, 255), 2);
      //}
      std::vector<cv::Point2d> image_points;
      image_points.push_back( // nose
        cv::Point2d( points[2].first*sub_img.cols, points[2].second*sub_img.rows) );
      image_points.push_back( // chin (place-holder for now)
        cv::Point2d( points[2].first*sub_img.cols, points[2].second*sub_img.rows) );
      image_points.push_back( // left eye
        cv::Point2d( points[0].first*sub_img.cols, points[0].second*sub_img.rows) );
      image_points.push_back( // right eye
        cv::Point2d( points[1].first*sub_img.cols, points[1].second*sub_img.rows) );
      image_points.push_back( // left mouth
        cv::Point2d( points[3].first*sub_img.cols, points[3].second*sub_img.rows) );
      image_points.push_back( // right mouth
        cv::Point2d( points[4].first*sub_img.cols, points[4].second*sub_img.rows) );
      // estimate approximate location of chin
      // let's assume that the chin location will behave similar as the nose location
      int eye_center_x = (image_points[2].x + image_points[3].x)/2;
      int eye_center_y = (image_points[2].y + image_points[3].y)/2;
      int nose_offset_x = (image_points[0].x - eye_center_x);
      int nose_offset_y = (image_points[0].y - eye_center_y);
      int mouth_center_x = (image_points[4].x + image_points[5].x)/2;
      int mouth_center_y = (image_points[4].y + image_points[5].y)/2;
      image_points[1].x = mouth_center_x + nose_offset_x;
      image_points[1].y = mouth_center_y + nose_offset_y;

      // draw points with different colors (for debug)
      //cv::circle(sub_img, image_points[0], 3, cv::Scalar(255, 255, 255), 2);
      //cv::circle(sub_img, image_points[1], 3, cv::Scalar(  0,   0, 255), 2);
      //cv::circle(sub_img, image_points[2], 3, cv::Scalar(  0, 255,   0), 2);
      //cv::circle(sub_img, image_points[3], 3, cv::Scalar(  0, 255,   0), 2);
      //cv::circle(sub_img, image_points[4], 3, cv::Scalar(255,   0,   0), 2);
      //cv::circle(sub_img, image_points[5], 3, cv::Scalar(255,   0,   0), 2);

      // draw head pose vector (draw in original image for full vector length)
      cv::Point2d headpose_vector_start;
      cv::Point2d headpose_vector_end;
      headpose_vector_start.x = roi.x + image_points[0].x;
      headpose_vector_start.y = roi.y + image_points[0].y;
      headpose_vector_end = headpose(sub_img,image_points);
      headpose_vector_end.x += roi.x;
      headpose_vector_end.y += roi.y;

      cv::line(image,headpose_vector_start, headpose_vector_end, cv::Scalar(255,0,0), 2);

      mt_results.emplace_back(face_landmark_results);
    }
  }

  return mt_results;
}
      
} // namespace ai
} // namespace vitis
