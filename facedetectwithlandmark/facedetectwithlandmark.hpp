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

using namespace std;
namespace vitis {
namespace ai {

struct FaceDetectWithLandmark {
  static std::unique_ptr<FaceDetectWithLandmark> create();
  FaceDetectWithLandmark();
  std::vector<vitis::ai::FaceLandmarkResult> run(const cv::Mat &input_image);
  int getInputWidth();
  int getInputHeight();
  size_t get_input_batch();

private:
  std::unique_ptr<vitis::ai::FaceDetect> face_detect_;
  std::unique_ptr<vitis::ai::FaceLandmark> face_landmark_;
};

std::unique_ptr<FaceDetectWithLandmark> FaceDetectWithLandmark::create() {
  return std::unique_ptr<FaceDetectWithLandmark>(new FaceDetectWithLandmark());
}
int FaceDetectWithLandmark::getInputWidth() { return face_detect_->getInputWidth(); }
int FaceDetectWithLandmark::getInputHeight() { return face_detect_->getInputHeight(); }
size_t FaceDetectWithLandmark::get_input_batch() { return face_detect_->get_input_batch(); }

FaceDetectWithLandmark::FaceDetectWithLandmark()
    : face_detect_{vitis::ai::FaceDetect::create("densebox_640_360")},
      face_landmark_{vitis::ai::FaceLandmark::create("face_landmark")} 
      {}


std::vector<vitis::ai::FaceLandmarkResult>
FaceDetectWithLandmark::run(const cv::Mat &input_image) {
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
      for (int i = 0; i < 5; ++i) {
        LOG_IF(INFO, false) << points[i].first << " " << points[i].second << " ";
        auto point = cv::Point{static_cast<int>(points[i].first * sub_img.cols),
                               static_cast<int>(points[i].second * sub_img.rows)};
        //cv::circle(sub_img, point, 3, cv::Scalar(255, 8, 18), -1);
        cv::circle(sub_img, point, 3, cv::Scalar(255, 255, 255), 2);
      }

      mt_results.emplace_back(face_landmark_results);
    }
  }

  return mt_results;
}
      
} // namespace ai
} // namespace vitis
