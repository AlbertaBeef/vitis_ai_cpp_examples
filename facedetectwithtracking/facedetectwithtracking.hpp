/*
 * Copyright 2020 Avnet Inc.
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

#include <glog/logging.h>
#include <iostream>
#include <memory>
#include <chrono>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/facedetect.hpp>

#include "./centroidtracker.h"

using namespace std;
using namespace cv;


cv::Mat process_result(cv::Mat &image,
                       const CentroidTracker &results,
                       bool is_jpeg) {

  auto objects = results.objects;

  string objectType = "Face";
  cv::Scalar objectColor = cv::Scalar(0,255,0);

  if (!objects.empty()) {
    for (unsigned i = 0; i < objects.size(); i++) {
      auto bbox = results.bboxes[i];
      int x = bbox.second[0];
      int y = bbox.second[1];
      int w = bbox.second[2] - bbox.second[0];
      int h = bbox.second[3] - bbox.second[1];
      cv::Rect roi(x,y,w,h);
      cv::rectangle(image,roi,objectColor,2);
      auto obj = objects[i];
      if ( 1 ) {
        cv::circle(image, cv::Point(obj.second.first, obj.second.second), 4, objectColor, -1);
      }
      string objectID = objectType;
      objectID.append( " " );
      objectID.append( std::to_string(obj.first) );
      cv::putText(image, objectID, cv::Point(x,y-10), cv::FONT_HERSHEY_COMPLEX, 0.5, objectColor, 2);
    }

    //drawing the path
    if ( true ) {
      auto path_keeper = results.path_keeper;
      for (auto obj: objects) {
        int k = 1;
        for (int i = 1; i < path_keeper[obj.first].size(); i++) {
          int thickness = int(sqrt(20 / float(k + 1) * 2.5));
          cv::line(image,
                   cv::Point(path_keeper[obj.first][i - 1].first, path_keeper[obj.first][i - 1].second),
                   cv::Point(path_keeper[obj.first][i].first, path_keeper[obj.first][i].second),
                   cv::Scalar(0, 0, 255), thickness);
          k += 1;
        }
      }
    }

  }

  return image;
}
namespace vitis {
namespace ai {

struct FaceDetectWithTracking {
  static std::unique_ptr<FaceDetectWithTracking> create();
  FaceDetectWithTracking();
  CentroidTracker run(const cv::Mat &input_image);
  int getInputWidth();
  int getInputHeight();
  size_t get_input_batch();

private:
  std::unique_ptr<vitis::ai::FaceDetect> face_detect_;
  std::unique_ptr<CentroidTracker> centroid_tracker_;
};

std::unique_ptr<FaceDetectWithTracking> FaceDetectWithTracking::create() {
  return std::unique_ptr<FaceDetectWithTracking>(new FaceDetectWithTracking());
}
int FaceDetectWithTracking::getInputWidth() { return face_detect_->getInputWidth(); }
int FaceDetectWithTracking::getInputHeight() { return face_detect_->getInputHeight(); }
size_t FaceDetectWithTracking::get_input_batch() { return face_detect_->get_input_batch(); }

FaceDetectWithTracking::FaceDetectWithTracking()
    : face_detect_{vitis::ai::FaceDetect::create("densebox_640_360")},
      centroid_tracker_{new CentroidTracker(20)}
      {}


CentroidTracker
FaceDetectWithTracking::run(const cv::Mat &input_image) {
  cv::Mat image;
  image = input_image;

  // run face detect (densebox)
  auto face_results = face_detect_->run(image);

  vector<vector<int>> boxes;
  for (const auto &r : face_results.rects) {
    int x1 = r.x * image.cols;
    int y1 = r.y * image.rows;
    int x2 = x1 + (r.width * image.cols);
    int y2 = y1 + (r.height * image.rows);
    boxes.insert(boxes.end(), {x1,y1,x2,y2});
  }

  auto objects = centroid_tracker_->update(boxes);

  process_result( image, *centroid_tracker_, false );

  return *centroid_tracker_;
}
      
} // namespace ai
} // namespace vitis


