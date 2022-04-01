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
#include <vitis/ai/platedetect.hpp>
#include <vitis/ai/platenum.hpp>
#include <vitis/ai/ssd.hpp>
#include <iostream>

using namespace std;
namespace vitis {
namespace ai {

struct PlateRecognition {
  static std::unique_ptr<PlateRecognition> create();
  PlateRecognition();
  std::vector<vitis::ai::PlateNumResult> run(const cv::Mat &input_image);
  std::vector<std::vector<vitis::ai::PlateNumResult>> run(const std::vector<cv::Mat>& input_images);
  int getInputWidth();
  int getInputHeight();
  size_t get_input_batch();

private:
  std::unique_ptr<vitis::ai::SSD> ssd_;
  std::unique_ptr<vitis::ai::PlateDetect> plate_detect_;
  std::unique_ptr<vitis::ai::PlateNum> plate_num_;

  bool debug;
};

std::unique_ptr<PlateRecognition> PlateRecognition::create() {
  return std::unique_ptr<PlateRecognition>(new PlateRecognition());
}
int PlateRecognition::getInputWidth() { return ssd_->getInputWidth(); }
int PlateRecognition::getInputHeight() { return ssd_->getInputHeight(); }
size_t PlateRecognition::get_input_batch() { return ssd_->get_input_batch(); }

PlateRecognition::PlateRecognition()
    : ssd_{vitis::ai::SSD::create("ssd_traffic_pruned_0_9")},
      plate_detect_{vitis::ai::PlateDetect::create("plate_detect")},
      plate_num_{vitis::ai::PlateNum::create("plate_num")} 
{
  const char * val = std::getenv( "PLATERECOGNITION_DEBUG" );
  if ( val == nullptr ) {
    debug = false;
  }
  else {
    cout << "[INFO] PLATERECOGNITION_DEBUG" << endl;
    debug = true;
  }
}


std::vector<vitis::ai::PlateNumResult>
PlateRecognition::run(const cv::Mat &input_image) {
  std::vector<vitis::ai::PlateNumResult> mt_results;
  cv::Mat image;
  image = input_image;

  static int frame_number = 0;
  frame_number++;
  if ( debug == true ) {
    cout << "Frame " << frame_number << endl;
  }

  // run vehicle detection (SSD)
  auto ssd_results = ssd_->run(image);

  for ( const auto bbox : ssd_results.bboxes ) {

    string label;
    cv::Scalar color;
    switch (bbox.label) {
      case 1:  
        label = "car";    
	color = cv::Scalar(0,255,0);
	break;
      case 2:  
	label = "cycle";      
	color = cv::Scalar(255,0,0);
	break;
      case 3:  
	label = "person"; 
	color = cv::Scalar(255,255,0);
	break;
      default: 
	label = "unknown";    
	color = cv::Scalar(0,0,255);
	break;	      
    }	     

    auto bbox_roi = cv::Rect{
      (int)(bbox.x * image.cols),
      (int)(bbox.y * image.rows),
      (int)(bbox.width  * image.cols),
      (int)(bbox.height * image.rows)
      };

    // only consider SSD results with confidence of 80% or more
    if ( bbox.score < 0.80 ) continue; 

    cv::rectangle(image, bbox_roi, color);
    cv::putText(image,label,cv::Point(bbox_roi.x,bbox_roi.y),cv::FONT_HERSHEY_SIMPLEX,0.5,color,2);
    if ( debug == true ) {
      cout << "  SSD : label=" << bbox.label << " x,y,w,h=" << bbox_roi.x << "," << bbox_roi.y << "," << bbox_roi.width << "," << bbox_roi.height << " confidence=" << bbox.score << endl;
    }

    // only look for license plates on vehicles
    if ( bbox.label != 1 ) continue; 

    cv::Mat bbox_img = image(bbox_roi);

    // run plate_detect
    auto plate_detect_result = plate_detect_->run(bbox_img);
    auto plate_bbox = plate_detect_result.box;
    auto roi = cv::Rect{
      (int)(plate_bbox.x * bbox_img.cols),
      (int)(plate_bbox.y * bbox_img.rows),
      (int)(plate_bbox.width  * bbox_img.cols),
      (int)(plate_bbox.height * bbox_img.rows)
      };

    // only consider platedetect results with confidence of 90% or more
    if ( plate_bbox.score < 0.90 ) continue; 

    cv::rectangle(bbox_img, roi, cv::Scalar(0, 0, 255));

    if ( debug == true ) {
      cout << "    PlateDetect : x,y,w,h=" << roi.x << "," << roi.y << "," << roi.width << "," << roi.height << " confidence=" << plate_bbox.score << endl;
    }

    if (0 <= roi.x && 0 <= roi.width && roi.x + roi.width <= bbox_img.cols && 0 <= roi.y && 0 <= roi.height && roi.y + roi.height <= bbox_img.rows) {

      cv::Mat sub_img = bbox_img(roi);

      // run plate_num
      auto plate_num_result = plate_num_->run(sub_img);
      //process_result(sub_img, plate_num_result, false);

      cv::putText(bbox_img,plate_num_result.plate_number,cv::Point(roi.x,roi.y),cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(0,0,255),2);
      if ( debug == true ) {
        cout << "      PlateNum : size=" << plate_num_result.width << "," << plate_num_result.height << " color=" << plate_num_result.plate_color << " number=[" << plate_num_result.plate_number << "]" << endl;
      }

      mt_results.emplace_back(plate_num_result);
    }
  }

  return mt_results;
}

std::vector<std::vector<vitis::ai::PlateNumResult>> PlateRecognition::run(
    const std::vector<cv::Mat>& input_images) {
  std::vector<std::vector<vitis::ai::PlateNumResult>> rets;
  for (auto& image : input_images) {
    rets.push_back(run(image));
  }
  return rets;
}
      
} // namespace ai
} // namespace vitis
