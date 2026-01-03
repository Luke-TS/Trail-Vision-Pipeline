#include "depth_recorder.h"
#include <json/json.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <sstream>

namespace trail_nav {
namespace tools {

class DepthRecorder::Impl {
 public:
  explicit Impl(hardware::DepthCameraInterface* camera)
      : camera_(camera),
        is_recording_(false),
        frame_count_(0),
        start_time_(0) {}
  
  bool StartRecording(const RecordingConfig& config) {
    config_ = config;
    
    // Create output directory structure
    namespace fs = std::filesystem;
    fs::create_directories(config.output_directory);
    fs::create_directories(config.output_directory + "/depth");
    if (config.record_color) {
      fs::create_directories(config.output_directory + "/color");
    }
    
    // Save camera intrinsics
    CameraIntrinsics intrinsics = camera_->GetIntrinsics();
    SaveIntrinsics(intrinsics);
    
    // Initialize metadata file
    metadata_file_.open(config.output_directory + "/metadata.jsonl");
    if (!metadata_file_.is_open()) {
      return false;
    }
    
    is_recording_ = true;
    frame_count_ = 0;
    start_time_ = GetTimestampNs();
    
    return true;
  }
  
  bool CaptureFrame() {
    if (!is_recording_) return false;
    
    // Check limits
    if (config_.max_frames > 0 && frame_count_ >= config_.max_frames) {
      return false;
    }
    
    uint64_t elapsed_ns = GetTimestampNs() - start_time_;
    float elapsed_s = elapsed_ns / 1e9f;
    if (config_.duration_seconds > 0 && elapsed_s >= config_.duration_seconds) {
      return false;
    }
    
    // Capture depth frame
    cv::Mat depth_flipped;
    if (!camera_->GetDepthFrame(depth_flipped)) {
      return false;
    }

    cv::Mat depth_mm;
    cv::flip(depth_flipped, depth_mm, -1);
    
    std::string depth_filename = GetDepthFilename(frame_count_);
    std::vector<int> compression_params;
    if (config_.compress_depth) {
      compression_params = {cv::IMWRITE_PNG_COMPRESSION, 9};
    }
    cv::imwrite(depth_filename, depth_mm, compression_params);
    
    // Capture color frame if enabled
    if (config_.record_color) {
      cv::Mat color_flipped;
      if (camera_->GetColorFrame(color_flipped)) {
        cv::Mat color_frame;
        cv::flip(color_flipped, color_frame, -1);
        std::string color_filename = GetColorFilename(frame_count_);
        cv::imwrite(color_filename, color_frame);
      }
    }
    
    // Write frame metadata
    WriteFrameMetadata(frame_count_, GetTimestampNs());
    
    frame_count_++;
    return true;
  }
  
  void StopRecording() {
    if (!is_recording_) return;
    
    // Write summary metadata
    WriteSummaryMetadata();
    
    metadata_file_.close();
    is_recording_ = false;
  }
  
  void AnnotateFrame(const std::string& annotation) {
    if (is_recording_ && !current_annotations_.empty()) {
      current_annotations_.push_back(annotation);
    }
  }
  
  DepthRecorder::Stats GetStats() const {
    Stats stats;
    stats.frames_captured = frame_count_;
    
    if (frame_count_ > 0) {
      uint64_t elapsed_ns = GetTimestampNs() - start_time_;
      stats.duration_seconds = elapsed_ns / 1e9f;
      stats.avg_fps = frame_count_ / stats.duration_seconds;
    } else {
      stats.duration_seconds = 0;
      stats.avg_fps = 0;
    }
    
    // Estimate storage size
    stats.total_size_mb = EstimateDatasetSize();
    
    return stats;
  }
  
 private:
  hardware::DepthCameraInterface* camera_;
  RecordingConfig config_;
  bool is_recording_;
  int frame_count_;
  uint64_t start_time_;
  std::ofstream metadata_file_;
  std::vector<std::string> current_annotations_;
  
  uint64_t GetTimestampNs() const {
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        now.time_since_epoch()).count();
  }
  
  std::string GetDepthFilename(int frame_num) const {
    std::ostringstream oss;
    oss << config_.output_directory << "/depth/frame_" 
        << std::setw(6) << std::setfill('0') << frame_num << ".png";
    return oss.str();
  }
  
  std::string GetColorFilename(int frame_num) const {
    std::ostringstream oss;
    oss << config_.output_directory << "/color/frame_" 
        << std::setw(6) << std::setfill('0') << frame_num << ".jpg";
    return oss.str();
  }
  
  void SaveIntrinsics(const CameraIntrinsics& intrinsics) {
    Json::Value root;
    root["fx"] = intrinsics.K(0, 0);
    root["fy"] = intrinsics.K(1, 1);
    root["cx"] = intrinsics.K(0, 2);
    root["cy"] = intrinsics.K(1, 2);
    root["width"] = intrinsics.image_size.width;
    root["height"] = intrinsics.image_size.height;
    
    Json::Value distortion(Json::arrayValue);
    for (int i = 0; i < 5; ++i) {
      distortion.append(intrinsics.distortion[i]);
    }
    root["distortion"] = distortion;
    
    std::ofstream file(config_.output_directory + "/intrinsics.json");
    file << root;
  }
  
  void WriteFrameMetadata(int frame_num, uint64_t timestamp_ns) {
    Json::Value frame_data;
    frame_data["frame"] = frame_num;
    frame_data["timestamp_ns"] = (Json::Int64)timestamp_ns;
    
    if (!current_annotations_.empty()) {
      Json::Value annotations(Json::arrayValue);
      for (const auto& ann : current_annotations_) {
        annotations.append(ann);
      }
      frame_data["annotations"] = annotations;
    }
    
    // Write as JSON Lines format (one JSON object per line)
    Json::StreamWriterBuilder builder;
    builder["indentation"] = "";
    metadata_file_ << Json::writeString(builder, frame_data) << "\n";
    
    current_annotations_.clear();
  }
  
  void WriteSummaryMetadata() {
    Json::Value summary;
    summary["total_frames"] = frame_count_;
    summary["duration_seconds"] = (GetTimestampNs() - start_time_) / 1e9;
    summary["recorded_color"] = config_.record_color;
    summary["compressed"] = config_.compress_depth;
    
    std::ofstream file(config_.output_directory + "/summary.json");
    file << summary;
  }
  
  size_t EstimateDatasetSize() const {
    // Rough estimate: ~0.5MB per depth frame compressed
    return (frame_count_ * 0.5) + (config_.record_color ? frame_count_ * 0.1 : 0);
  }
};

DepthRecorder::DepthRecorder(hardware::DepthCameraInterface* camera)
    : impl_(std::make_unique<Impl>(camera)) {}

DepthRecorder::~DepthRecorder() = default;

bool DepthRecorder::StartRecording(const RecordingConfig& config) {
  return impl_->StartRecording(config);
}

bool DepthRecorder::CaptureFrame() {
  return impl_->CaptureFrame();
}

void DepthRecorder::StopRecording() {
  impl_->StopRecording();
}

DepthRecorder::Stats DepthRecorder::GetStats() const {
  return impl_->GetStats();
}

void DepthRecorder::AnnotateFrame(const std::string& annotation) {
  impl_->AnnotateFrame(annotation);
}

}  // namespace tools
}  // namespace trail_nav
