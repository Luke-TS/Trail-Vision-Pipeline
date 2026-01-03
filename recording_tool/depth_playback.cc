#include "depth_recorder.h"
#include <opencv2/imgcodecs.hpp>
#include <json/json.h>
#include <filesystem>
#include <fstream>
#include <thread>

namespace trail_nav {
namespace tools {

class DepthPlayback::Impl {
 public:
  explicit Impl(const PlaybackConfig& config)
      : config_(config),
        current_frame_(config.start_frame),
        is_paused_(false),
        is_initialized_(false) {}
  
  bool Initialize() {
    namespace fs = std::filesystem;
    
    // Load intrinsics
    std::string intrinsics_path = config_.dataset_directory + "/intrinsics.json";
    if (!LoadIntrinsics(intrinsics_path)) {
      return false;
    }
    
    // Load metadata
    std::string metadata_path = config_.dataset_directory + "/metadata.jsonl";
    if (!LoadMetadata(metadata_path)) {
      return false;
    }
    
    // Count depth frames
    std::string depth_dir = config_.dataset_directory + "/depth";
    total_frames_ = 0;
    for (const auto& entry : fs::directory_iterator(depth_dir)) {
      if (entry.path().extension() == ".png") {
        total_frames_++;
      }
    }
    
    if (total_frames_ == 0) {
      return false;
    }
    
    // Check if color frames exist
    std::string color_dir = config_.dataset_directory + "/color";
    has_color_ = fs::exists(color_dir);
    
    is_initialized_ = true;
    last_frame_time_ = std::chrono::steady_clock::now();
    
    return true;
  }
  
  bool GetDepthFrame(cv::Mat& depth_frame) {
    if (!is_initialized_ || is_paused_) {
      return false;
    }
    
    // Rate limiting based on playback speed
    if (!WaitForNextFrame()) {
      return false;
    }
    
    // Load depth frame
    std::string filename = GetDepthFilename(current_frame_);
    cv::Mat depth_mm = cv::imread(filename, cv::IMWRITE_PNG_COMPRESSION);
    
    if (depth_mm.empty()) {
      return false;
    }
    
    // Convert from uint16 millimeters to float32 meters
    depth_mm.convertTo(depth_frame, CV_32FC1, 0.001);
    
    // Advance to next frame
    current_frame_++;
    if (current_frame_ >= total_frames_) {
      if (config_.loop) {
        current_frame_ = 0;
      } else {
        return false;
      }
    }
    
    return true;
  }
  
  bool GetColorFrame(cv::Mat& color_frame) {
    if (!has_color_ || current_frame_ >= total_frames_) {
      return false;
    }
    
    std::string filename = GetColorFilename(current_frame_);
    color_frame = cv::imread(filename);
    
    return !color_frame.empty();
  }
  
  CameraIntrinsics GetIntrinsics() const {
    return intrinsics_;
  }
  
  void SeekToFrame(int frame_number) {
    current_frame_ = std::clamp(frame_number, 0, total_frames_ - 1);
  }
  
  DepthPlayback::FrameMetadata GetCurrentFrameMetadata() const {
    FrameMetadata metadata;
    
    if (current_frame_ < frame_metadata_.size()) {
      metadata = frame_metadata_[current_frame_];
    }
    
    return metadata;
  }
  
  int GetTotalFrames() const { return total_frames_; }
  
  float GetDatasetDuration() const {
    if (frame_metadata_.empty()) return 0.0f;
    
    uint64_t start_ns = frame_metadata_.front().timestamp_ns;
    uint64_t end_ns = frame_metadata_.back().timestamp_ns;
    
    return (end_ns - start_ns) / 1e9f;
  }
  
  void Pause() { is_paused_ = true; }
  void Resume() { 
    is_paused_ = false;
    last_frame_time_ = std::chrono::steady_clock::now();
  }
  
  void SetPlaybackSpeed(float speed) {
    config_.playback_speed = std::max(0.1f, speed);
  }
  
 private:
  PlaybackConfig config_;
  int current_frame_;
  int total_frames_;
  bool is_paused_;
  bool is_initialized_;
  bool has_color_;
  CameraIntrinsics intrinsics_;
  std::vector<FrameMetadata> frame_metadata_;
  std::chrono::steady_clock::time_point last_frame_time_;
  
  bool LoadIntrinsics(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) return false;
    
    Json::Value root;
    file >> root;
    
    intrinsics_.K(0, 0) = root["fx"].asFloat();
    intrinsics_.K(1, 1) = root["fy"].asFloat();
    intrinsics_.K(0, 2) = root["cx"].asFloat();
    intrinsics_.K(1, 2) = root["cy"].asFloat();
    intrinsics_.K(2, 2) = 1.0f;
    
    intrinsics_.image_size.width = root["width"].asInt();
    intrinsics_.image_size.height = root["height"].asInt();
    
    const Json::Value& distortion = root["distortion"];
    for (int i = 0; i < 5; ++i) {
      intrinsics_.distortion[i] = distortion[i].asFloat();
    }
    
    return true;
  }
  
  bool LoadMetadata(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) return false;
    
    std::string line;
    while (std::getline(file, line)) {
      Json::Value frame_data;
      Json::CharReaderBuilder builder;
      std::istringstream iss(line);
      
      if (Json::parseFromStream(builder, iss, &frame_data, nullptr)) {
        FrameMetadata metadata;
        metadata.frame_number = frame_data["frame"].asInt();
        metadata.timestamp_ns = frame_data["timestamp_ns"].asInt64();
        
        if (frame_data.isMember("annotations")) {
          const Json::Value& annotations = frame_data["annotations"];
          for (const auto& ann : annotations) {
            metadata.annotations.push_back(ann.asString());
          }
        }
        
        frame_metadata_.push_back(metadata);
      }
    }
    
    return !frame_metadata_.empty();
  }
  
  bool WaitForNextFrame() {
    // Calculate frame duration based on dataset FPS and playback speed
    float dataset_fps = 30.0f;  // Assume 30 FPS, could load from summary
    float target_fps = dataset_fps * config_.playback_speed;
    auto frame_duration = std::chrono::microseconds(
        static_cast<int>(1e6 / target_fps));
    
    auto now = std::chrono::steady_clock::now();
    auto elapsed = now - last_frame_time_;
    
    if (elapsed < frame_duration) {
      std::this_thread::sleep_for(frame_duration - elapsed);
    }
    
    last_frame_time_ = std::chrono::steady_clock::now();
    return true;
  }
  
  std::string GetDepthFilename(int frame_num) const {
    std::ostringstream oss;
    oss << config_.dataset_directory << "/depth/frame_" 
        << std::setw(6) << std::setfill('0') << frame_num << ".png";
    return oss.str();
  }
  
  std::string GetColorFilename(int frame_num) const {
    std::ostringstream oss;
    oss << config_.dataset_directory << "/color/frame_" 
        << std::setw(6) << std::setfill('0') << frame_num << ".jpg";
    return oss.str();
  }
};

DepthPlayback::DepthPlayback(const PlaybackConfig& config)
    : impl_(std::make_unique<Impl>(config)) {}

DepthPlayback::~DepthPlayback() = default;

bool DepthPlayback::Initialize(uint32_t width, uint32_t height, uint32_t fps) {
  return impl_->Initialize();
}

bool DepthPlayback::GetDepthFrame(cv::Mat& depth_frame) {
  return impl_->GetDepthFrame(depth_frame);
}

bool DepthPlayback::GetColorFrame(cv::Mat& color_frame) {
  return impl_->GetColorFrame(color_frame);
}

CameraIntrinsics DepthPlayback::GetIntrinsics() const {
  return impl_->GetIntrinsics();
}

void DepthPlayback::Shutdown() {}

bool DepthPlayback::IsReady() const {
  return true;
}

void DepthPlayback::Pause() {
  impl_->Pause();
}

void DepthPlayback::Resume() {
  impl_->Resume();
}

void DepthPlayback::SeekToFrame(int frame_number) {
  impl_->SeekToFrame(frame_number);
}

void DepthPlayback::SetPlaybackSpeed(float speed) {
  impl_->SetPlaybackSpeed(speed);
}

DepthPlayback::FrameMetadata DepthPlayback::GetCurrentFrameMetadata() const {
  return impl_->GetCurrentFrameMetadata();
}

int DepthPlayback::GetTotalFrames() const {
  return impl_->GetTotalFrames();
}

float DepthPlayback::GetDatasetDuration() const {
  return impl_->GetDatasetDuration();
}

}  // namespace tools
}  // namespace trail_nav
