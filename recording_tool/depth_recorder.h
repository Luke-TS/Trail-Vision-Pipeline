// depth_recorder.h
#ifndef TRAIL_NAVIGATION_TOOLS_DEPTH_RECORDER_H_
#define TRAIL_NAVIGATION_TOOLS_DEPTH_RECORDER_H_

#include "trail_sense/core/types.h"
#include "trail_sense/hardware/depth_camera_interface.h"
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <fstream>
#include <memory>

namespace trail_nav {
namespace tools {

// Captures depth and color video with metadata for testing and analysis.
//
// Saves depth as 16-bit PNG sequences (lossless) and color as MP4.
// Records camera intrinsics, timestamps, and frame metadata to JSON.
// Enables repeatable testing with real-world trail conditions.
class DepthRecorder {
 public:
  struct RecordingConfig {
    std::string output_directory;
    bool record_color;              // Also capture RGB video
    bool record_imu;                // Capture IMU data if available
    uint32_t max_frames;            // 0 = unlimited
    float duration_seconds;         // 0 = unlimited
    bool compress_depth;            // Use lossless compression
    
    RecordingConfig()
        : record_color(true),
          record_imu(false),
          max_frames(0),
          duration_seconds(0),
          compress_depth(true) {}
  };
  
  explicit DepthRecorder(hardware::DepthCameraInterface* camera);
  ~DepthRecorder();
  
  // Starts recording to specified directory.
  bool StartRecording(const RecordingConfig& config);
  
  // Captures one frame (call repeatedly in loop).
  bool CaptureFrame();
  
  // Stops recording and saves metadata.
  void StopRecording();
  
  // Gets recording statistics.
  struct Stats {
    int frames_captured;
    float duration_seconds;
    float avg_fps;
    size_t total_size_mb;
  };
  Stats GetStats() const;
  
  // Marks current frame with annotation (e.g., "obstacle_detected").
  void AnnotateFrame(const std::string& annotation);
  
 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

// Plays back recorded depth video for testing.
//
// Simulates live camera feed from recorded data.
// Supports frame-by-frame stepping, speed control, and looping.
// Implements DepthCameraInterface so can drop into existing code.
class DepthPlayback : public hardware::DepthCameraInterface {
 public:
  struct PlaybackConfig {
    std::string dataset_directory;
    bool loop;                      // Restart at end
    float playback_speed;           // 1.0 = realtime, 2.0 = 2x speed
    int start_frame;                // Skip to frame
    
    PlaybackConfig()
        : loop(false),
          playback_speed(1.0f),
          start_frame(0) {}
  };
  
  explicit DepthPlayback(const PlaybackConfig& config);
  ~DepthPlayback() override;
  
  // DepthCameraInterface implementation
  bool Initialize(uint32_t width, uint32_t height, uint32_t fps) override;
  bool GetDepthFrame(cv::Mat& depth_frame) override;
  bool GetColorFrame(cv::Mat& color_frame) override;
  CameraIntrinsics GetIntrinsics() const override;
  void Shutdown() override;
  bool IsReady() const override;
  
  // Playback control
  void Pause();
  void Resume();
  void SeekToFrame(int frame_number);
  void SetPlaybackSpeed(float speed);
  
  // Gets metadata for current frame.
  struct FrameMetadata {
    int frame_number;
    uint64_t timestamp_ns;
    std::vector<std::string> annotations;
    bool has_imu;
    cv::Vec3f imu_accel;
    cv::Vec3f imu_gyro;
  };
  FrameMetadata GetCurrentFrameMetadata() const;
  
  // Dataset information
  int GetTotalFrames() const;
  float GetDatasetDuration() const;
  
 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace tools
}  // namespace trail_nav

#endif  // TRAIL_NAVIGATION_TOOLS_DEPTH_RECORDER_H_
