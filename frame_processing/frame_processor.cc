#include "trail_sense/vision/frame_processor.h"
#include <depthai/pipeline/node/PointCloud.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <deque>
#include <chrono>

namespace trail_nav::vision {

class FrameProcessor::Impl {
public:
    Impl()
        : vert_flip_enabled_(true), 
        spatial_filter_enabled_(false),
        temporal_filter_enabled_(false),
        temporal_alpha_(0.5f),
        frame_number_(0) {}

    ProcessedFrame Process(const cv::Mat& raw_depth) {
        ProcessedFrame result;

        // Validate input
        if (raw_depth.empty()) {
            return result;
        }

        // Flip if specified
        cv::Mat raw_depth_corrected = raw_depth.clone();
        if (vert_flip_enabled_) {
            cv::flip(raw_depth, raw_depth_corrected, -1);
        }

        // Convert to CV_32FC1 if needed
        cv::Mat depth_float;
        if (raw_depth_corrected.type() == CV_16UC1) {
            // Convert from millimeters (uint16) to meters (float32)
            raw_depth_corrected.convertTo(depth_float, CV_32FC1, 0.001);
        } else if (raw_depth_corrected.type() == CV_32FC1) {
            depth_float = raw_depth_corrected.clone();
        } else {
            // Unsupported format
            return result;
        }

        // Apply spatial filtering
        if (spatial_filter_enabled_) {
            ApplySpatialFilter(depth_float, 5);
        }

        // Apply temporal filtering
        if (temporal_filter_enabled_) {
            ApplyTemporalFilter(depth_float);
        }

        // Fill processed frame
        result.depth = depth_float;
        result.timestamp_ns = GetTimestampNs();
        result.frame_number = frame_number_++;

        // Note: color, points, can voxels can be filled by caller
        // points conversion requires intrinsics which caller has

        return result;
    }

    void ApplySpatialFilter(cv::Mat& frame, int kernel_size) {
        if (!spatial_filter_enabled_ || kernel_size < 3) {
            return;
        }

        // Ensure odd kernel size
        if (kernel_size % 2 == 0) {
            kernel_size++;
        }

        // Use median filter to reduce noise while preserving edges
        // Important for outdoor scenes with branches and irregular objects
        cv::Mat filtered;
        cv::medianBlur(frame, filtered, kernel_size);

        // Optional: Apply bilateral filter for edge-preserving smoothing
        // Uncomment if median alone isn't sufficient
        // cv::bilateralFilter(filtered, frame, kernel_size, 50.0, 50.0);

        frame = filtered;
    }

    void ApplyTemporalFilter(cv::Mat& frame) {
        if (!temporal_filter_enabled_) {
            return;
        }

        // Initialize temporal buffer on first frame
        if (temporal_buffer_.empty()) {
            temporal_buffer_.push_back(frame.clone());
            return;
        }

        // Get most recent frame from buffer
        cv::Mat& prev_frame = temporal_buffer_.back();

        // Ensure same size
        if (prev_frame.size() != frame.size()) {
            temporal_buffer_.clear();
            temporal_buffer_.push_back(frame.clone());
            return;
        }

        // Exponential moving average: 
        // filtered = alpha * current + (1 - alpha) * previous
        cv::Mat filtered;
        cv::addWeighted(frame, 1.0f - temporal_alpha_, 
                        prev_frame, temporal_alpha_, 
                        0.0, filtered);

        // Handle invalid depth values (0 or NaN)
        // Use current frame value if previous was invalid
        cv::Mat mask_invalid_prev = (prev_frame == 0.0f) | (prev_frame != prev_frame);
        frame.copyTo(filtered, mask_invalid_prev);

        frame = filtered;

        // Update buffer
        temporal_buffer_.push_back(frame.clone());

        // Keep buffer size limited (last 5 frames)
        if (temporal_buffer_.size() > 5) {
            temporal_buffer_.pop_front();
        }
    }

    PointCloud ToPointCloud(const cv::Mat& depth,
                            const CameraIntrinsics& intrinsics,
                            const int row_inc = 1,
                            const int col_inc = 1) const {
        PointCloud cloud;

        if (depth.empty() || depth.type() != CV_32FC1) {
            return cloud;
        }

        // Extract intrinsic parameters
        float fx = intrinsics.K(0, 0);
        float fy = intrinsics.K(1, 1);
        float cx = intrinsics.K(0, 2);
        float cy = intrinsics.K(1, 2);

        // Pre-allocate approximate size (assume ~50% valid pixels)
        cloud.reserve(depth.rows * depth.cols / (2 * row_inc * col_inc));

        // Convert each pixel to 3D point
        for (int v = 0; v < depth.rows; v+=row_inc) {
            const float* depth_row = depth.ptr<float>(v);

            for (int u = 0; u < depth.cols; u+=col_inc) {
                float z = depth_row[u];

                // Skip invalid depth values
                if (z <= 0.0f || z != z || z > 10.0f) {  // z > 10m likely invalid
                    continue;
                }

                // Back-project to 3D using pinhole camera model
                // X = (u - cx) * Z / fx
                // Y = (v - cy) * Z / fy
                // Z = Z
                float x = (u - cx) * z / fx;
                float y = (v - cy) * z / fy;

                cloud.push_back(cv::Point3f(x, y, z));
            }
        }

        // Shrink to actual size
        cloud.shrink_to_fit();

        return cloud;
    }

    VoxelMap ToVoxelMap(const PointCloud& pc, float voxel_size_m) {
        VoxelMap voxel_grid{};
        for (const cv::Point3f& p : pc) {
            VoxelCoord vc{
                (int)std::floor(p.x / voxel_size_m),
                (int)std::floor(p.y / voxel_size_m),
                (int)std::floor(p.z / voxel_size_m)
            };

            auto& v = voxel_grid[vc];
            v.count++;
        }
        return voxel_grid;
    }

    cv::Mat ExtractRoi(const cv::Mat& frame, const cv::Rect& roi) const {
        if (frame.empty()) {
            return cv::Mat();
        }

        // Clamp ROI to frame bounds
        cv::Rect safe_roi = roi & cv::Rect(0, 0, frame.cols, frame.rows);

        if (safe_roi.width <= 0 || safe_roi.height <= 0) {
            return cv::Mat();
        }

        return frame(safe_roi).clone();
    }

    void SetVertFlip(bool enabled) {
        vert_flip_enabled_ = enabled;
    }

    void SetSpatialFilterEnabled(bool enabled) {
        spatial_filter_enabled_ = enabled;
    }

    void SetTemporalFilterEnabled(bool enabled) {
        temporal_filter_enabled_ = enabled;
        if (!enabled) {
            temporal_buffer_.clear();
        }
    }

    void SetTemporalFilterAlpha(float alpha) {
        // Clamp to valid range
        temporal_alpha_ = std::clamp(alpha, 0.0f, 1.0f);
    }

private:
    bool vert_flip_enabled_;
    bool spatial_filter_enabled_;
    bool temporal_filter_enabled_;
    float temporal_alpha_;
    int frame_number_;
    std::deque<cv::Mat> temporal_buffer_;

    uint64_t GetTimestampNs() const {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
            now.time_since_epoch()).count();
    }
};

// ============================================================================
// Public Interface Implementation
// ============================================================================

FrameProcessor::FrameProcessor() 
: impl_(std::make_unique<Impl>()) {}

FrameProcessor::~FrameProcessor() = default;

ProcessedFrame FrameProcessor::Process(const cv::Mat& raw_depth) {
    return impl_->Process(raw_depth);
}

void FrameProcessor::ApplySpatialFilter(cv::Mat& frame, int kernel_size) {
    impl_->ApplySpatialFilter(frame, kernel_size);
}

void FrameProcessor::ApplyTemporalFilter(cv::Mat& frame) {
    impl_->ApplyTemporalFilter(frame);
}

PointCloud FrameProcessor::ToPointCloud(const cv::Mat& depth,
                                        const CameraIntrinsics& intrinsics,
                                        const int row_inc,
                                        const int col_inc) const {
    return impl_->ToPointCloud(depth, intrinsics);
}

VoxelMap FrameProcessor::ToVoxelMap(const PointCloud& pc, float voxel_size_m) const {
    return impl_->ToVoxelMap(pc, voxel_size_m);
}

cv::Mat FrameProcessor::ExtractRoi(const cv::Mat& frame, 
                                   const cv::Rect& roi) const {
    return impl_->ExtractRoi(frame, roi);
}

void FrameProcessor::SetVertFlip(bool enabled) {
    impl_->SetVertFlip(enabled);
}

void FrameProcessor::SetSpatialFilterEnabled(bool enabled) {
    impl_->SetSpatialFilterEnabled(enabled);
}

void FrameProcessor::SetTemporalFilterEnabled(bool enabled) {
    impl_->SetTemporalFilterEnabled(enabled);
}

void FrameProcessor::SetTemporalFilterAlpha(float alpha) {
    impl_->SetTemporalFilterAlpha(alpha);
}

}  // namespace trail_nav::vision
