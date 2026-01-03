#ifndef TRAIL_NAVIGATION_VISION_FRAME_PROCESSOR_H_
#define TRAIL_NAVIGATION_VISION_FRAME_PROCESSOR_H_

#include "trail_sense/core/types.h"
#include <opencv2/core.hpp>
#include <memory>

namespace trail_nav {
namespace vision {

// Processes incoming depth and color frames.
//
// Handles frame preprocessing: spatial/temporal filtering, normalization.
// Manages frame buffering and timestamp synchronization.
// Converts raw depth data to point clouds for geometric analysis.
// Applies noise reduction appropriate for outdoor trail conditions.
class FrameProcessor {
public:
    FrameProcessor();
    ~FrameProcessor();

    // Processes raw depth frame into filtered, calibrated format.
    ProcessedFrame Process(const cv::Mat& raw_depth);

    // Applies spatial median filter to reduce depth noise.
    // kernel_size: typically 3, 5, or 7 for different smoothing levels
    void ApplySpatialFilter(cv::Mat& frame, int kernel_size);

    // Applies temporal averaging filter across recent frames.
    // Reduces flickering but may introduce lag.
    void ApplyTemporalFilter(cv::Mat& frame);

    // Converts depth frame to 3D point cloud in camera coordinates.
    // Uses camera intrinsics to unproject pixels to 3D.
    // Will convert every <row_inc> pixel row
    // and every <col_inc> pixel column
    PointCloud ToPointCloud(const cv::Mat& depth,
                            const CameraIntrinsics& intrinsics,
                            const int row_inc = 1,
                            const int col_inc = 1) const;

    // creates std::unordered_map<VoxelCoord, VoxelStats, VoxelHash>;
    // using voxel size 'voxel_size_m' in meters
    VoxelMap ToVoxelMap(const PointCloud& pc, float voxel_size_m) const;

    // Extracts rectangular region of interest from frame.
    cv::Mat ExtractRoi(const cv::Mat& frame, const cv::Rect& roi) const;

    // Configuration
    void SetVertFlip(bool enabled);
    void SetSpatialFilterEnabled(bool enabled);
    void SetTemporalFilterEnabled(bool enabled);
    void SetTemporalFilterAlpha(float alpha);  // 0.0 = no history, 1.0 = only history

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace vision
}  // namespace trail_nav

#endif  // TRAIL_NAVIGATION_VISION_FRAME_PROCESSOR_H_
