#ifndef TRAIL_NAVIGATION_VISION_GROUND_ESTIMATOR_H_
#define TRAIL_NAVIGATION_VISION_GROUND_ESTIMATOR_H_

#include "trail_sense/core/types.h"
#include <opencv2/core.hpp>
#include <memory>

namespace trail_nav {
namespace vision {

// Estimates ground plane from depth data.
//
// Uses RANSAC or least-squares fitting to find dominant ground plane.
// Tracks plane parameters across frames for temporal stability.
// Handles varying terrain: slopes, stairs, uneven ground, roots.
// Provides height-above-ground measurement for scene points.
// Critical for distinguishing trip hazards from normal ground variation.
class GroundEstimator {
public:
    GroundEstimator();
    ~GroundEstimator();

    // Estimates ground plane from point cloud using RANSAC.
    // points: point cloud in camera coordinates
    // Returns best-fit plane model
    PlaneModel EstimateGroundPlane(const PointCloud& points);

    // Returns new pointcould removing points belonging to
    // the current ground plane.
    // Typically called before converting pointcloud to voxels
    PointCloud RemoveGroundPoints(const PointCloud& pc) const;

    // Calculates signed height of point above current ground plane.
    // Negative values indicate point is below ground (dips, holes).
    float GetHeightAboveGround(const cv::Point3f& point) const;

    // Returns current ground plane parameters (normal vector + distance).
    PlaneModel GetCurrentGroundPlane() const;

    // Updates ground plane estimation with temporal smoothing.
    // Prevents jitter while allowing adaptation to terrain changes.
    void UpdateGroundPlane(const PlaneModel& new_plane);

    // Checks if ground estimation is confident and stable.
    // Returns false during initialization or on challenging terrain.
    bool IsStable() const;

    // Resets estimation state (e.g., when environment changes drastically).
    void Reset();

    // Configuration
    void SetRansacIterations(int iterations);
    void SetInlierThreshold(float threshold_m);
    void SetTemporalSmoothingFactor(float alpha);
    void SetConfidenceThreshold(float conf);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace vision
}  // namespace trail_nav

#endif  // TRAIL_NAVIGATION_VISION_GROUND_ESTIMATOR_H_
