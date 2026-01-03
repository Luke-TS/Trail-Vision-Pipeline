#include "trail_sense/vision/ground_estimator.h"
#include <cstddef>
#include <deque>
#include <opencv2/core.hpp>
#include <random>
#include <algorithm>
#include <cmath>

namespace trail_nav {
namespace vision {

class GroundEstimator::Impl {
public:
    Impl()
        : ransac_iterations_(100),
        inlier_threshold_(0.05f),  // 5cm tolerance
        temporal_alpha_(0.8f),      // Heavy smoothing for stability
        is_initialized_(false),
        confidence_threshold_(0.5f),
        min_inliers_(100) {
        // Initialize with default horizontal ground plane (y-up coordinate system)
        current_plane_.normal = cv::Vec3f(0, -1, 0);
        current_plane_.distance = 0.0f;
        current_plane_.confidence = 0.0f;
        current_plane_.inlier_count = 0;
    }

    PlaneModel EstimateGroundPlane(const PointCloud& points) {
        if (points.size() < 3) {
            return current_plane_;  // Need at least 3 points for a plane
        }

        PlaneModel best_plane;
        best_plane.confidence = 0.0f;
        best_plane.inlier_count = 0;

        size_t num_points = points.size();

        for (int iter = 0; iter < ransac_iterations_; ++iter) {
            // Randomly select 3 points
            int idx1 = RandomPointIndex(num_points);
            int idx2 = RandomPointIndex(num_points);
            int idx3 = RandomPointIndex(num_points);

            // Ensure they're different
            if (idx1 == idx2 || idx1 == idx3 || idx2 == idx3) {
                continue;
            }

            cv::Point3f p1 = points[idx1];
            cv::Point3f p2 = points[idx2];
            cv::Point3f p3 = points[idx3];

            // Compute plane from 3 points
            PlaneModel candidate = FitPlaneToPoints(p1, p2, p3);

            // Check if plane is reasonable (not too vertical)
            // Ground should be mostly horizontal
            float verticality = std::abs(candidate.normal[1]);
            if (verticality < 0.5f) {  // Normal should point mostly up/down
                continue;
            }

            // Count inliers
            int inlier_count = 0;
            for (const auto& point : points) {
                float distance = PointToPlaneDistance(point, candidate);
                if (std::abs(distance) < inlier_threshold_) {
                    inlier_count++;
                }
            }

            // Update best plane if this is better
            if (inlier_count > best_plane.inlier_count) {
                best_plane = candidate;
                best_plane.inlier_count = inlier_count;
            }
        }

        // Calculate confidence based on inlier percentage
        if (points.size() > 0) {
            float inlier_ratio = static_cast<float>(best_plane.inlier_count) / 
                static_cast<float>(points.size());
            best_plane.confidence = std::clamp(inlier_ratio, 0.0f, 1.0f);
        }

        // Refine plane using all inliers (least squares)
        if (best_plane.inlier_count >= min_inliers_) {
            best_plane = RefinePlaneWithInliers(points, best_plane);
        }

        return best_plane;
    }

    PointCloud RemoveGroundPoints(const PointCloud& pc) const {
        PointCloud result{};
        for (const auto& p : pc) {
            if ( std::abs(GetHeightAboveGround(p)) < inlier_threshold_) {
                result.push_back(p);
            }
        }
        return result;
    }

    float GetHeightAboveGround(const cv::Point3f& point) const {
        if (!is_initialized_) {
            return 0.0f;
        }

        // Signed distance from point to plane
        // Positive = above ground, negative = below ground
        return PointToPlaneDistance(point, current_plane_);
    }

    PlaneModel GetCurrentGroundPlane() const {
        return current_plane_;
    }

    void UpdateGroundPlane(const PlaneModel& new_plane) {
        if (!is_initialized_) {
            // First observation - accept it directly
            current_plane_ = new_plane;
            is_initialized_ = true;
            return;
        }

        // Check if new plane is valid
        if (new_plane.confidence < confidence_threshold_) {
            // Don't update with low-confidence estimates
            return;
        }

        // Temporal smoothing using exponential moving average
        // Blend new plane with current plane

        // Smooth normal vector
        cv::Vec3f blended_normal = 
            temporal_alpha_ * current_plane_.normal + 
            (1.0f - temporal_alpha_) * new_plane.normal;

        // Normalize the blended normal
        float norm = cv::norm(blended_normal);
        if (norm > 0.0f) {
            blended_normal = blended_normal / norm;
        }

        // Smooth distance
        float blended_distance = 
            temporal_alpha_ * current_plane_.distance + 
            (1.0f - temporal_alpha_) * new_plane.distance;

        // Update current plane
        current_plane_.normal = blended_normal;
        current_plane_.distance = blended_distance;
        current_plane_.confidence = new_plane.confidence;
        current_plane_.inlier_count = new_plane.inlier_count;

        // Add to history for stability checking
        plane_history_.push_back(current_plane_);
        if (plane_history_.size() > 10) {
            plane_history_.pop_front();
        }
    }

    bool IsStable() const {
        if (!is_initialized_) {
            return false;
        }

        // Check confidence
        if (current_plane_.confidence < confidence_threshold_) {
            return false;
        }

        // Check if we have enough history
        if (plane_history_.size() < 5) {
            return false;
        }

        // Calculate variance in normal direction over recent history
        float sum_x = 0, sum_y = 0, sum_z = 0;
        for (const auto& plane : plane_history_) {
            sum_x += plane.normal[0];
            sum_y += plane.normal[1];
            sum_z += plane.normal[2];
        }

        float mean_x = sum_x / plane_history_.size();
        float mean_y = sum_y / plane_history_.size();
        float mean_z = sum_z / plane_history_.size();

        float variance = 0.0f;
        for (const auto& plane : plane_history_) {
            float dx = plane.normal[0] - mean_x;
            float dy = plane.normal[1] - mean_y;
            float dz = plane.normal[2] - mean_z;
            variance += dx*dx + dy*dy + dz*dz;
        }
        variance /= plane_history_.size();

        // Stable if variance is low (normal vector isn't changing much)
        return variance < 0.01f;  // Threshold for stability
    }

    void Reset() {
        current_plane_.normal = cv::Vec3f(0, -1, 0);
        current_plane_.distance = 0.0f;
        current_plane_.confidence = 0.0f;
        current_plane_.inlier_count = 0;
        is_initialized_ = false;
        plane_history_.clear();
    }

    void SetRansacIterations(int iterations) {
        ransac_iterations_ = std::max(10, iterations);
    }

    void SetInlierThreshold(float threshold_m) {
        inlier_threshold_ = std::max(0.01f, threshold_m);
    }

    void SetTemporalSmoothingFactor(float alpha) {
        temporal_alpha_ = std::clamp(alpha, 0.0f, 1.0f);
    }

    void SetConfidenceThreshold(float confidence) {
        confidence_threshold_ = std::clamp(confidence, 0.0f, 0.8f);
    }

private:
    int ransac_iterations_;
    float inlier_threshold_;
    float temporal_alpha_;
    bool is_initialized_;
    float confidence_threshold_;
    int min_inliers_;
    PlaneModel current_plane_;
    std::deque<PlaneModel> plane_history_;

    // Fits a plane to three points
    // Plane equation: ax + by + cz + d = 0
    // Normal: (a, b, c), Distance: d
    PlaneModel FitPlaneToPoints(const cv::Point3f& p1, 
                                const cv::Point3f& p2,
                                const cv::Point3f& p3) const {
        PlaneModel plane;

        // Create two vectors in the plane
        cv::Vec3f v1(p2.x - p1.x, p2.y - p1.y, p2.z - p1.z);
        cv::Vec3f v2(p3.x - p1.x, p3.y - p1.y, p3.z - p1.z);

        // Cross product gives normal vector
        cv::Vec3f normal = v1.cross(v2);

        // Normalize
        float norm = cv::norm(normal);
        if (norm > 0.0f) {
            normal = normal / norm;
        } else {
            // Degenerate case - points are collinear
            normal = cv::Vec3f(0, -1, 0);
        }

        // Ensure normal points upward (negative y direction in camera coords)
        // Camera coordinate system: x-right, y-down, z-forward
        if (normal[1] > 0) {
            normal = -normal;
        }

        // Calculate distance using point p1
        // d = -(ax + by + cz)
        float distance = -(normal[0] * p1.x + normal[1] * p1.y + normal[2] * p1.z);

        plane.normal = normal;
        plane.distance = distance;
        plane.confidence = 0.0f;
        plane.inlier_count = 0;

        return plane;
    }

    size_t RandomPointIndex(size_t num_points) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis01(0.0f, 1.0f);

        float r = dis01(gen);

        // Bias toward 1.0 (bottom)
        float bias = 1 - (r * r);     // r^2 tends towards 0

        size_t idx = static_cast<size_t>(bias * (num_points - 1));
        return idx;
    }

    // Calculates signed distance from point to plane
    float PointToPlaneDistance(const cv::Point3f& point, 
                               const PlaneModel& plane) const {
        // Distance = ax + by + cz + d
        return plane.normal[0] * point.x + 
        plane.normal[1] * point.y + 
        plane.normal[2] * point.z + 
        plane.distance;
    }

    // Refines plane estimate using least squares on all inliers
    PlaneModel RefinePlaneWithInliers(const PointCloud& points,
                                      const PlaneModel& initial_plane) const {
        // Collect inlier points
        std::vector<cv::Point3f> inliers;
        inliers.reserve(initial_plane.inlier_count);

        for (const auto& point : points) {
            float distance = PointToPlaneDistance(point, initial_plane);
            if (std::abs(distance) < inlier_threshold_) {
                inliers.push_back(point);
            }
        }

        if (inliers.size() < 3) {
            return initial_plane;
        }

        // Least squares plane fitting using SVD
        // Build matrix of points (centered around centroid)
        cv::Point3f centroid(0, 0, 0);
        for (const auto& p : inliers) {
            centroid.x += p.x;
            centroid.y += p.y;
            centroid.z += p.z;
        }
        centroid.x /= inliers.size();
        centroid.y /= inliers.size();
        centroid.z /= inliers.size();

        // Create centered point matrix
        cv::Mat A(inliers.size(), 3, CV_32F);
        for (size_t i = 0; i < inliers.size(); ++i) {
            A.at<float>(i, 0) = inliers[i].x - centroid.x;
            A.at<float>(i, 1) = inliers[i].y - centroid.y;
            A.at<float>(i, 2) = inliers[i].z - centroid.z;
        }

        // Perform SVD
        cv::Mat w, u, vt;
        cv::SVD::compute(A, w, u, vt);

        // Normal is last row of Vt (smallest singular value)
        cv::Vec3f normal(vt.at<float>(2, 0), 
                         vt.at<float>(2, 1), 
                         vt.at<float>(2, 2));

        // Ensure normal points upward
        if (normal[1] > 0) {
            normal = -normal;
        }

        // Calculate distance
        float distance = -(normal[0] * centroid.x + 
            normal[1] * centroid.y + 
            normal[2] * centroid.z);

        PlaneModel refined_plane;
        refined_plane.normal = normal;
        refined_plane.distance = distance;
        refined_plane.confidence = initial_plane.confidence;
        refined_plane.inlier_count = inliers.size();

        return refined_plane;
    }
};

// ============================================================================
// Public Interface Implementation
// ============================================================================

GroundEstimator::GroundEstimator() 
: impl_(std::make_unique<Impl>()) {}

GroundEstimator::~GroundEstimator() = default;

PlaneModel GroundEstimator::EstimateGroundPlane(const PointCloud& points) {
    return impl_->EstimateGroundPlane(points);
}

PointCloud GroundEstimator::RemoveGroundPoints(const PointCloud& pc) const {
    return impl_->RemoveGroundPoints(pc);
}

float GroundEstimator::GetHeightAboveGround(const cv::Point3f& point) const {
    return impl_->GetHeightAboveGround(point);
}

PlaneModel GroundEstimator::GetCurrentGroundPlane() const {
    return impl_->GetCurrentGroundPlane();
}

void GroundEstimator::UpdateGroundPlane(const PlaneModel& new_plane) {
    impl_->UpdateGroundPlane(new_plane);
}

bool GroundEstimator::IsStable() const {
    return impl_->IsStable();
}

void GroundEstimator::Reset() {
    impl_->Reset();
}

void GroundEstimator::SetRansacIterations(int iterations) {
    impl_->SetRansacIterations(iterations);
}

void GroundEstimator::SetInlierThreshold(float threshold_m) {
    impl_->SetInlierThreshold(threshold_m);
}

void GroundEstimator::SetTemporalSmoothingFactor(float alpha) {
    impl_->SetTemporalSmoothingFactor(alpha);
}

void GroundEstimator::SetConfidenceThreshold(float conf) {
    impl_->SetConfidenceThreshold(conf);
}

}  // namespace vision
}  // namespace trail_nav
