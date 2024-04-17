#pragma once

#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/navigation/ManifoldPreintegration.h>
#include <gtsam/navigation/TangentPreintegration.h>
#include <gtsam/base/debug.h>
#include <eigen3/Eigen/Dense>
#include <ceres/ceres.h>
#include "factor/marginalization_factor.h"

#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/navigation/ManifoldPreintegration.h>
#include <gtsam/navigation/TangentPreintegration.h>

using namespace std;
using namespace gtsam;
using namespace ceres;



class MYMARFactor: public NoiseModelFactor1<double**> 
{

private:
  // measurement information
  typedef MYMARFactor This;
  typedef NoiseModelFactor1<double**> Base;

  MarginalizationInfo* marginalization_info;
  // PreintegratedImuMeasurements _PIM_;
public:

  /**
   * Constructor
   * @param poseKey    associated pose varible key
   * @param model      noise model for GPS snesor, in X-Y
   * @param m          Point2 measurement
   */
  Matrix9 preintMeasCov_;
  MYMARFactor(Key pose_i, MarginalizationInfo* info) :
      Base(noiseModel::Gaussian::Covariance(preintMeasCov_ ), pose_i), marginalization_info(info) {
  }
  ~MYMARFactor() override {
  }

  // error function
  // @param p    the pose in Pose2
  // @param H    the optional Jacobian matrix, which use boost optional and has default null pointer

  gtsam::Vector evaluateError(double const *const *parameters,
      boost::optional<gtsam::Matrix&> H1 = boost::none) const {
  
    // note that use boost optional like a pointer
    // only calculate jacobian matrix when non-null pointer exists
    int n = marginalization_info->n;
    int m = marginalization_info->m;
    Eigen::VectorXd dx(n);
    for (int i = 0; i < static_cast<int>(marginalization_info->keep_block_size.size()); i++)
    {
        int size = marginalization_info->keep_block_size[i];
        int idx = marginalization_info->keep_block_idx[i] - m;
        Eigen::VectorXd x = Eigen::Map<const Eigen::VectorXd>(parameters[i], size);
        Eigen::VectorXd x0 = Eigen::Map<const Eigen::VectorXd>(marginalization_info->keep_block_data[i], size);
        if (size != 7)
            dx.segment(idx, size) = x - x0;
        else
        {
            dx.segment<3>(idx + 0) = x.head<3>() - x0.head<3>();
            dx.segment<3>(idx + 3) = 2.0 * Utility::positify(Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() * Eigen::Quaterniond(x(6), x(3), x(4), x(5))).vec();
            if (!((Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() * Eigen::Quaterniond(x(6), x(3), x(4), x(5))).w() >= 0))
            {
                dx.segment<3>(idx + 3) = 2.0 * -Utility::positify(Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() * Eigen::Quaterniond(x(6), x(3), x(4), x(5))).vec();
            }
        }
    }
    Eigen::VectorXd residuals(n);
    residuals = marginalization_info->linearized_residuals + marginalization_info->linearized_jacobians * dx;
    if (H1)
    {

        for (int i = 0; i < static_cast<int>(marginalization_info->keep_block_size.size()); i++)
        {
            if (H1)
            {
                int size = marginalization_info->keep_block_size[i], local_size = marginalization_info->localSize(size);
                int idx = marginalization_info->keep_block_idx[i] - m;
                Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> jacobian(n, size);
                jacobian.setZero();
                jacobian.leftCols(local_size) = marginalization_info->linearized_jacobians.middleCols(idx, local_size);
                *H1 =jacobian;
            }
        }
    }
    return residuals;
  }

};

int main(){
  Rot3 rots(1.0, 0.0, 0.0, 0.0);
  Point3 points(0.0, 0.0, 1.0);
  Pose3 position(rots, points);
  Vector3 vi(1.0,2.0,3.0);

  pair<Pose3, Vector3> p1(position, vi);
  
  imuBias::ConstantBias bias_i(vi,vi);
  cout << position.rotation().quaternion()[0] << endl;
  cout << position.translation()[2] << endl;
  cout << bias_i.accelerometer()[0] << endl;
  cout << bias_i.gyroscope()[0] << endl;
  cout << p1.first.rotation().quaternion()[0] << endl;

}

