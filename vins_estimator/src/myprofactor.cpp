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
#include "parameters.h"
#include "utility/utility.h"
#include "utility/tic_toc.h"

#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/navigation/ManifoldPreintegration.h>
#include <gtsam/navigation/TangentPreintegration.h>

using namespace std;
using namespace gtsam;
using namespace ceres;



class MYPROFactor: public NoiseModelFactor5<Pose3, Pose3, Pose3, double, double> 
{

private:
  // measurement information
  typedef MYPROFactor This;
  typedef NoiseModelFactor5<Pose3, Pose3, Pose3, double, double> Base;

  Eigen::Vector3d pts_i;
  Eigen::Vector3d pts_j;
  Eigen::Vector3d velocity_i;
  Eigen::Vector3d velocity_j;
  double td_i;
  double td_j;
  double row_i;
  double row_j;
  static Eigen::Matrix2d sqrt_info;
  static double sum_t;
//   Eigen::Matrix<double, 2, 3> tangent_base;
  // PreintegratedImuMeasurements _PIM_;
public:

  /**
   * Constructor
   * @param poseKey    associated pose varible key
   * @param model      noise model for GPS snesor, in X-Y
   * @param m          Point2 measurement
   */
  Matrix9 preintMeasCov_;
  MYPROFactor(Key pose_i,Key pose_j,Key pose_k,Key dep_i,Key td, 
  const Eigen::Vector3d &_pts_i, const Eigen::Vector3d &_pts_j,
  const Eigen::Vector2d &_velocity_i, const Eigen::Vector2d &_velocity_j,
  const double _td_i, const double _td_j, const double _row_i, const double _row_j) :
      Base(noiseModel::Gaussian::Covariance(preintMeasCov_ ), pose_i, pose_j, pose_k, dep_i, td), 
      pts_i(_pts_i),  pts_j(_pts_j),
      td_i(_td_i),td_j(_td_j),row_i(_row_i),row_j(_row_j){
        velocity_i.x() = _velocity_i.x();
        velocity_i.y() = _velocity_i.y();
        velocity_i.z() = 0;
        velocity_j.x() = _velocity_j.x();
        velocity_j.y() = _velocity_j.y();
        velocity_j.z() = 0;
  }
  ~MYPROFactor() override {
  }

  // error function
  // @param p    the pose in Pose2
  // @param H    the optional Jacobian matrix, which use boost optional and has default null pointer

  gtsam::Vector evaluateError(const Pose3& pose_i, const Pose3& pose_j, const Pose3& pose_k,
    const double& dep_i,const double& td_i,
    boost::optional<gtsam::Matrix&> H1 = boost::none, boost::optional<gtsam::Matrix&> H2 = boost::none,
    boost::optional<gtsam::Matrix&> H3 = boost::none, boost::optional<gtsam::Matrix&> H4 = boost::none,
    boost::optional<gtsam::Matrix&> H5 = boost::none) const {
  
    // note that use boost optional like a pointer
    // only calculate jacobian matrix when non-null pointer exists
    TicToc tic_toc;
    Eigen::Vector3d Pi(pose_i.translation()[0], pose_i.translation()[1], pose_i.translation()[2]);
    Eigen::Quaterniond Qi(pose_i.rotation().quaternion()[0], pose_i.rotation().quaternion()[1], pose_i.rotation().quaternion()[2], pose_i.rotation().quaternion()[3]);
    
    Eigen::Vector3d Pj(pose_j.translation()[0], pose_j.translation()[1], pose_j.translation()[2]);
    Eigen::Quaterniond Qj(pose_j.rotation().quaternion()[0], pose_j.rotation().quaternion()[1], pose_j.rotation().quaternion()[2], pose_j.rotation().quaternion()[3]);
    
    Eigen::Vector3d tic(pose_k.translation()[0], pose_k.translation()[1], pose_k.translation()[2]);
    Eigen::Quaterniond qic(pose_k.rotation().quaternion()[0], pose_k.rotation().quaternion()[1], pose_k.rotation().quaternion()[2], pose_k.rotation().quaternion()[3]);

    double inv_dep_i = dep_i;

    double td = td_i;

    Eigen::Vector3d pts_i_td, pts_j_td;
    pts_i_td = pts_i - (td - td_i + TR / ROW * row_i) * velocity_i;
    pts_j_td = pts_j - (td - td_j + TR / ROW * row_j) * velocity_j;
    Eigen::Vector3d pts_camera_i = pts_i_td / inv_dep_i;
    Eigen::Vector3d pts_imu_i = qic * pts_camera_i + tic;
    Eigen::Vector3d pts_w = Qi * pts_imu_i + Pi;
    Eigen::Vector3d pts_imu_j = Qj.inverse() * (pts_w - Pj);
    Eigen::Vector3d pts_camera_j = qic.inverse() * (pts_imu_j - tic);
    Eigen::Vector2d residual;

#ifdef UNIT_SPHERE_ERROR 
    residual =  tangent_base * (pts_camera_j.normalized() - pts_j_td.normalized());
#else
    double dep_j = pts_camera_j.z();
    residual = (pts_camera_j / dep_j).head<2>() - pts_j_td.head<2>();
#endif

    residual = sqrt_info * residual;

    if (H1)
    {
        Eigen::Matrix3d Ri = Qi.toRotationMatrix();
        Eigen::Matrix3d Rj = Qj.toRotationMatrix();
        Eigen::Matrix3d ric = qic.toRotationMatrix();
        Eigen::Matrix<double, 2, 3> reduce(2, 3);
#ifdef UNIT_SPHERE_ERROR
        double norm = pts_camera_j.norm();
        Eigen::Matrix3d norm_jaco;
        double x1, x2, x3;
        x1 = pts_camera_j(0);
        x2 = pts_camera_j(1);
        x3 = pts_camera_j(2);
        norm_jaco << 1.0 / norm - x1 * x1 / pow(norm, 3), - x1 * x2 / pow(norm, 3),            - x1 * x3 / pow(norm, 3),
                     - x1 * x2 / pow(norm, 3),            1.0 / norm - x2 * x2 / pow(norm, 3), - x2 * x3 / pow(norm, 3),
                     - x1 * x3 / pow(norm, 3),            - x2 * x3 / pow(norm, 3),            1.0 / norm - x3 * x3 / pow(norm, 3);
        reduce = tangent_base * norm_jaco;
#else
        reduce << 1. / dep_j, 0, -pts_camera_j(0) / (dep_j * dep_j),
            0, 1. / dep_j, -pts_camera_j(1) / (dep_j * dep_j);
#endif
        reduce = sqrt_info * reduce;

        if (H1)
        {
            Eigen::Matrix<double, 2, 7, Eigen::RowMajor> jacobian_pose_i;

            Eigen::Matrix<double, 3, 6> jaco_i;
            jaco_i.leftCols<3>() = ric.transpose() * Rj.transpose();
            jaco_i.rightCols<3>() = ric.transpose() * Rj.transpose() * Ri * -Utility::skewSymmetric(pts_imu_i);

            jacobian_pose_i.leftCols<6>() = reduce * jaco_i;
            jacobian_pose_i.rightCols<1>().setZero();

            *H1 = jacobian_pose_i;
        }

        if (H2)
        {
            Eigen::Matrix<double, 2, 7, Eigen::RowMajor> jacobian_pose_j;

            Eigen::Matrix<double, 3, 6> jaco_j;
            jaco_j.leftCols<3>() = ric.transpose() * -Rj.transpose();
            jaco_j.rightCols<3>() = ric.transpose() * Utility::skewSymmetric(pts_imu_j);

            jacobian_pose_j.leftCols<6>() = reduce * jaco_j;
            jacobian_pose_j.rightCols<1>().setZero();

            *H2 = jacobian_pose_j;
        }
        if (H3)
        {
            Eigen::Matrix<double, 2, 7, Eigen::RowMajor> jacobian_ex_pose;
            Eigen::Matrix<double, 3, 6> jaco_ex;
            jaco_ex.leftCols<3>() = ric.transpose() * (Rj.transpose() * Ri - Eigen::Matrix3d::Identity());
            Eigen::Matrix3d tmp_r = ric.transpose() * Rj.transpose() * Ri * ric;
            jaco_ex.rightCols<3>() = -tmp_r * Utility::skewSymmetric(pts_camera_i) + Utility::skewSymmetric(tmp_r * pts_camera_i) +
                                     Utility::skewSymmetric(ric.transpose() * (Rj.transpose() * (Ri * tic + Pi - Pj) - tic));
            jacobian_ex_pose.leftCols<6>() = reduce * jaco_ex;
            jacobian_ex_pose.rightCols<1>().setZero();

            *H3 = jacobian_ex_pose;
        }
        if (H4)
        {
            Eigen::Vector2d jacobian_feature;
            jacobian_feature = reduce * ric.transpose() * Rj.transpose() * Ri * ric * pts_i_td * -1.0 / (inv_dep_i * inv_dep_i);

            *H4 = jacobian_feature;
        }
        if (H5)
        {
            Eigen::Vector2d jacobian_td;
            jacobian_td = reduce * ric.transpose() * Rj.transpose() * Ri * ric * velocity_i / inv_dep_i * -1.0  +
                          sqrt_info * velocity_j.head(2);

            *H5 = jacobian_td;
        }
    }
    sum_t += tic_toc.toc();

    return residual;
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

