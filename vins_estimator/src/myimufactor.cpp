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
#include "factor/integration_base.h"

#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/navigation/ManifoldPreintegration.h>
#include <gtsam/navigation/TangentPreintegration.h>

using namespace std;
using namespace gtsam;
using namespace ceres;



class MYIMUFactor: public NoiseModelFactor4<Pose3, pair<Vector3,imuBias::ConstantBias>, Pose3, pair<Vector3,imuBias::ConstantBias>> 
{

private:
  // measurement information
  typedef MYIMUFactor This;
  typedef NoiseModelFactor4<Pose3, pair<Vector3,imuBias::ConstantBias>, Pose3, pair<Vector3,imuBias::ConstantBias>> Base;

  IntegrationBase* pre_integration;
  // PreintegratedImuMeasurements _PIM_;
public:

  /**
   * Constructor
   * @param poseKey    associated pose varible key
   * @param model      noise model for GPS snesor, in X-Y
   * @param m          Point2 measurement
   */
  Matrix9 preintMeasCov_;
  MYIMUFactor(Key pose_i, Key vel_i, Key pose_j, Key vel_j, IntegrationBase* pim) :
      Base(noiseModel::Gaussian::Covariance(preintMeasCov_ ), pose_i, vel_i,
        pose_j, vel_j), pre_integration(pim) {
  }
  ~MYIMUFactor() override {
  }

  // error function
  // @param p    the pose in Pose2
  // @param H    the optional Jacobian matrix, which use boost optional and has default null pointer

  gtsam::Vector evaluateError(const Pose3& pose_i, const pair<Vector3,imuBias::ConstantBias>& vel_i,
      const Pose3& pose_j, const pair<Vector3,imuBias::ConstantBias>& vel_j,
      boost::optional<gtsam::Matrix&> H1 = boost::none,boost::optional<gtsam::Matrix&> H2 = boost::none,
      boost::optional<gtsam::Matrix&> H3 = boost::none,boost::optional<gtsam::Matrix&> H4 = boost::none) const {
  
    // note that use boost optional like a pointer
    // only calculate jacobian matrix when non-null pointer exists

    Eigen::Vector3d Pi(pose_i.translation()[0], pose_i.translation()[1], pose_i.translation()[2]);
    Eigen::Quaterniond Qi(pose_i.rotation().quaternion()[0], pose_i.rotation().quaternion()[1], pose_i.rotation().quaternion()[2], pose_i.rotation().quaternion()[3]);

    Eigen::Vector3d Vi(vel_i.first[0], vel_i.first[1], vel_i.first[2]);
    Eigen::Vector3d Bai(vel_i.second.accelerometer()[0], vel_i.second.accelerometer()[1], vel_i.second.accelerometer()[2]);
    Eigen::Vector3d Bgi(vel_i.second.gyroscope()[0], vel_i.second.gyroscope()[1], vel_i.second.gyroscope()[2]);

    Eigen::Vector3d Pj(pose_j.translation()[0], pose_j.translation()[1], pose_j.translation()[2]);
    Eigen::Quaterniond Qj(pose_j.rotation().quaternion()[0], pose_j.rotation().quaternion()[1], pose_j.rotation().quaternion()[2], pose_j.rotation().quaternion()[3]);

    Eigen::Vector3d Vj(vel_j.first[0], vel_j.first[1], vel_j.first[2]);
    Eigen::Vector3d Baj(vel_j.second.accelerometer()[0], vel_j.second.accelerometer()[1], vel_j.second.accelerometer()[2]);
    Eigen::Vector3d Bgj(vel_j.second.gyroscope()[0], vel_j.second.gyroscope()[1], vel_j.second.gyroscope()[2]);

    //
    Eigen::Matrix<double, 15, 1> residual;
    residual = pre_integration->evaluate(Pi, Qi, Vi, Bai, Bgi,
                                        Pj, Qj, Vj, Baj, Bgj);

    Eigen::Matrix<double, 15, 15> sqrt_info = Eigen::LLT<Eigen::Matrix<double, 15, 15>>(pre_integration->covariance.inverse()).matrixL().transpose();
    //sqrt_info.setIdentity();
    residual = sqrt_info * residual;

    if (H1)
    {
        double sum_dt = pre_integration->sum_dt;
        Eigen::Matrix3d dp_dba = pre_integration->jacobian.template block<3, 3>(O_P, O_BA);
        Eigen::Matrix3d dp_dbg = pre_integration->jacobian.template block<3, 3>(O_P, O_BG);

        Eigen::Matrix3d dq_dbg = pre_integration->jacobian.template block<3, 3>(O_R, O_BG);

        Eigen::Matrix3d dv_dba = pre_integration->jacobian.template block<3, 3>(O_V, O_BA);
        Eigen::Matrix3d dv_dbg = pre_integration->jacobian.template block<3, 3>(O_V, O_BG);

        if (pre_integration->jacobian.maxCoeff() > 1e8 || pre_integration->jacobian.minCoeff() < -1e8)
        {
            ROS_WARN("numerical unstable in preintegration");
            //std::cout << pre_integration->jacobian << std::endl;
///                ROS_BREAK();
        }

        if (H1)
        {
            Eigen::Matrix<double, 15, 7, Eigen::RowMajor> jacobian_pose_i;
            jacobian_pose_i.setZero();

            jacobian_pose_i.block<3, 3>(O_P, O_P) = -Qi.inverse().toRotationMatrix();
            jacobian_pose_i.block<3, 3>(O_P, O_R) = Utility::skewSymmetric(Qi.inverse() * (0.5 * G * sum_dt * sum_dt + Pj - Pi - Vi * sum_dt));

            Eigen::Quaterniond corrected_delta_q = pre_integration->delta_q * Utility::deltaQ(dq_dbg * (Bgi - pre_integration->linearized_bg));
            jacobian_pose_i.block<3, 3>(O_R, O_R) = -(Utility::Qleft(Qj.inverse() * Qi) * Utility::Qright(corrected_delta_q)).bottomRightCorner<3, 3>();

            jacobian_pose_i.block<3, 3>(O_V, O_R) = Utility::skewSymmetric(Qi.inverse() * (G * sum_dt + Vj - Vi));

            jacobian_pose_i = sqrt_info * jacobian_pose_i;

            if (jacobian_pose_i.maxCoeff() > 1e8 || jacobian_pose_i.minCoeff() < -1e8)
            {
                ROS_WARN("numerical unstable in preintegration");
                //std::cout << sqrt_info << std::endl;
                //ROS_BREAK();
            }
            *H1 = jacobian_pose_i;
        }
        if (H2)
        {
            Eigen::Matrix<double, 15, 9, Eigen::RowMajor> jacobian_speedbias_i;
            jacobian_speedbias_i.setZero();
            jacobian_speedbias_i.block<3, 3>(O_P, O_V - O_V) = -Qi.inverse().toRotationMatrix() * sum_dt;
            jacobian_speedbias_i.block<3, 3>(O_P, O_BA - O_V) = -dp_dba;
            jacobian_speedbias_i.block<3, 3>(O_P, O_BG - O_V) = -dp_dbg;

            //Eigen::Quaterniond corrected_delta_q = pre_integration->delta_q * Utility::deltaQ(dq_dbg * (Bgi - pre_integration->linearized_bg));
            //jacobian_speedbias_i.block<3, 3>(O_R, O_BG - O_V) = -Utility::Qleft(Qj.inverse() * Qi * corrected_delta_q).bottomRightCorner<3, 3>() * dq_dbg;
            jacobian_speedbias_i.block<3, 3>(O_R, O_BG - O_V) = -Utility::Qleft(Qj.inverse() * Qi * pre_integration->delta_q).bottomRightCorner<3, 3>() * dq_dbg;

            jacobian_speedbias_i.block<3, 3>(O_V, O_V - O_V) = -Qi.inverse().toRotationMatrix();
            jacobian_speedbias_i.block<3, 3>(O_V, O_BA - O_V) = -dv_dba;
            jacobian_speedbias_i.block<3, 3>(O_V, O_BG - O_V) = -dv_dbg;

            jacobian_speedbias_i.block<3, 3>(O_BA, O_BA - O_V) = -Eigen::Matrix3d::Identity();

            jacobian_speedbias_i.block<3, 3>(O_BG, O_BG - O_V) = -Eigen::Matrix3d::Identity();

            jacobian_speedbias_i = sqrt_info * jacobian_speedbias_i;

            //ROS_ASSERT(fabs(jacobian_speedbias_i.maxCoeff()) < 1e8);
            //ROS_ASSERT(fabs(jacobian_speedbias_i.minCoeff()) < 1e8);
            *H2 = jacobian_speedbias_i;
        }
        if (H3)
        {
            Eigen::Matrix<double, 15, 7, Eigen::RowMajor> jacobian_pose_j;
            jacobian_pose_j.setZero();

            jacobian_pose_j.block<3, 3>(O_P, O_P) = Qi.inverse().toRotationMatrix();

            Eigen::Quaterniond corrected_delta_q = pre_integration->delta_q * Utility::deltaQ(dq_dbg * (Bgi - pre_integration->linearized_bg));
            jacobian_pose_j.block<3, 3>(O_R, O_R) = Utility::Qleft(corrected_delta_q.inverse() * Qi.inverse() * Qj).bottomRightCorner<3, 3>();

            jacobian_pose_j = sqrt_info * jacobian_pose_j;

            //ROS_ASSERT(fabs(jacobian_pose_j.maxCoeff()) < 1e8);
            //ROS_ASSERT(fabs(jacobian_pose_j.minCoeff()) < 1e8);
            *H3 = jacobian_pose_j;
        }
        if (H4)
        {
            Eigen::Matrix<double, 15, 9, Eigen::RowMajor> jacobian_speedbias_j;
            jacobian_speedbias_j.setZero();

            jacobian_speedbias_j.block<3, 3>(O_V, O_V - O_V) = Qi.inverse().toRotationMatrix();

            jacobian_speedbias_j.block<3, 3>(O_BA, O_BA - O_V) = Eigen::Matrix3d::Identity();

            jacobian_speedbias_j.block<3, 3>(O_BG, O_BG - O_V) = Eigen::Matrix3d::Identity();

            jacobian_speedbias_j = sqrt_info * jacobian_speedbias_j;

            //ROS_ASSERT(fabs(jacobian_speedbias_j.maxCoeff()) < 1e8);
            //ROS_ASSERT(fabs(jacobian_speedbias_j.minCoeff()) < 1e8);
            *H4 = jacobian_speedbias_j;
        }

    }


    // if (H1) *H1 = (gtsam::Matrix23() << 1.0, 0.0, 0.0, 
    //                                   0.0, 1.0, 0.0).finished();
    
    // return error vector
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

