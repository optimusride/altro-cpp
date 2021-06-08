#include <Eigen/Dense>
#include <iostream>
#include <vector>

namespace altro {

namespace trajectory {
class Trajectory
{
  using Point = Eigen::VectorXd;
  using scalar_t = float; 
  template <class T>
  using traj_t = typename std::vector<T>;

 public:
  Trajectory(const traj_t<Point>& X, const traj_t<scalar_t>& h) : samples_{X}, steps_{h}, t_{h}
  {
    CalcIndependentVariable();
  }
  Trajectory(const traj_t<Point>& X, const scalar_t h) 
    : samples_{X}, steps_(X.size()-1, h), t_(X.size(), h) {
    CalcIndependentVariable();
  }

  traj_t<Point>& GetSamples() { return samples_; }
  traj_t<scalar_t>& GetSteps() { return steps_; }
  traj_t<scalar_t>& GetIndependentVars() { return t_; }

  const Point& GetSample(int k) const { return samples_[k]; }
  Point& GetSample(int k) { return samples_[k]; }
  scalar_t GetStep(int k) const { return steps_[k]; }
  scalar_t GetIndependentVar(int k) const { return t_[k]; }
  Point Interpolate(float t);
  size_t NumSegments() const { 
    assert(samples_.size() == t_.size());
    return samples_.size() - 1; 
  }
  
 private:
  void CalcIndependentVariable() {
    // Make sure the number of steps is one less than the number of points
    int N = samples_.size();
    if (steps_.size() == N) {
      steps_.pop_back();
    }
    if (t_.size() == N-1) {
      t_.push_back(0);
    }
    assert(steps_.size() == N-1);
    assert(t_.size() == N);

    // Cumulative sum to get independent variable from time steps
    t_[0] = 0;
    for (int k = 1; k < N; ++k) {
      t_[k] = t_[k-1] + steps_[k-1];
    }
  }

  traj_t<Point> samples_;
  traj_t<scalar_t> steps_;
  traj_t<scalar_t> t_;
};

} // namepsace trajectory
} // namespace altro