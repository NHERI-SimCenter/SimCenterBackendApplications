#include <vector>

namespace Wind {
enum ExposureCategory { A, B, C, D };
class WindForces;

// When calling this function note that the output is 10 min simulation of the
// wind with dt = 0.1 (i.e. 6000 values) Note that inputs are in English units,
// and outputs are in SI units! we can change that, if needed
WindForces GetWindForces(ExposureCategory category, double gustWindSpeed,
                         double dragCoeff, double width,
                         std::vector<double> floorsHeights, double seed);

class WindForces {
public:
  WindForces(int numFloors, int size);
  ~WindForces();
  WindForces(const WindForces &other);
  std::vector<double> getFloorForces(int i);
  void setFloorForces(int i, const double *vector);
  double getTimeStep();
  void setTimeStep(double timeStep);

private:
  double **m_data;
  int m_size;
  int m_numFloors;
  double m_timeStep;
};
} // namespace Wind
