#ifndef WIND_FIELD_MODEL_H_
#define WIND_FILED_MODEL_H_

#include <Eigen/Dense>


class WindFieldModel
{
private:

    // constants
    const double R = 6371.0 * 1e3;
    const double EDDY_VISCOCITY = 75.0;
    const double AIR_DENSITY = 1.1;
    const double PI = 3.14159265358979323846;
    const double EPS = std::numeric_limits<double>::epsilon();

    // properties
    // param file
    // param(0): latitude of the landfall (only for Monte-Carlo)
    // param(1): longtitude of the landfall (only for Monte-Carlo)
    // param(2): azimuth angle at landfall (only for Monte-Carlo)
    // param(3): central pressure different in hPa
    // param(4): forward translation speed in km/h
    // param(5): radius of the maximum wind in km
    Eigen::ArrayXd param;

    // delta_p file
    // delta_p(0): inner radius of the meshed storm cycle
    // delta_p(1): division size along radius
    // delta_p(2): outer radius of the meshed storm cycle
    // delta_p(3): starting angle of the meshed storm cycle
    // delta_p(4): angle interval
    // delta_p(5): ending angle of the meshed storm cycle
    // delta_p(6): wind speed evaluation height (bottom)
    // delta_p(7): height interval
    // delta_p(8): wind speed evaluation height (top)
    Eigen::ArrayXd delta_p;

    // del_par: deviation from main track
    // uses Zero(3) as default except for Monte-Carlo simulations
    Eigen::ArrayXd del_par;

    // terrain z0
    Eigen::MatrixXd Lat_wr;
    Eigen::MatrixXd Long_wr;
    Eigen::ArrayXd z0r;
    Eigen::ArrayXd Wr_sizes;
    int num_region;

    // stations
    Eigen::ArrayXd Lat_wout;
    Eigen::ArrayXd Long_wout;
    int num_station;

    // track
    Eigen::ArrayXd Lat_track;
    Eigen::ArrayXd Long_track;
    Eigen::ArrayXd Lat_w;
    Eigen::ArrayXd Long_w;

    // dP, dV, dR
    double dP;
    double dV;
    double dR;

    // methods
    double wrapTo360(double angle);
    int inpolygon(Eigen::ArrayXd PolyX, Eigen::ArrayXd PolyY, int N, double px, double py);
    int min(int a, int b);

public:

    // configuring the simulation
    int ConfigSimu(std::string config_file, std::string stn_file, 
        std::string trk_file, std::string latw_file);

    // defining the perturbation
    int PertubPath(std::string dpath_file);

    // defining reference surface roughness z0
    int DefineTern(std::string refz0_file);

    // interpolating z0 for given locations
    int ComputeStationZ0(std::string dirOutput);

    // simulating wind field
    int SimulateWind(std::string dirOutput);

};

#endif
