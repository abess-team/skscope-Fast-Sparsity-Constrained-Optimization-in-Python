#include <random>
#include <iostream>

#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>

#include <Eigen/Eigen>

using Eigen::MatrixXd;
using Eigen::MatrixXi;
using Eigen::VectorXd;
using Eigen::VectorXi;
using Eigen::ArrayXd;
using Eigen::Matrix;
namespace py = pybind11;

struct RegressionData {
    MatrixXd x;
    VectorXd y;
    MatrixXd y_multi;
    RegressionData(MatrixXd X, MatrixXd Y) : x(X), y_multi(Y){
        this->y = Y.col(0);
        if (X.rows() != Y.rows()) {
            throw std::invalid_argument("x and y must have the same number of rows");
        }
    }
};

//**********************************************************************************
// linear model 
//**********************************************************************************
double linear_loss(VectorXd const& para, py::object const& ex_data) {
    RegressionData* data = ex_data.cast<RegressionData*>();  // unwrap the pointer
    return (data->x * para - data->y).squaredNorm();  // compute the loss
}
VectorXd linear_gradient(VectorXd const& para, py::object const& ex_data) {
    RegressionData* data = ex_data.cast<RegressionData*>();  // unwrap the pointer
    VectorXd result = 2 * data->x.transpose() * (data->x * para - data->y);
    return result;  // compute the gradient
}
//**********************************************************************************
// multi output linear model 
//**********************************************************************************
double multitask_loss(VectorXd const& para, py::object const& ex_data) {
    RegressionData* data = ex_data.cast<RegressionData*>();  // unwrap the pointer
    // reshape para
    int m = data->y_multi.cols();
    int p = data->x.cols();
    MatrixXd para_mat = MatrixXd::Zero(p, m);
    for(int i = 0; i < p; i++){
        para_mat.row(i) = para.segment(i*m, m);
    }
    return (data->x * para_mat - data->y_multi).squaredNorm();  // compute the loss
}
VectorXd multitask_gradient(VectorXd const& para, py::object const& ex_data) {
    RegressionData* data = ex_data.cast<RegressionData*>();  // unwrap the pointer
    // reshape para
    int m = data->y_multi.cols();
    int p = data->x.cols();
    MatrixXd para_mat = MatrixXd::Zero(p, m);
    for(int i = 0; i < p; i++){
        para_mat.row(i) = para.segment(i*m, m);
    }
    MatrixXd result = 2 * data->x.transpose() * (data->x * para_mat - data->y_multi);
    // flatten result
    VectorXd result_vec = VectorXd::Zero(p*m);
    for(int i = 0; i < p; i++){
        result_vec.segment(i*m, m) = result.row(i);
    }
    return result_vec;  // compute the gradient
}

//**********************************************************************************
// positive linear model 
//**********************************************************************************
double positive_linear_loss(VectorXd const& para, py::object const& ex_data) {
    RegressionData* data = ex_data.cast<RegressionData*>();  // unwrap the pointer
    return (data->x * para.cwiseAbs() - data->y).squaredNorm();  // compute the loss
}
VectorXd positive_linear_gradient(VectorXd const& para, py::object const& ex_data) {
    RegressionData* data = ex_data.cast<RegressionData*>();  // unwrap the pointer
    VectorXd result = 2 * data->x.transpose() * (data->x * para.cwiseAbs() - data->y);
    VectorXd sig = para.cwiseSign();
    // set to 1 if sig is 0
    for(int i = 0; i < sig.size(); i++){
        if(sig(i) == 0){
            sig(i) = 1;
        }
    }
    return result.cwiseProduct(sig);  // compute the gradient
}


//**********************************************************************************
// logistic model
//**********************************************************************************
double logistic_loss(VectorXd const& para, py::object const& ex_data) {
    RegressionData* data = ex_data.cast<RegressionData*>();
    ArrayXd xbeta = (data->x * para).array();
    xbeta = xbeta.max(-30.0).min(30.0);
    return ((xbeta.exp()+1.0).log() - (data->y).array()*xbeta).sum();
}

VectorXd logistic_gradient(VectorXd const& para, py::object const& ex_data) {
    RegressionData* data = ex_data.cast<RegressionData*>();
    ArrayXd xbeta = (data->x * para).array();
    xbeta = xbeta.max(-30.0).min(30.0);
    ArrayXd xbeta_exp = xbeta.exp();
    return data->x.transpose() * (xbeta_exp / (xbeta_exp + 1.0) - (data->y).array()).matrix();
}

//**********************************************************************************
// Ising model data structure and generator
//**********************************************************************************
Eigen::MatrixXd comp_conf(int num_conf, int p){
  Eigen::MatrixXi conf = Eigen::MatrixXi::Zero(num_conf, p);
  Eigen::VectorXi num = Eigen::VectorXi::LinSpaced(num_conf, 0, num_conf - 1);

  for (int i = p - 1; i >= 0; i--){
    conf.col(i) = num - num / 2 * 2;
    num /= 2;
  }
  conf = conf.array() * 2 - 1;
  return conf.cast<double>();
}

// n is the number of samples
Eigen::MatrixXd sample_by_conf(int sample_size, Eigen::MatrixXd theta, int seed) {
  int p = theta.rows();
  int num_conf = pow(2, p);
  
  Eigen::MatrixXd table = comp_conf(num_conf, p);
  Eigen::VectorXd weight(num_conf);
  
  Eigen::VectorXd vec_diag = theta.diagonal();
  Eigen::MatrixXd theta_diag = vec_diag.asDiagonal();
  Eigen::MatrixXd theta_off = theta - theta_diag;
  
  for (int num = 0; num < num_conf; num++) {
    Eigen::VectorXd conf = table.row(num);
    weight(num) = 0.5 * (double) (conf.transpose() * theta_off * conf);
  }
  weight = weight.array().exp();
  
  std::vector<double> w;
  w.resize(weight.size());
  Eigen::VectorXd::Map(w.data(), weight.size()) = weight;
  
  // int sd = (((long long int)time(0)) * 2718) % 314159265;
  // Rcout << "Seed: "<< sd << endl;
  // std::default_random_engine generator(seed);  // implementation-defined
  // std::default_random_engine generator(1);

  std::mt19937_64 generator;                      // 64-bit Mersenne Twister by Matsumoto and Nishimura, 2000
  generator.seed(seed);
  std::discrete_distribution<int> distribution(std::begin(w), std::end(w));
  
  Eigen::VectorXd freq = Eigen::VectorXd::Zero(num_conf);
  
  for (int i = 0; i < sample_size; i++) {
    freq(distribution(generator))++;
  }
  
  Eigen::MatrixXd data(num_conf, p + 1);
  data.col(0) = freq;
  data.rightCols(p) = table;
  return data;
}

struct IsingData{
    MatrixXd table;
    VectorXd freq;
    const int p;
    MatrixXi index_translator;
    IsingData(VectorXd freq, MatrixXd table): p(table.cols()) {
        this->freq = freq;
        this->table = table;
        index_translator.resize(p,p);
        int count = 0;
        for(Eigen::Index i = 0; i < p; i++){
            for(Eigen::Index j = i+1; j < p; j++){
                index_translator(i,j) = count;
                index_translator(j,i) = count;
                count++;
            }
        }
    }
};

//**********************************************************************************
// Ising model loss
//**********************************************************************************
double ising_loss(VectorXd const& para, py::object const& ex_data) {
    IsingData* data = ex_data.cast<IsingData*>();
    double loss = 0.0;

    for(int i = 0; i < data->table.rows(); i++){
        for(int k=0; k< data->p; k++){
            double tmp = 0.0;
            for(int j = 0; j < data->p; j++){
                if (j == k) continue;
                tmp += data->table(i,k) * data->table(i,j) * para(data->index_translator(k,j));
            }
            // avoid inf
            if (tmp < -30) 
                loss += data->freq(i) * (-2 * tmp);
            else if (tmp < 30)
                loss += data->freq(i) * log(1+exp(-2 * tmp));
        }
    }
    return loss;
}

VectorXd ising_grad(VectorXd const& para, py::object const& ex_data) {
    IsingData* data = ex_data.cast<IsingData*>();
    VectorXd grad_para = VectorXd::Zero(para.size());

    for(int i = 0; i < data->table.rows(); i++){
        for(int k=0; k< data->p; k++){
            double tmp = 0.0;
            for(int j = 0; j < data->p; j++){
                if (j == k) continue;
                tmp += data->table(i,k) * data->table(i,j) * para(data->index_translator(k,j));
            }
            double exp_tmp = 2 * data->freq(i) * data->table(i,k)  / (1+exp(2 * tmp));
            for(int j = 0; j < data->p; j++){
                if (j == k) continue;
                grad_para(data->index_translator(k,j)) -= exp_tmp * data->table(i,j);
            }
        }
    }

    return grad_para;
}

//**********************************************************************************
// trend filter
//**********************************************************************************
struct TimeSeriesData {
    VectorXd y;
    TimeSeriesData(VectorXd Y) : y(Y){}
};

double trend_filter_loss(VectorXd const& para, py::object const& ex_data) {
    TimeSeriesData* data = ex_data.cast<TimeSeriesData*>();
    VectorXd cumsum_para = VectorXd::Zero(para.size());
    std::partial_sum(para.data(), para.data() + para.size(), cumsum_para.data(), std::plus<double>());
    return (data->y - cumsum_para).squaredNorm() / para.size();
}
VectorXd trend_filter_grad(VectorXd const& para, py::object const& ex_data){
    TimeSeriesData* data = ex_data.cast<TimeSeriesData*>();
    VectorXd cumsum_para = VectorXd::Zero(para.size());
    std::partial_sum(para.data(), para.data() + para.size(), cumsum_para.data(), std::plus<double>());
    VectorXd grad = 2 * (cumsum_para - data->y);
    VectorXd grad_para = VectorXd::Zero(grad.size());
    for(int i = 0; i < grad.size(); i++) grad_para(i) = grad.tail(grad.size() - i).sum();
    return grad_para / para.size();
}


//**********************************************************************************
// pybind11 module
//**********************************************************************************
PYBIND11_MODULE(_skscope_experiment, m) {
    m.def("ising_generator", &sample_by_conf);
    pybind11::class_<RegressionData>(m, "RegressionData").def(py::init<MatrixXd, MatrixXd>());
    m.def("linear_loss", &linear_loss);
    m.def("linear_grad", &linear_gradient);
    m.def("logistic_loss", &logistic_loss);
    m.def("logistic_grad", &logistic_gradient);
    py::class_<IsingData>(m, "IsingData").def(py::init<VectorXd, MatrixXd>());
    m.def("ising_loss", &ising_loss);
    m.def("ising_grad", &ising_grad);
    m.def("multitask_loss", &multitask_loss);
    m.def("multitask_grad", &multitask_gradient);
    m.def("positive_loss", &positive_linear_loss);
    m.def("positive_grad", &positive_linear_gradient);
    py::class_<TimeSeriesData>(m, "TimeSeriesData").def(py::init<VectorXd>());
    m.def("trend_filter_loss", &trend_filter_loss);
    m.def("trend_filter_grad", &trend_filter_grad);
}
