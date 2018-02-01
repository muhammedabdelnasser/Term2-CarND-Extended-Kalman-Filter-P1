#include <iostream>
#include "tools.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
							  const vector<VectorXd> &ground_truth)
{
	VectorXd rmse(4);
	rmse << 0, 0, 0, 0;

	if (estimations.size() == 0)
	{
		cout << "Error - estimations size is 0" << endl;
		return rmse;
	}
	if (ground_truth.size() == 0)
	{
		cout << "Error - ground_truth size is 0" << endl;
		return rmse;
	}
	if (ground_truth.size() != estimations.size())
	{
		cout << "Error - ground_truth size and estimates size are not equal" << endl;
		return rmse;
	}

	//accumulate squared residuals
	for (int i = 0; i < estimations.size(); ++i)
	{
		VectorXd res = estimations[i] - ground_truth[i];

		res = res.array() * res.array();
		rmse += res;
	}

	//calculate the mean
	rmse = rmse.array() / estimations.size();

	//calculate the squared root
	rmse = rmse.array().sqrt();

	//return the result
	return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd &x_state)
{

	MatrixXd Hj(3, 4);
	//recover state parameters
	float px = x_state(0);
	float py = x_state(1);
	float vx = x_state(2);
	float vy = x_state(3);

	float dom = (px * px) + (py * py);
	float dom_root = sqrt(dom);
	float dom_3_2 = dom * dom_root;

	//check division by zero
	if (dom == 0 || dom_root == 0 || dom_3_2 == 0)
	{
		cout << "CalculateJacobian () - Error - Dividing by zero" << endl;
		return Hj;
	}

	//compute the Jacobian matrix
	Hj(0, 0) = px / dom_root;
	Hj(0, 1) = py / dom_root;
	Hj(0, 2) = 0;
	Hj(0, 3) = 0;
	//--------------------
	Hj(1, 0) = -py / dom;
	Hj(1, 1) = px / dom;
	Hj(1, 2) = 0;
	Hj(1, 3) = 0;
	//--------------------
	Hj(2, 0) = py * (vx * py - vy * px) / dom_3_2;
	Hj(2, 1) = px * (px * vy - py * vx) / dom_3_2;
	Hj(2, 2) = px / dom_root;
	Hj(2, 3) = py / dom_root;
	//--------------------

	return Hj;
}
