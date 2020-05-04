#include "Utils.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <boost/compute.hpp>

namespace compute = boost::compute;
using namespace std;

int main()
{
	typedef int mytype;

	// get default device and setup context
	compute::device device = compute::system::default_device();
	compute::context context(device);
	compute::command_queue queue(context, device);

	cout << "Running on " << device.name() << endl;

	// create vectors on the host
	vector<mytype> A = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
	vector<mytype> B = { 0, 1, 2, 0, 1, 2, 0, 1, 2, 0 };
	vector<mytype> C(A.size());

	// create vectors on the device
	// compute::vector<mytype> devA(A.size());
	compute::vector<mytype> devA(A.size(), context);
	// compute::vector<mytype> devB(B.size());
	compute::vector<mytype> devB(B.size(), context);
	// compute::vector<mytype> devC(C.size());
	compute::vector<mytype> devC(C.size(), context);

	// copy input data to the device
	// compute::copy(A.begin(), A.end(), devA.begin());
	compute::copy(A.begin(), A.end(), devA.begin(), queue);
	// compute::copy(B.begin(), B.end(), devB.begin());
	compute::copy(B.begin(), B.end(), devB.begin(), queue);

	// perform C = A + B (use the map pattern)
	// compute::transform(devA.begin(), devA.end(), devB.begin(), devC.begin(), compute::plus<mytype>());
	compute::transform(devA.begin(), devA.end(), devB.begin(), devC.begin(), compute::plus<mytype>(), queue);

	// copy data back to the host
	// compute::copy(devC.begin(), devC.end(), C.begin());
	compute::copy(devC.begin(), devC.end(), C.begin(), queue);

	cout << "A = " << A << endl;
	cout << "B = " << B << endl;
	cout << "C = " << C << endl;

	return 0;
} // end main