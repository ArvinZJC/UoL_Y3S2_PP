// a simple smoothing kernel averaging values in a local window of size 3/radius 1 (Section 3.1 in Tutorial 2)
kernel void avg_filter(global const int* A, global int* B)
{
	int id = get_global_id(0);
	int size = get_global_size(0);
	int id_new = id;

	// one way to handle the boundary conditions
	if (id == 0)
		id_new = 1;
	
	if (id == size - 1)
		id_new = size - 2;

	B[id] = (A[id_new - 1] + A[id_new] + A[id_new + 1]) / 3;
} // end function avg_filter