// a simple OpenCL kernel which adds two vectors A and B together into a third vector C
kernel void add(global const int* A, global const int* B, global int* C)
{
	int id = get_global_id(0);
	C[id] = A[id] + B[id];
} // end function add

// a simple OpenCL kernel which multiply two vectors A and B together into a third vector C
kernel void mult(global const int* A, global const int* B, global int* C)
{
	int id = get_global_id(0);
	C[id] = A[id] * B[id];
} // end function mult

// a simple OpenCL kernel which multiply two vectors A and B and then add to B to get a third vector C
kernel void multadd(global const int* A, global const int* B, global int* C)
{
	int id = get_global_id(0);
	C[id] = A[id] * B[id] + B[id];
} // end function multadd