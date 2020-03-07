// a simple 2D kernel (Section 3.2 in Tutorial 2)
kernel void add2D(global const int* A, global const int* B, global int* C)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int width = get_global_size(0);
	int height = get_global_size(1);
	int id = x + y * width;

	// printf("id = %d x = %d y = %d w = %d h = %d\n", id, x, y, width, height); // help to better understand the function

	C[id]= A[id]+ B[id];
} // end function add2D