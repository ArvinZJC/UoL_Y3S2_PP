/*
Blelloch basic exclusive scan;
the following implementation has a limitation that the size of A must be a power of 2;
please note that this implementation only works within a single workgroup as it uses barriers
*/
kernel void scan_bl(global int* A)
{
	int id = get_global_id(0);
	int N = get_global_size(0);
	int t;

	// up-sweep
	for (int stride = 1; stride < N; stride *= 2)
	{
		if (((id + 1) % (stride * 2)) == 0)
			A[id] += A[id - stride];

		barrier(CLK_GLOBAL_MEM_FENCE); // sync the step
	} // end for

	// down-sweep
	if (id == 0)
		A[N - 1] = 0; // exclusive scan

	barrier(CLK_GLOBAL_MEM_FENCE); // sync the step

	for (int stride = N / 2; stride > 0; stride /= 2)
	{
		if (((id + 1) % (stride * 2)) == 0)
		{
			t = A[id];
			A[id] += A[id - stride]; // reduce
			A[id - stride] = t; // move
		} // end if

		barrier(CLK_GLOBAL_MEM_FENCE); // sync the step
	} // end for
} // end function scan_bl