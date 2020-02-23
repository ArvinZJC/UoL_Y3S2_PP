// fixed 4 step reduce
kernel void reduce_add_1(global const int* A, global int* B)
{
	int id = get_global_id(0);
	int N = get_global_size(0);

	B[id] = A[id]; // copy input to output

	barrier(CLK_GLOBAL_MEM_FENCE); // wait for all threads to finish copying
	 
	/*
	 * perform reduce on the output array;
	 * modulo operator is used to skip a set of values (e.g. 2 in the next line);
	 * we also check if the added element is within bounds (i.e. < N)
	 */
	if (((id % 2) == 0) && ((id + 1) < N)) 
		B[id] += B[id + 1];

	barrier(CLK_GLOBAL_MEM_FENCE);

	if (((id % 4) == 0) && ((id + 2) < N)) 
		B[id] += B[id + 2];

	barrier(CLK_GLOBAL_MEM_FENCE);

	if (((id % 8) == 0) && ((id + 4) < N)) 
		B[id] += B[id + 4];

	barrier(CLK_GLOBAL_MEM_FENCE);

	if (((id % 16) == 0) && ((id + 8) < N)) 
		B[id] += B[id + 8];
} // end function reduce_add_1

// flexible step reduce 
kernel void reduce_add_2(global const int* A, global int* B)
{
	int id = get_global_id(0);
	int N = get_global_size(0);

	B[id] = A[id];

	barrier(CLK_GLOBAL_MEM_FENCE);

	// i is a stride
	for (int i = 1; i < N; i *= 2)
	{
		if (!(id % (i * 2)) && ((id + i) < N)) 
			B[id] += B[id + i];

		barrier(CLK_GLOBAL_MEM_FENCE);
	} // end for
} // end function reduce_add_2

// reduce using local memory (so-called privatisation)
kernel void reduce_add_3(global const int* A, global int* B, local int* scratch)
{
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	scratch[lid] = A[id]; // cache all N values from global memory to local memory

	barrier(CLK_LOCAL_MEM_FENCE); // wait for all local threads to finish copying from global to local memory

	for (int i = 1; i < N; i *= 2)
	{
		if (!(lid % (i * 2)) && ((lid + i) < N)) 
			scratch[lid] += scratch[lid + i];

		barrier(CLK_LOCAL_MEM_FENCE);
	} // end for

	B[id] = scratch[lid]; // copy the cache to output array
} // end function reduce_add_3

/*
 * reduce using local memory + accumulation of local sums into a single location;
 * it works with any number of groups - not optimal
 */
kernel void reduce_add_4(global const int* A, global int* B, local int* scratch)
{
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	scratch[lid] = A[id]; // cache all N values from global memory to local memory

	barrier(CLK_LOCAL_MEM_FENCE); // wait for all local threads to finish copying from global to local memory

	for (int i = 1; i < N; i *= 2)
	{
		if (!(lid % (i * 2)) && ((lid + i) < N)) 
			scratch[lid] += scratch[lid + i];

		barrier(CLK_LOCAL_MEM_FENCE);
	} // end for

	/*
	 * we add results from all local groups to the first element of the array;
	 * serial operation, but works for any group size;
	 * copy the cache to output array
	 */
	if (!lid)
		atomic_add(&B[0],scratch[lid]);
} // end function reduce_add_4

// a very simple histogram implementation
kernel void hist_simple(global const int* A, global int* H)
{ 
	int id = get_global_id(0);

	/*
	 * assume that H has been initialised to 0;
	 * take value as a bin index
	 */
	int bin_index = A[id];

	atomic_inc(&H[bin_index]); // serial operation, not very efficient
} // end function hist_simple

/*
 * Hillis-Steele basic inclusive scan;
 * require additional buffer B to avoid data overwrite
 */
kernel void scan_hs(global int* A, global int* B)
{
	int id = get_global_id(0);
	int N = get_global_size(0);
	global int* C;

	for (int stride = 1; stride < N; stride *= 2)
	{
		B[id] = A[id];

		if (id >= stride)
			B[id] += A[id - stride];

		barrier(CLK_GLOBAL_MEM_FENCE); // sync the step

		// swap A & B between steps
		C = A;
		A = B;
		B = C;
	} // end for
} // end function scan_hs

/*
 * a double-buffered version of the Hillis-Steele inclusive scan;
 * require 2 additional input arguments which correspond to 2 local buffers
 */
kernel void scan_add(__global const int* A, global int* B, local int* scratch_1, local int* scratch_2)
{
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);
	local int *scratch_3; // used for buffer swap

	scratch_1[lid] = A[id]; // cache all N values from global memory to local memory

	barrier(CLK_LOCAL_MEM_FENCE); // wait for all local threads to finish copying from global to local memory

	for (int i = 1; i < N; i *= 2
	{
		if (lid >= i)
			scratch_2[lid] = scratch_1[lid] + scratch_1[lid - i];
		else
			scratch_2[lid] = scratch_1[lid];

		barrier(CLK_LOCAL_MEM_FENCE);

		// buffer swap
		scratch_3 = scratch_2;
		scratch_2 = scratch_1;
		scratch_1 = scratch_3;
	} // end for

	B[id] = scratch_1[lid]; // copy the cache to output array
} // end function scan_add

// Blelloch basic exclusive scan
kernel void scan_bl(global int* A)
{
	int id = get_global_id(0);
	int N = get_global_size(0);
	int t;

	// up-sweep
	for (int stride = 1; stride < N; stride *= 2)
	{
		if (((id + 1) % (stride*2)) == 0)
			A[id] += A[id - stride];

		barrier(CLK_GLOBAL_MEM_FENCE); // sync the step
	} // end for

	//down-sweep
	if (id == 0)
		A[N-1] = 0;// exclusive scan

	barrier(CLK_GLOBAL_MEM_FENCE); // sync the step

	for (int stride = N/2; stride > 0; stride /= 2)
	{
		if (((id + 1) % (stride*2)) == 0)
		{
			t = A[id];
			A[id] += A[id - stride]; // reduce 
			A[id - stride] = t;		 // move
		} // end if

		barrier(CLK_GLOBAL_MEM_FENCE); // sync the step
	} // end for
} // end function scan_bl

// calculate the block sums
kernel void block_sum(global const int* A, global int* B, int local_size)
{
	int id = get_global_id(0);
	B[id] = A[(id+1)*local_size-1];
} // end function block_sum

// simple exclusive serial scan based on atomic operations - sufficient for small number of elements
kernel void scan_add_atomic(global int* A, global int* B)
{
	int id = get_global_id(0);
	int N = get_global_size(0);

	for (int i = id+1; i < N; i++)
		atomic_add(&B[i], A[id]);
} // end function scan_add_atomic

// adjust the values stored in partial scans by adding block sums to corresponding blocks
kernel void scan_add_adjust(global int* A, global const int* B)
{
	int id = get_global_id(0);
	int gid = get_group_id(0);
	A[id] += B[gid];
} // end function scan_add_adjust