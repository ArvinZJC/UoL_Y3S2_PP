/*
 * a double-buffered version of the Hillis-Steele inclusive scan;
 * require 2 additional input arguments which correspond to 2 local buffers;
 * allow only for calculating partial reductions in a single work group separately
 */
kernel void scan_add(__global const int* A, global int* B, local int* scratch_1, local int* scratch_2)
{
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);
	local int *scratch_3; // used for buffer swap

	scratch_1[lid] = A[id]; // cache all N values from global memory to local memory

	barrier(CLK_LOCAL_MEM_FENCE); // wait for all local threads to finish copying from global to local memory

	// "i" represents the stride
	for (int i = 1; i < N; i *= 2)
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

// calculate the block sums
kernel void block_sum(global const int* A, global int* B, int local_size)
{
	int id = get_global_id(0);
	B[id] = A[(id + 1) * local_size - 1];
} // end function block_sum

// simple exclusive serial scan based on atomic operations - sufficient for small number of elements
kernel void scan_add_atomic(global int* A, global int* B)
{
	int id = get_global_id(0);
	int N = get_global_size(0);

	for (int i = id + 1; i < N && id < N; i++)
		atomic_add(&B[i], A[id]);
} // end function scan_add_atomic

// adjust the values stored in partial scans by adding block sums to corresponding blocks
kernel void scan_add_adjust(global int* A, global const int* B)
{
	int id = get_global_id(0);
	int gid = get_group_id(0);
	A[id] += B[gid];
} // end function scan_add_adjust