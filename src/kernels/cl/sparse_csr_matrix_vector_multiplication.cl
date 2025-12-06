#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
sparse_csr_matrix_vector_multiplication(
    __global const uint* offsets,
    __global const uint* columns,
    __global const uint* values,
    __global const uint* vector_values,
    __global uint* output,
    uint nrows,
    uint ncols,
    uint nnz)
{
    const uint group_index = get_group_id(0);
    const uint local_index = get_local_id(0);

    __local uint local_mem[GROUP_SIZE];

    if (group_index >= nrows) {
        local_mem[local_index] = 0;
    } else {
        uint offset = offsets[group_index];
        uint next_offset = offsets[group_index + 1];

        if (local_index + offset >= next_offset) {
            local_mem[local_index] = 0;
        } else {
            local_mem[local_index] = values[local_index + offset] * vector_values[columns[local_index + offset]];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        if (local_index == 0) {
            int sum = 0;
            for (uint j = 0; j < GROUP_SIZE; j++) {
                sum += local_mem[j];
            }
            atomic_add(&output[group_index], sum);
        }
    }
}
