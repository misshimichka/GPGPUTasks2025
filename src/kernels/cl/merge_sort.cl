#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "../defines.h"
#include "helpers/rassert.cl"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void
merge_sort(
    __global const uint* input_data,
    __global uint* output_data,
    int sorted_k,
    int n)
{
    const unsigned int i = get_global_id(0);
    if (i >= n) {
        return;
    }
    int l = -1, r = sorted_k;
    int adjust = 0;
    if ((i / sorted_k) % 2 == 0) {
        adjust = i / sorted_k * sorted_k + sorted_k;
    } else {
        adjust = i / sorted_k * sorted_k - sorted_k;
    }

    while (r - 1 > l) {
        int mid = (l + r) / 2;
        int mid_adjusted = mid + adjust;
        if (mid_adjusted >= n) {
            r = mid;
            continue;
        }
        if ((i / sorted_k) % 2 == 0) {
            // поток A
            if (input_data[i] > input_data[mid_adjusted]) {
                l = mid;
            } else {
                r = mid;
            }
        } else {
            // поток B
            if (input_data[i] >= input_data[mid_adjusted]) {
                l = mid;
            } else {
                r = mid;
            }
        }
    }
    output_data[i % sorted_k + r + i / sorted_k / 2 * sorted_k * 2] = input_data[i];
}
