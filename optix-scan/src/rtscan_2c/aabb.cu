#include <optix.h>
#include <sutil/vec_math.h>
#include "optixScan.h"

__global__ void kGenAABB_t (void* raw_points, double radius, unsigned int N, 
                            OptixAabb* aabb,int column_num) {
    unsigned int particleIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (particleIndex >= N)
        return;

    if (column_num == 3) {
        double3* points = (double3*)raw_points;
        float3 center = {points[particleIndex].x, points[particleIndex].y, points[particleIndex].z};
        float3 m_min = center - radius;
        float3 m_max = center + radius;
        aabb[particleIndex] =
        {
            m_min.x, m_min.y, m_min.z,
            m_max.x, m_max.y, m_max.z
        };
    } else if (column_num == 2) {
        double2* points = (double2*)raw_points;
        float2 center = {points[particleIndex].x, points[particleIndex].y};
        float2 m_min = center - radius;
        float2 m_max = center + radius;
        aabb[particleIndex] =
        {
            1.0f, m_min.x, m_min.y,
            1.0f + float(2 * radius), m_max.x, m_max.y
        };
    }
}

extern "C" void kGenAABB(void* raw_points, double width, unsigned int numPrims, 
                         OptixAabb* d_aabb, int column_num) {
    unsigned int threadsPerBlock = 64;
    unsigned int numOfBlocks = numPrims / threadsPerBlock + 1;

    kGenAABB_t<<<numOfBlocks, threadsPerBlock>>>(
        raw_points,
        width,
        numPrims,
        d_aabb,
        column_num
    );
}