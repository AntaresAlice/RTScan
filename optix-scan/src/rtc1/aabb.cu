#include <optix.h>
#include <sutil/vec_math.h>
#include "optixScan.h"

__global__ void kGenAABB_t (
      double3* points,
      double radius,
      unsigned int N,
      OptixAabb* aabb
) {
  unsigned int particleIndex = blockIdx.x * blockDim.x + threadIdx.x;
  if (particleIndex >= N) return;

  float3 center = {points[particleIndex].x, points[particleIndex].y, points[particleIndex].z};
  float3 m_min = center - radius;
  float3 m_max = center + radius;
  aabb[particleIndex] =
  {
    m_min.x, m_min.y, m_min.z,
    m_max.x, m_max.y, m_max.z
  };
}

extern "C" void kGenAABB(double3* points, double width, unsigned int numPrims, OptixAabb* d_aabb) {
  unsigned int threadsPerBlock = 64;
  unsigned int numOfBlocks = numPrims / threadsPerBlock + 1;

  // 多线程设置所有的 aabb
  kGenAABB_t <<<numOfBlocks, threadsPerBlock>>> (
      points,
      width,
      numPrims,
      d_aabb
     );
}