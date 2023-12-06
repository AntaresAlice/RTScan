#ifndef OPTIXSCAN_H
#define OPTIXSCAN_H

#ifndef DEBUG_ISHIT_CMP_RAY
#define DEBUG_ISHIT_CMP_RAY 0
#endif

#ifndef DEBUG_INFO
#define DEBUG_INFO 0 // info about ray
#endif

#ifndef PRIMITIVE_TYPE
#define PRIMITIVE_TYPE 2
#endif

struct Params
{
    double3*                points;
    float3*                 d_triangles;
    CUdeviceptr             d_sphere;
    OptixAabb*              d_aabb;

    int                     width;              // number of ray
    int                     height;             // number of ray
    int                     direction;          // direction = (x = 0, y = 1, z = 2)
    double                  aabb_width;
    double                  ray_interval;
    double                  ray_space;
    double                  ray_length;         // length of each ray
    double                  ray_last_length;    // length of the last ray
    double                  ray_stride;         // ray_stride = ray_length + ray_space          
    double*                 predicate;
    OptixTraversableHandle  handle;
    unsigned int*           result;             // record scan result
    unsigned int*           ray_primitive_hits; // intersection times of ray and primitive
    double                  tmin;
    double                  tmax;
    unsigned int*           intersection_test_num;
    unsigned int*           hit_num;            // number of setting result
    bool                    inverse;
};


struct RayGenData
{
    // No data needed
};


struct MissData
{
};


struct HitGroupData
{
    double x1, x2;
    double y1, y2;
    double z1, z2;

    void print() {
        printf("predicate = {x1: %f, x2: %f, y1: %f, y2: %f, z1: %f, z2: %f}\n",
                x1, x2, y1, y2, z1, z2);
    }
};

typedef HitGroupData Predicate;

#endif