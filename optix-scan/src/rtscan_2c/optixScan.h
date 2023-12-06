#ifndef OPTIXSCAN_H
#define OPTIXSCAN_H

#ifndef DEBUG_ISHIT_CMP_RAY
#define DEBUG_ISHIT_CMP_RAY 0
#endif
#ifndef DEBUG_INFO
#define DEBUG_INFO 0 // info about ray
#endif

#define DEBUG_RAY 0
struct Params
{
    double2*                points2;
    double3*                points3;    
    int                     width;           // 光线的个数
    int                     height;          // 光线的个数
    int                     column_num;
    double                  aabb_width; 
    double*                 predicate;
    double                  ray_stride;      // 每个 depth，对应的光线长度          
    OptixTraversableHandle  handle;
    unsigned int*           result;          // 记录 scan 结果
    unsigned int*           ray_primitive_hits;  // 光线和图元相交个数
    double                  tmin;
    double                  tmax;
    unsigned int*           intersection_test_num;
    unsigned int*           hit_num;            // set result 次数
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
    enum PREDICATE {
        NONE,   // 未作判别
        LT,     // <
        LE,     // <=
        GT,     // >
        GE,     // >=
        EQ,     // ==
        BT      // var1 < var < var2
    };

    double x1, x2;
    double y1, y2;
    double z1, z2;

    PREDICATE PREDICATE_X;
    PREDICATE PREDICATE_Y;
    PREDICATE PREDICATE_Z;

    void print() {
        printf("predicate = {x1: %f, x2: %f, y1: %f, y2: %f, z1: %f, z2: %f}\n",
                x1, x2, y1, y2, z1, z2);
    }
};

typedef HitGroupData Predicate;

#endif