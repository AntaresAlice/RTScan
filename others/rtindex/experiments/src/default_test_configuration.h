// rays should be cast perpendicularly to the line of triangles (point query only)
#define PERPENDICULAR_RAYS 0
// enable BHV compaction
#define COMPACTION 1
// set the OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL flag for BHV
#define FORCE_SINGLE_ANYHIT 1
// start rays at zero and limit the hit range using tmin/tmax (range query only)
#define START_RAY_AT_ZERO 1
// enable 64bit keys
#define LARGE_KEYS 0
// update the BVH after construction
#define PERFORM_UPDATES 0
// do not insert all keys of the build set, these keys can be probed later to simulate misses
#define LEAVE_GAPS_FOR_MISSES 0

// VALUES: 0 (triangle), 1 (sphere), 2 (aabb)
#define PRIMITIVE 0

// VALUES: -1 (descending), 0 (shuffle), 1 (ascending)
#define INSERT_SORTED 0
// VALUES: -1 (descending), 0 (shuffle), 1 (ascending), 2 (sort on gpu)
#define PROBE_SORTED 2

// a bias to add to the exponent during key conversion
// this is done to check whether scaling all keys impacts performance
#define EXPONENT_BIAS 0

#define NUM_BUILD_KEYS_LOG 25
#define NUM_PROBE_KEYS_LOG 27
#define RANGE_QUERY_HIT_COUNT_LOG 6
#define NUM_UPDATES_LOG 0
#define MISS_PERCENTAGE 0
#define OUT_OF_RANGE_PERCENTAGE 0
#define KEY_STRIDE_LOG 0

// VALUES: 0 (safe), 1 (unsafe), 2 (extended), 3 (3d)
#define INT_TO_FLOAT_CONVERSION_MODE 3
