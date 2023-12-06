
//#include "default_test_configuration.h"
#include "test_configuration_override.h"


#if MISS_PERCENTAGE != 0 && RANGE_QUERY_HIT_COUNT > 0
#error miss percentage does not work in range query mode
#endif

#if OUT_OF_RANGE_PERCENTAGE != 0 && RANGE_QUERY_HIT_COUNT > 0
#error out of range percentage does not work in range query mode
#endif

#if MISS_PERCENTAGE + OUT_OF_RANGE_PERCENTAGE > 100
#error cannot generate more than 100% combined misses
#endif

#if MISS_PERCENTAGE + OUT_OF_RANGE_PERCENTAGE > 0 && LEAVE_GAPS_FOR_MISSES == 0
#error cannot generate misses if there are no gaps
#endif

#if MISS_PERCENTAGE + OUT_OF_RANGE_PERCENTAGE == 0 && LEAVE_GAPS_FOR_MISSES != 0
#warning generating gaps without misses
#endif

#if PRIMITIVE == 1 && INT_TO_FLOAT_CONVERSION_MODE == 1
#error spheres do not work with unsafe key range
#endif

#if PRIMITIVE == 1 && INT_TO_FLOAT_CONVERSION_MODE == 2
#error spheres do not work with extended key range
#endif

#if PRIMITIVE == 2 && INT_TO_FLOAT_CONVERSION_MODE == 1
#error aabbs do not work with unsafe key range
#endif

#if START_RAY_AT_ZERO == 0 && INT_TO_FLOAT_CONVERSION_MODE == 2
#error ray has to be started at zero when using extended key range
#endif
