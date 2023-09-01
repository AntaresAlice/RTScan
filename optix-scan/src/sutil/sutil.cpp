//
// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//


#include <sampleConfig.h>
#include <sutil/Exception.h>
#include <sutil/sutil.h>
#include <sutil/vec_math.h>

//#include <GLFW/glfw3.h>
//#include <glad/glad.h>
//#include <imgui/imgui.h>
//#include <imgui/imgui_impl_glfw.h>
//#include <imgui/imgui_impl_opengl3.h>
//#define STB_IMAGE_IMPLEMENTATION
//#include <tinygltf/stb_image.h>
//#define STB_IMAGE_WRITE_IMPLEMENTATION
//#include <tinygltf/stb_image_write.h>
//#define TINYEXR_IMPLEMENTATION
//#include <tinyexr/tinyexr.h>

#include <nvrtc.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <sstream>
#include <vector>
#if defined(_WIN32)
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN 1
#    endif
#    include<windows.h>
#    include<mmsystem.h>
#else
#    include<sys/time.h>
#    include <unistd.h>
#    include <dirent.h>
#endif


namespace sutil
{


static bool dirExists( const char* path )
{
#if defined( _WIN32 )
    DWORD attrib = GetFileAttributes( path );
    return ( attrib != INVALID_FILE_ATTRIBUTES ) && ( attrib & FILE_ATTRIBUTE_DIRECTORY );
#else
    DIR* dir = opendir( path );
    if( dir == NULL )
        return false;

    closedir( dir );
    return true;
#endif
}

static bool fileExists( const char* path )
{
    std::ifstream str( path );
    return static_cast<bool>( str );
}

static bool fileExists( const std::string& path )
{
    return fileExists( path.c_str() );
}

static std::string existingFilePath( const char* directory, const char* relativeSubDir, const char* relativePath )
{
    std::string path = directory ? directory : "";
    if( relativeSubDir )
    {
        path += '/';
        path += relativeSubDir;
    }
    if( relativePath )
    {
        path += '/';
        path += relativePath;
    }
    return fileExists( path ) ? path : "";
}

std::string getSampleDir()
{
    static const char* directories[] =
    {
        // TODO: Remove the environment variable OPTIX_EXP_SAMPLES_SDK_DIR once SDK 6/7 packages are split
        getenv( "OPTIX_EXP_SAMPLES_SDK_DIR" ),
        getenv( "OPTIX_SAMPLES_SDK_DIR" ),
        SAMPLES_DIR,
        "."
    };
    for( const char* directory : directories )
    {
        if( directory && dirExists( directory ) )
            return directory;
    }

    throw Exception( "sutil::getSampleDir couldn't locate an existing sample directory" );
}

const char* sampleFilePath( const char* relativeSubDir, const char* relativePath )
{
    static std::string s;

    // Allow for overrides.
    static const char* directories[] =
    {
        // TODO: Remove the environment variable OPTIX_EXP_SAMPLES_SDK_DIR once SDK 6/7 packages are split
        getenv( "OPTIX_EXP_SAMPLES_SDK_DIR" ),
        getenv( "OPTIX_SAMPLES_SDK_DIR" ),
        SAMPLES_DIR,
        "."
    };
    for( const char* directory : directories )
    {
        if( directory )
        {
            s = existingFilePath( directory, relativeSubDir, relativePath );
            if( !s.empty() )
            {
                return s.c_str();
            }
        }
    }
    throw Exception( ( std::string{ "sutil::sampleDataFilePath couldn't locate " } +relativePath ).c_str() );
}

const char* sampleDataFilePath( const char* relativePath )
{
    return sampleFilePath( "data", relativePath );
}

void parseDimensions( const char* arg, int& width, int& height )
{
    // look for an 'x': <width>x<height>
    size_t width_end    = strchr( arg, 'x' ) - arg;
    size_t height_begin = width_end + 1;

    if( height_begin < strlen( arg ) )
    {
        // find the beginning of the height string/
        const char* height_arg = &arg[height_begin];

        // copy width to null-terminated string
        char width_arg[32];
        strncpy( width_arg, arg, width_end );
        width_arg[width_end] = '\0';

        // terminate the width string
        width_arg[width_end] = '\0';

        width  = atoi( width_arg );
        height = atoi( height_arg );
        return;
    }
    const std::string err = "Failed to parse width, height from string '" + std::string( arg ) + "'";
    throw std::invalid_argument( err.c_str() );
}

double currentTime()
{
    return std::chrono::duration_cast< std::chrono::duration< double > >
        ( std::chrono::high_resolution_clock::now().time_since_epoch() ).count();
}


#define STRINGIFY( x ) STRINGIFY2( x )
#define STRINGIFY2( x ) #x
#define LINE_STR STRINGIFY( __LINE__ )

// Error check/report helper for users of the C API
#define NVRTC_CHECK_ERROR( func )                                                                                           \
    do                                                                                                                      \
    {                                                                                                                       \
        nvrtcResult code = func;                                                                                            \
        if( code != NVRTC_SUCCESS )                                                                                         \
            throw std::runtime_error( "ERROR: " __FILE__ "(" LINE_STR "): " + std::string( nvrtcGetErrorString( code ) ) ); \
    } while( 0 )

static bool readSourceFile( std::string& str, const std::string& filename )
{
    // Try to open file
    std::ifstream file( filename.c_str(), std::ios::binary );
    if( file.good() )
    {
        // Found usable source file
        std::vector<unsigned char> buffer = std::vector<unsigned char>( std::istreambuf_iterator<char>( file ), {} );
        str.assign(buffer.begin(), buffer.end());
        return true;
    }
    return false;
}

// Returns string of file extension including '.'
static std::string fileExtensionForLoading()
{
    std::string extension;
#if SAMPLES_INPUT_GENERATE_PTX
    extension = ".ptx";
#endif
#if SAMPLES_INPUT_GENERATE_OPTIXIR
    extension = ".optixir";
#endif
    if( const char* ext = getenv("OPTIX_SAMPLES_INPUT_EXTENSION") )
    {
        extension = ext;
        if( extension.size() && extension[0] != '.' )
            extension = "." + extension;
    }
    return extension;
}

#if CUDA_NVRTC_ENABLED

static void getCuStringFromFile( std::string& cu, std::string& location, const char* sampleDir, const char* filename )
{
    std::vector<std::string> source_locations;

    const std::string base_dir = getSampleDir();

    // Potential source locations (in priority order)
    if( sampleDir )
        source_locations.push_back( base_dir + '/' + sampleDir + '/' + filename );
    source_locations.push_back( base_dir + "/cuda/" + filename );

    for( const std::string& loc : source_locations )
    {
        // Try to get source code from file
        if( readSourceFile( cu, loc ) )
        {
            location = loc;
            return;
        }
    }

    // Wasn't able to find or open the requested file
    throw std::runtime_error( "Couldn't open source file " + std::string( filename ) );
}

static std::string g_nvrtcLog;

static void getPtxFromCuString( std::string&                    ptx,
                                const char*                     sample_directory,
                                const char*                     cu_source,
                                const char*                     name,
                                const char**                    log_string,
                                const std::vector<const char*>& compiler_options )
{
    // Create program
    nvrtcProgram prog = 0;
    NVRTC_CHECK_ERROR( nvrtcCreateProgram( &prog, cu_source, name, 0, NULL, NULL ) );

    // Gather NVRTC options
    std::vector<const char*> options;

    const std::string base_dir = getSampleDir();

    // Set sample dir as the primary include path
    std::string sample_dir;
    if( sample_directory )
    {
        sample_dir = std::string( "-I" ) + base_dir + '/' + sample_directory;
        options.push_back( sample_dir.c_str() );
    }

    // Collect include dirs
    std::vector<std::string> include_dirs;
    const char*              abs_dirs[] = {SAMPLES_ABSOLUTE_INCLUDE_DIRS};
    const char*              rel_dirs[] = {SAMPLES_RELATIVE_INCLUDE_DIRS};

    for( const char* dir : abs_dirs )
    {
        include_dirs.push_back( std::string( "-I" ) + dir );
    }
    for( const char* dir : rel_dirs )
    {
        include_dirs.push_back( "-I" + base_dir + '/' + dir );
    }
    for( const std::string& dir : include_dirs)
    {
        options.push_back( dir.c_str() );
    }

    // Collect NVRTC options
    std::copy( std::begin( compiler_options ), std::end( compiler_options ), std::back_inserter( options ) );

    // JIT compile CU to PTX
    const nvrtcResult compileRes = nvrtcCompileProgram( prog, (int)options.size(), options.data() );

    // Retrieve log output
    size_t log_size = 0;
    NVRTC_CHECK_ERROR( nvrtcGetProgramLogSize( prog, &log_size ) );
    g_nvrtcLog.resize( log_size );
    if( log_size > 1 )
    {
        NVRTC_CHECK_ERROR( nvrtcGetProgramLog( prog, &g_nvrtcLog[0] ) );
        if( log_string )
            *log_string = g_nvrtcLog.c_str();
    }
    if( compileRes != NVRTC_SUCCESS )
        throw std::runtime_error( "NVRTC Compilation failed.\n" + g_nvrtcLog );

    // Retrieve PTX code
    size_t ptx_size = 0;
    NVRTC_CHECK_ERROR( nvrtcGetPTXSize( prog, &ptx_size ) );
    ptx.resize( ptx_size );
    NVRTC_CHECK_ERROR( nvrtcGetPTX( prog, &ptx[0] ) );

    // Cleanup
    NVRTC_CHECK_ERROR( nvrtcDestroyProgram( &prog ) );
}

#else  // CUDA_NVRTC_ENABLED

static std::string sampleInputFilePath( const char* sampleName, const char* fileName )
{
    // Allow for overrides.
    static const char* directories[] =
    {
        // TODO: Remove the environment variable OPTIX_EXP_SAMPLES_SDK_PTX_DIR once SDK 6/7 packages are split
        getenv( "OPTIX_EXP_SAMPLES_SDK_PTX_DIR" ),
        getenv( "OPTIX_SAMPLES_SDK_PTX_DIR" ),
 #if defined(CMAKE_INTDIR)
        SAMPLES_PTX_DIR "/" CMAKE_INTDIR,
#endif
        SAMPLES_PTX_DIR,
        "."
    };

    // Allow overriding the file extension
    std::string extension = fileExtensionForLoading();

    if( !sampleName )
        sampleName = "sutil";
    for( const char* directory : directories )
    {
        if( directory )
        {
            std::string path = directory;
            path += '/';
            path += sampleName;
            path += "_generated_";
            path += fileName;
            path += extension;
            if( fileExists( path ) )
                return path;
        }
    }

    std::string error = "sutil::samplePTXFilePath couldn't locate ";
    error += fileName;
    error += " for sample ";
    error += sampleName;
    throw Exception( error.c_str() );
}

static void getInputDataFromFile( std::string& ptx, const char* sample_name, const char* filename )
{
    const std::string sourceFilePath = sampleInputFilePath( sample_name, filename );

    // Try to open source PTX file
    if( !readSourceFile( ptx, sourceFilePath ) )
    {
        std::string err = "Couldn't open source file " + sourceFilePath;
        throw std::runtime_error( err.c_str() );
    }
}

#endif  // CUDA_NVRTC_ENABLED

struct PtxSourceCache
{
    std::map<std::string, std::string*> map;
    ~PtxSourceCache()
    {
        for( std::map<std::string, std::string*>::const_iterator it = map.begin(); it != map.end(); ++it )
            delete it->second;
    }
};
static PtxSourceCache g_ptxSourceCache;

const char* getInputData( const char*                     sample,
                          const char*                     sampleDir,
                          const char*                     filename,
                          size_t&                         dataSize,
                          const char**                    log,
                          const std::vector<const char*>& compilerOptions )
{
    if( log )
        *log = NULL;

    std::string *                                 ptx, cu;
    std::string                                   key  = std::string( filename ) + ";" + ( sample ? sample : "" );
    std::map<std::string, std::string*>::iterator elem = g_ptxSourceCache.map.find( key );

    if( elem == g_ptxSourceCache.map.end() )
    {
        ptx = new std::string();
#if CUDA_NVRTC_ENABLED
        SUTIL_ASSERT( fileExtensionForLoading() == ".ptx" );
        std::string location;
        getCuStringFromFile( cu, location, sampleDir, filename );
        getPtxFromCuString( *ptx, sampleDir, cu.c_str(), location.c_str(), log, compilerOptions );
#else
        getInputDataFromFile( *ptx, sample, filename );
#endif
        g_ptxSourceCache.map[key] = ptx;
    }
    else
    {
        ptx = elem->second;
    }
    dataSize = ptx->size();
    return ptx->c_str();
}

void ensureMinimumSize( int& w, int& h )
{
    if( w <= 0 )
        w = 1;
    if( h <= 0 )
        h = 1;
}

void ensureMinimumSize( unsigned& w, unsigned& h )
{
    if( w == 0 )
        w = 1;
    if( h == 0 )
        h = 1;
}

void reportErrorMessage( const char* message )
{
    std::cerr << "OptiX Error: '" << message << "'\n";
#if defined( _WIN32 ) && defined( RELEASE_PUBLIC )
    {
        char s[2048];
        sprintf( s, "OptiX Error: %s", message );
        MessageBoxA( 0, s, "OptiX Error", MB_OK | MB_ICONWARNING | MB_SYSTEMMODAL );
    }
#endif
}

} // namespace sutil
