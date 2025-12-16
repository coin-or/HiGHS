# !!! note
#     This file is a version of the file we use to package HiGHS for the Julia
#     ecosystem. If you make changes to this file during the development of
#     HiGHS, please tag `@odow` so we can make the correponding changes to:
#     https://github.com/JuliaPackaging/Yggdrasil/blob/master/H/HiGHS

using BinaryBuilder, Pkg

name = "HiGHS"
version = VersionNumber(ENV["HIGHS_RELEASE"])

sources = [GitSource(ENV["HIGHS_URL"], ENV["HIGHS_COMMIT"])
    # ArchiveSource("https://github.com/xianyi/OpenBLAS/releases/download/v0.3.21/OpenBLAS-0.3.21.tar.gz",
    #               "f36ba3d7a60e7c8bcc54cd9aaa9b1223dd42eaf02c811791c37e8ca707c241ca")
    ArchiveSource(
        "https://github.com/OpenMathLib/OpenBLAS/releases/download/v0.3.30/OpenBLAS-0.3.30.tar.gz",
                   "27342cff518646afb4c2b976d809102e368957974c250a25ccc965e53063c95d")
    ]

script = raw"""
export BUILD_SHARED="OFF"
export BUILD_STATIC="ON"

cd $WORKSPACE/srcdir/HiGHS

# Remove system CMake to use the jll version
apk del cmake

mkdir -p build
cd build

# Do fully static build only on Windows
if [[ "${BUILD_SHARED}" == "OFF" ]] && [[ "${target}" == *-mingw* ]]; then
    export CXXFLAGS="-static"
fi

if [[ "${target}" == *-darwin-* ]]; then
cmake -DCMAKE_INSTALL_PREFIX=${prefix} \
    -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TARGET_TOOLCHAIN} \
    -DCMAKE_BUILD_TYPE=Release \
    -DZLIB_USE_STATIC_LIBS=${BUILD_STATIC} \
    -DHIPO=ON \
    -DBUILD_STATIC_EXE=ON \
    ..
else
cd $WORKSPACE/srcdir/OpenBLAS*
make DYNAMIC_ARCH=1 NO_SHARED=1 USE_OPENMP=0 NUM_THREADS=64 BINARY=64 -j${nproc}
make install PREFIX=${prefix} NO_SHARED=1

cd $WORKSPACE/srcdir/HiGHS
apk del cmake
mkdir -p build
cd build

cmake -DCMAKE_INSTALL_PREFIX=${prefix} \
    -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TARGET_TOOLCHAIN} \
    -DCMAKE_BUILD_TYPE=Release \
    -DZLIB_USE_STATIC_LIBS=${BUILD_STATIC} \
    -DHIPO=ON \
    -DBUILD_STATIC_EXE=ON \
    -DBLAS_LIBRARIES="${prefix}/lib/libopenblas.a" \
    ..
fi

if [[ "${target}" == *-linux-* ]]; then
        make -j ${nproc}
else
    if [[ "${target}" == *-mingw* ]]; then
        cmake --build . --config Release
    else
        cmake --build . --config Release --parallel
    fi
fi
make install

install_license ../LICENSE.txt
"""

products = [
    LibraryProduct("libhighs", :libhighs),
    ExecutableProduct("highs", :highs),
]

platforms = supported_platforms()
platforms = expand_cxxstring_abis(platforms)

dependencies = [
    Dependency("CompilerSupportLibraries_jll"),
    Dependency("Zlib_jll"),
    HostBuildDependency(PackageSpec(; name="CMake_jll")),
]

build_tarballs(
    ARGS,
    name,
    version,
    sources,
    script,
    platforms,
    products,
    dependencies;
    preferred_gcc_version = v"6",
    julia_compat = "1.6",
)
