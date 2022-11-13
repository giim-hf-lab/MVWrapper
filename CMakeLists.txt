include(cmake/FindMVS.cmake)

add_library(mvscpp STATIC mvs.ixx)
target_compile_definitions(mvscpp
	PUBLIC
		MSGPACK_USE_DEFINE_MAP
)
target_compile_features(mvscpp PUBLIC cxx_std_20)
target_compile_options(mvscpp
	PRIVATE
		"/interface"
)
target_include_directories(mvscpp SYSTEM PUBLIC ${OPENCV_INCLUDE_DIRS})
target_link_libraries(mvscpp
	PUBLIC
		fmt::fmt
		msgpackc-cxx
		opencv_core
	PRIVATE
		mvs::mvs
		mvs::vm
)
add_library(mvs::cpp ALIAS mvscpp)
