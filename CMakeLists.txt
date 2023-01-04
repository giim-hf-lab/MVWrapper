if (WIN32)
	if (DEFINED ENV{PROGRAMFILES\(x86\)})
		set(MVS_DEVELOPMENT_DIR "$ENV{PROGRAMFILES\(x86\)}/MVS/Development")
		set(MVS_LIBRARIES_DIR "${MVS_DEVELOPMENT_DIR}/Libraries/win64")
	elseif (DEFINED ENV{PROGRAMFILES})
		set(MVS_DEVELOPMENT_DIR "$ENV{PROGRAMFILES}/MVS/Development")
		set(MVS_LIBRARIES_DIR "${MVS_DEVELOPMENT_DIR}/Libraries/win32")
	else ()
		message(FATAL_ERROR "MV not found in the default location.")
	endif ()

	set(MVS_INCLUDE_DIR "${MVS_DEVELOPMENT_DIR}/Includes")
	file(GLOB MVS_LIBRARIES "${MVS_LIBRARIES_DIR}/*.lib")
else ()
	message(FATAL_ERROR "MV not supported on ${CMAKE_HOST_SYSTEM_NAME}.")
endif ()

add_library(mvs INTERFACE)
target_include_directories(mvs SYSTEM
	INTERFACE
		${MVS_INCLUDE_DIR}
)
target_link_libraries(mvs
	INTERFACE
		${MVS_LIBRARIES}
)
add_library(mv::sdk ALIAS mvs)

add_library(mvcpp STATIC mv.ixx)
target_compile_definitions(mvcpp
	PUBLIC
		MSGPACK_USE_DEFINE_MAP
)
target_compile_features(mvcpp PUBLIC cxx_std_20)
target_compile_options(mvcpp
	PRIVATE
		"/interface"
)
target_include_directories(mvcpp SYSTEM
	PUBLIC
		${OPENCV_INCLUDE_DIRS}
)
target_link_libraries(mvcpp
	PUBLIC
		opencv_core
		opencv_imgproc
	PRIVATE
		mv::sdk
)
add_library(mv::cpp ALIAS mvcpp)
