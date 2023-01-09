if (WIN32)
	if (DEFINED ENV{PROGRAMFILES\(x86\)})
		set(MVS_DEVELOPMENT_DIR "$ENV{PROGRAMFILES\(x86\)}/MVS/Development")
		set(MVS_LIBRARIES_DIR "${MVS_DEVELOPMENT_DIR}/Libraries/win64")
	elseif (DEFINED ENV{PROGRAMFILES})
		set(MVS_DEVELOPMENT_DIR "$ENV{PROGRAMFILES}/MVS/Development")
		set(MVS_LIBRARIES_DIR "${MVS_DEVELOPMENT_DIR}/Libraries/win32")
	else ()
		message(FATAL_ERROR "Unknown environment detected.")
	endif ()

	set(MVS_INCLUDE_DIR "${MVS_DEVELOPMENT_DIR}/Includes")
	set(MVS_LIBRARY "${MVS_LIBRARIES_DIR}/MvCameraControl.lib")
else ()
	message(FATAL_ERROR "Wrappers not supported on ${CMAKE_HOST_SYSTEM_NAME}.")
endif ()

add_library(mvs_api STATIC IMPORTED)
set_target_properties(mvs_api PROPERTIES
	IMPORTED_LOCATION "${MVS_LIBRARY}"
	INTERFACE_INCLUDE_DIRECTORIES
		"${MVS_INCLUDE_DIR}"
	SYSTEM TRUE
)

if (MSVC)
	create_msvc_cxx_module_target(
		TARGET mvs
		COMPILE_DEFINITIONS
			MSGPACK_USE_DEFINE_MAP
		SYSTEM_INCLUDE_DIRECTORIES
			${OPENCV_INCLUDE_DIRS}
		LINK_LIBRARIES
			opencv_core
			opencv_imgproc

			mvs_api
	)
	add_library(mvs::mvs ALIAS mvs)
else ()
	message(FATAL_ERROR "MVS wrapper requires C++20 module support")
endif ()

add_executable(mvs_driver EXCLUDE_FROM_ALL ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
target_link_libraries(mvs_driver
	PRIVATE
		mvs::mvs
)
