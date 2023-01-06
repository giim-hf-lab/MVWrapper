if (NOT GEMALTO_SUPERDOG_VERSION)
	set(GEMALTO_SUPERDOG_VERSION "2.5")
endif ()

if (NOT GEMALTO_SUPERDOG_BUILD)
	set(GEMALTO_SUPERDOG_BUILD "demo")
endif ()

if (WIN32)
	if (DEFINED ENV{PROGRAMFILES\(x86\)})
		set(MVS_DEVELOPMENT_DIR "$ENV{PROGRAMFILES\(x86\)}/MVS/Development")
		set(MVS_LIBRARIES_DIR "${MVS_DEVELOPMENT_DIR}/Libraries/win64")

		set(GEMALTO_SUPERDOG_LICENSING_API_DIR "$ENV{PROGRAMFILES\(x86\)}/Gemalto/SuperDog/${GEMALTO_SUPERDOG_VERSION}/API/Licensing/C/x64")
		set(GEMALTO_SUPERDOG_LICENSING_API_LIBRARY "${GEMALTO_SUPERDOG_LICENSING_API_DIR}/libdog_windows_x64_${GEMALTO_SUPERDOG_BUILD}.lib")
	elseif (DEFINED ENV{PROGRAMFILES})
		set(MVS_DEVELOPMENT_DIR "$ENV{PROGRAMFILES}/MVS/Development")
		set(MVS_LIBRARIES_DIR "${MVS_DEVELOPMENT_DIR}/Libraries/win32")

		set(GEMALTO_SUPERDOG_LICENSING_API_DIR "$ENV{PROGRAMFILES}/Gemalto/SuperDog/${GEMALTO_SUPERDOG_VERSION}/API/Licensing/C/win32")
		set(GEMALTO_SUPERDOG_LICENSING_API_LIBRARY "${GEMALTO_SUPERDOG_LICENSING_API_DIR}/libdog_windows_${GEMALTO_SUPERDOG_BUILD}.lib")
	else ()
		message(FATAL_ERROR "Unknown environment detected.")
	endif ()

	set(MVS_INCLUDE_DIR "${MVS_DEVELOPMENT_DIR}/Includes")
	set(MVS_LIBRARY "${MVS_LIBRARIES_DIR}/MvCameraControl.lib")
else ()
	message(FATAL_ERROR "Wrappers not supported on ${CMAKE_HOST_SYSTEM_NAME}.")
endif ()

add_library(mvs INTERFACE)
target_include_directories(mvs SYSTEM
	INTERFACE
		${MVS_INCLUDE_DIR}
)
target_link_libraries(mvs
	INTERFACE
		${MVS_LIBRARY}
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

		mv::sdk
)
add_library(mv::cpp ALIAS mvcpp)

add_library(gemalto_superdog INTERFACE)
target_include_directories(gemalto_superdog SYSTEM
	INTERFACE
		${GEMALTO_SUPERDOG_LICENSING_API_DIR}
)
target_link_libraries(gemalto_superdog
	INTERFACE
		${GEMALTO_SUPERDOG_LICENSING_API_LIBRARY}
)
add_library(gemalto::superdog ALIAS gemalto_superdog)

add_library(gemalto STATIC gemalto.ixx)
target_compile_features(gemalto PUBLIC cxx_std_20)
target_compile_options(gemalto
	PRIVATE
		"/interface"
)
target_link_libraries(gemalto
	PUBLIC
		fmt::fmt

		gemalto::superdog
)
add_library(gemalto::gemalto ALIAS gemalto)
