if (WIN32)
	if (DEFINED ENV{PROGRAMFILES\(x86\)})
		set(MVS_DEVELOPMENT_DIR "$ENV{PROGRAMFILES\(x86\)}/MVS/Development")
		set(MVS_LIBRARIES_DIR "${MVS_DEVELOPMENT_DIR}/Libraries/win64")

		set(MVD_DEVELOPMENT_DIR "$ENV{PROGRAMFILES\(x86\)}/MVDAlgorithmSDK")
		set(MVD_LIBRARIES_DIR "${MVD_DEVELOPMENT_DIR}/Libraries/x64")
	elseif (DEFINED ENV{PROGRAMFILES})
		set(MVS_DEVELOPMENT_DIR "$ENV{PROGRAMFILES}/MVS/Development")
		set(MVS_LIBRARIES_DIR "${MVS_DEVELOPMENT_DIR}/Libraries/win32")

		set(MVD_DEVELOPMENT_DIR "$ENV{PROGRAMFILES}/MVDAlgorithmSDK")
		set(MVD_LIBRARIES_DIR "${MVD_DEVELOPMENT_DIR}/Libraries/win32")
	else ()
		message(FATAL_ERROR "MV not found in the default location.")
	endif ()

	set(MVS_INCLUDE_DIR "${MVS_DEVELOPMENT_DIR}/Includes")
	file(GLOB MVS_LIBRARIES "${MVS_LIBRARIES_DIR}/*.lib")

	set(MVD_INCLUDE_DIR "${MVD_DEVELOPMENT_DIR}/Includes")
	file(GLOB MVD_LIBRARIES "${MVD_LIBRARIES_DIR}/Algorithms/*.lib" "${MVD_LIBRARIES_DIR}/Common/*.lib")
else ()
	message(FATAL_ERROR "MV not supported on ${CMAKE_HOST_SYSTEM_NAME}.")
endif ()

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
	PRIVATE
		${MVS_INCLUDE_DIR}
		"${MVD_INCLUDE_DIR}/Algorithms"
		"${MVD_INCLUDE_DIR}/Common/VisionDesigner"
)
target_link_libraries(mvcpp
	PUBLIC
		fmt::fmt
		msgpackc-cxx
		opencv_core
	PRIVATE
		${MVS_LIBRARIES}
		${MVD_LIBRARIES}
)
add_library(mv::cpp ALIAS mvcpp)
