if (DEFINED MVS_VM_VERSION_MAJOR)
	if (NOT DEFINED MVS_VM_MINOR OR NOT DEFINED MVS_VM_VERSION_PATCH)
		message(FATAL_ERROR "Incomplete MVS VM version specified.")
	endif ()
else ()
	set(MVS_VM_VERSION_MAJOR 4)
	set(MVS_VM_VERSION_MINOR 2)
	set(MVS_VM_VERSION_PATCH 0)
endif ()

set(MVS_VM_DEVELOPMENT_DIR "$ENV{PROGRAMFILES}\\VisionMaster${MVS_VM_VERSION_MAJOR}.${MVS_VM_VERSION_MINOR}.${MVS_VM_VERSION_PATCH}\\Development\\V${MVS_VM_VERSION_MAJOR}.x")

if (NOT IS_DIRECTORY ${MVS_VM_DEVELOPMENT_DIR})
	message(FATAL_ERROR "${MVS_VM_DEVELOPMENT_DIR} is not a directory.")
endif ()

if (WIN32)
	if (DEFINED ENV{PROGRAMFILES\(x86\)})
		set(MVS_DEVELOPMENT_DIR "$ENV{PROGRAMFILES\(x86\)}\\MVS\\Development")
		set(MVS_LIBRARIES_DIR "${MVS_DEVELOPMENT_DIR}\\Libraries\\win64")

		set(MVS_VM_LIBRARIES_DIR "${MVS_VM_DEVELOPMENT_DIR}\\Libraries\\win64")
	elseif (DEFINED ENV{PROGRAMFILES})
		set(MVS_DEVELOPMENT_DIR "$ENV{PROGRAMFILES}\\MVS\\Development")
		set(MVS_LIBRARIES_DIR "${MVS_DEVELOPMENT_DIR}\\Libraries\\win32")

		set(MVS_VM_LIBRARIES_DIR "${MVS_VM_DEVELOPMENT_DIR}\\Libraries\\win32")
	else ()
		message(FATAL_ERROR "MVS not found in the default location.")
	endif ()

	if (NOT IS_DIRECTORY ${MVS_DEVELOPMENT_DIR})
		message(FATAL_ERROR "${MVS_DEVELOPMENT_DIR} is not a directory.")
	endif ()

	set(MVS_INCLUDE_DIR "${MVS_DEVELOPMENT_DIR}\\Includes")
	set(MVS_LIBRARY "${MVS_LIBRARIES_DIR}\\MvCameraControl.lib")

	if (NOT IS_DIRECTORY ${MVS_INCLUDE_DIR} OR NOT EXISTS ${MVS_LIBRARY})
		message(FATAL_ERROR "${MVS_DEVELOPMENT_DIR} does not contain development files.")
	endif ()

	set(MVS_VM_INCLUDE_DIR "${MVS_VM_DEVELOPMENT_DIR}\\Includes")
	set(MVS_VM_LIBRARY "${MVS_VM_LIBRARIES_DIR}\\C\\iMVS-6000PlatformSDK.lib")

	if (NOT IS_DIRECTORY ${MVS_VM_INCLUDE_DIR} OR NOT EXISTS ${MVS_VM_LIBRARY})
		message(FATAL_ERROR "${MVS_VM_DEVELOPMENT_DIR} does not contain development files.")
	endif ()
else ()
	message(FATAL_ERROR "MVS not supported on ${CMAKE_HOST_SYSTEM_NAME}.")
endif ()

add_library(mvs STATIC IMPORTED)
set_target_properties(mvs PROPERTIES
	INCLUDE_DIRECTORIES ${MVS_INCLUDE_DIR}
	IMPORTED_LOCATION ${MVS_LIBRARY}
)
target_include_directories(mvs SYSTEM INTERFACE ${MVS_INCLUDE_DIR})
target_link_libraries(mvs INTERFACE ${MVS_LIBRARY})
add_library(mvs::mvs ALIAS mvs)

add_library(mvsvm STATIC IMPORTED)
set_target_properties(mvsvm PROPERTIES
	INCLUDE_DIRECTORIES ${MVS_VM_INCLUDE_DIR}
	IMPORTED_LOCATION ${MVS_VM_LIBRARY}
)
target_include_directories(mvsvm SYSTEM INTERFACE ${MVS_VM_INCLUDE_DIR})
target_link_libraries(mvsvm INTERFACE ${MVS_VM_LIBRARY})
add_library(mvs::vm ALIAS mvsvm)
