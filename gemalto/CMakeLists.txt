if (NOT GEMALTO_SUPERDOG_VERSION)
	set(GEMALTO_SUPERDOG_VERSION "2.5")
endif ()

if (NOT GEMALTO_SUPERDOG_BUILD)
	set(GEMALTO_SUPERDOG_BUILD "demo")
endif ()

if (NOT GEMALTO_SUPERDOG_VENDOR_CODE_LABEL)
	set(GEMALTO_SUPERDOG_VENDOR_CODE_LABEL "DEMOMA")
endif ()

if (WIN32)
	if (DEFINED ENV{PROGRAMFILES\(x86\)})
		set(GEMALTO_SUPERDOG_LICENSING_BASE "$ENV{PROGRAMFILES\(x86\)}/Gemalto/SuperDog/${GEMALTO_SUPERDOG_VERSION}")
		set(GEMALTO_SUPERDOG_LICENSING_API_DIR "${GEMALTO_SUPERDOG_LICENSING_BASE}/API/Licensing/C/x64")
		set(GEMALTO_SUPERDOG_LICENSING_API_LIBRARY "${GEMALTO_SUPERDOG_LICENSING_API_DIR}/dog_windows_x64_${GEMALTO_SUPERDOG_BUILD}.lib")
	elseif (DEFINED ENV{PROGRAMFILES})
		set(GEMALTO_SUPERDOG_LICENSING_BASE "$ENV{PROGRAMFILES}/Gemalto/SuperDog/${GEMALTO_SUPERDOG_VERSION}")
		set(GEMALTO_SUPERDOG_LICENSING_API_DIR "${GEMALTO_SUPERDOG_LICENSING_BASE}/API/Licensing/C/win32")
		set(GEMALTO_SUPERDOG_LICENSING_API_LIBRARY "${GEMALTO_SUPERDOG_LICENSING_API_DIR}/dog_windows_${GEMALTO_SUPERDOG_BUILD}.lib")
	else ()
		message(FATAL_ERROR "Unknown environment detected.")
	endif ()
else ()
	message(FATAL_ERROR "Gemalto wrapper not supported on ${CMAKE_HOST_SYSTEM_NAME}.")
endif ()

set(GEMALTO_VENDOR_CODE_DIR "${CMAKE_CURRENT_BINARY_DIR}/vendor_codes")
set(GEMALTO_SUPERDOR_VENDOR_CODE "${GEMALTO_VENDOR_CODE_DIR}/superdog.vc")

file(REMOVE_RECURSE "${GEMALTO_VENDOR_CODE_DIR}")
file(MAKE_DIRECTORY "${GEMALTO_VENDOR_CODE_DIR}")
add_custom_command(
	OUTPUT
		"${GEMALTO_SUPERDOR_VENDOR_CODE}"
	COMMAND
		"${Python_EXECUTABLE}"
		ARGS
			superdog.py
			-b "${GEMALTO_SUPERDOG_LICENSING_BASE}"
			-o "${GEMALTO_VENDOR_CODE_DIR}"
			-L "${GEMALTO_SUPERDOG_VENDOR_CODE_LABEL}"
	DEPENDS
		"${CMAKE_CURRENT_SOURCE_DIR}/superdog.py"
	WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
	COMMENT "Generating C++ snippet for Gemalto SuperDog vendor code"
	VERBATIM
	USES_TERMINAL
)
add_custom_target(gemalto_superdog_vendor_code
	DEPENDS
		"${GEMALTO_SUPERDOR_VENDOR_CODE}"
)

add_executable(gemalto_exe_superdog "${CMAKE_CURRENT_SOURCE_DIR}/src/exe/superdog.cpp")

if (MSVC)
	set(GEMALTO_CPPM "${CMAKE_CURRENT_SOURCE_DIR}/gemalto.cppm")
	set(GEMALTO_CPPM_OBJECT "${CMAKE_CURRENT_BINARY_DIR}/gemalto.obj")
	set(GEMALTO_IFC "${CMAKE_CURRENT_BINARY_DIR}/gemalto.ifc")

	file(REMOVE "${GEMALTO_CPPM_OBJECT}" "${GEMALTO_IFC}")

	add_custom_command(
		OUTPUT
			"${GEMALTO_IFC}"
			"${GEMALTO_CPPM_OBJECT}"
		COMMAND
			"${CMAKE_CXX_COMPILER}"
			ARGS
				/nologo
				/TP
				/external:W0
				/external:I "${GEMALTO_SUPERDOG_LICENSING_API_DIR}"
				/I "${CMAKE_CURRENT_BINARY_DIR}/vendor_codes"
				/I "${CMAKE_CURRENT_SOURCE_DIR}/include"
				${CMAKE_CXX_FLAGS}
				/std:c++20
				/c
				/interface "${GEMALTO_CPPM}"
		DEPENDS
			gemalto_superdog_vendor_code
			"${GEMALTO_CPPM}"
		WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
		COMMENT "Force compiling C++ module interface unit beforehand"
		VERBATIM
		USES_TERMINAL
		COMMAND_EXPAND_LISTS
	)
	add_custom_target(gemalto_module_build
		DEPENDS
			"${GEMALTO_IFC}"
			"${GEMALTO_CPPM_OBJECT}"
	)

	add_library(gemalto_module INTERFACE)
	add_dependencies(gemalto_module gemalto_module_build)
	target_compile_options(gemalto_module
		INTERFACE
			/reference "${GEMALTO_IFC}"
	)
	target_link_libraries(gemalto_module
		INTERFACE
			"${GEMALTO_SUPERDOG_LICENSING_API_LIBRARY}"
			"${GEMALTO_CPPM_OBJECT}"
	)
	add_library(gemalto::module ALIAS gemalto_module)

	target_compile_definitions(gemalto_exe_superdog
		PRIVATE
			__GEMALTO_USE_MODULE__
	)
	target_link_libraries(gemalto_exe_superdog
		PRIVATE
			gemalto::module
	)
else ()
	add_library(gemalto_superdog_api STATIC IMPORTED)
	set_target_properties(gemalto_superdog_api PROPERTIES
		IMPORTED_LOCATION "${GEMALTO_SUPERDOG_LICENSING_API_LIBRARY}"
		INTERFACE_INCLUDE_DIRECTORIES
			"${GEMALTO_SUPERDOG_LICENSING_API_DIR}"
	)

	add_library(gemalto STATIC EXCLUDE_FROM_ALL "${CMAKE_CURRENT_SOURCE_DIR}/src/lib/superdog.cpp")
	target_include_directories(gemalto
		PUBLIC
			"${CMAKE_CURRENT_SOURCE_DIR}/include"
		PRIVATE
			"${CMAKE_CURRENT_BINARY_DIR}/vendor_codes"
	)
	target_link_libraries(gemalto
		PUBLIC
			gemalto_superdog_api
	)
	add_dependencies(gemalto gemalto_superdog_vendor_code)
	add_library(gemalto::gemalto ALIAS gemalto)

	target_link_libraries(gemalto_exe_superdog
		PRIVATE
			gemalto::gemalto
	)
endif ()
