function (create_msvc_cxx_module_target TARGET)
	list(APPEND options)
	list(APPEND one_value_keywords
		PRIMARY_INTERFACE
		PRIMARY_INTERFACE_SUFFIX
	)
	list(APPEND multi_value_keywords
		COMPILE_DEFINITIONS
		COMPILE_OPTIONS
		INCLUDE_DIRECTORIES
		SYSTEM_INCLUDE_DIRECTORIES
		LINK_LIBRARIES
		SOURCES
	)
	cmake_parse_arguments(PARSE_ARGV 1 ARGS "${options}" "${one_value_keywords}" "${multi_value_keywords}")

	if (NOT ARGS_PRIMARY_INTERFACE)
		if (NOT ARGS_PRIMARY_INTERFACE_SUFFIX)
			set(ARGS_PRIMARY_INTERFACE_SUFFIX ".ixx")
		endif ()
		string(CONCAT ARGS_PRIMARY_INTERFACE "${TARGET}" "${ARGS_PRIMARY_INTERFACE_SUFFIX}")
	endif ()

	string(CONCAT TARGET_OBJECT "${TARGET}" "_object")
	string(CONCAT MODULE_INTERFACE_UNIT_IFC_NAME "${ARGS_PRIMARY_INTERFACE}" ".ifc")
	cmake_path(
		ABSOLUTE_PATH MODULE_INTERFACE_UNIT_IFC_NAME
		BASE_DIRECTORY "${CMAKE_BINARY_DIR}"
		NORMALIZE
		OUTPUT_VARIABLE MODULE_INTERFACE_UNIT_IFC
	)
	add_library("${TARGET_OBJECT}" OBJECT "${ARGS_PRIMARY_INTERFACE}")
	target_compile_definitions("${TARGET_OBJECT}"
		PUBLIC
			${ARGS_COMPILE_DEFINITIONS}
	)
	target_compile_options("${TARGET_OBJECT}"
		PUBLIC
			${ARGS_COMPILE_OPTIONS}
		PRIVATE
			"/interface"
			"/ifcOutput" "${MODULE_INTERFACE_UNIT_IFC}"
	)
	target_include_directories("${TARGET_OBJECT}"
		PUBLIC
			${ARGS_INCLUDE_DIRECTORIES}
	)
	target_include_directories("${TARGET_OBJECT}" SYSTEM
		PUBLIC
			${ARGS_SYSTEM_INCLUDE_DIRECTORIES}
	)
	target_link_libraries("${TARGET_OBJECT}"
		PUBLIC
			${ARGS_LINK_LIBRARIES}
	)

	set_property(
		SOURCE
			"${MODULE_INTERFACE_UNIT_IFC}"
		PROPERTY GENERATED TRUE
	)
	set_property(
		SOURCE
			"${MODULE_INTERFACE_UNIT_IFC}"
		APPEND
		PROPERTY OBJECT_DEPENDS
			"${ARGS_PRIMARY_INTERFACE}"
	)
	set_property(
		SOURCE
			"${ARGS_PRIMARY_INTERFACE}"
		APPEND
		PROPERTY OBJECT_OUTPUTS
			"${MODULE_INTERFACE_UNIT_IFC}"
	)

	string(CONCAT TARGET_IFC "${TARGET}" "_ifc")
	add_custom_target("${TARGET_IFC}"
		DEPENDS
			"${MODULE_INTERFACE_UNIT_IFC}"
	)

	add_library("${TARGET}" STATIC EXCLUDE_FROM_ALL ${ARGS_SOURCES})
	add_dependencies("${TARGET}" "${TARGET_IFC}")
	target_compile_options("${TARGET}"
		PUBLIC
			"/reference" "${MODULE_INTERFACE_UNIT_IFC}"
	)
	target_link_libraries("${TARGET}"
		PRIVATE
			"${TARGET_OBJECT}"
	)
endfunction ()
