module;

#include <cstddef>

#include <chrono>
#include <ranges>
#include <type_traits>
#include <utility>

#include <dog_api.h>

#define __GEMALTO_MODULE_EXPORT__ export

export module gemalto;

#include "./src/lib/superdog.cpp"
