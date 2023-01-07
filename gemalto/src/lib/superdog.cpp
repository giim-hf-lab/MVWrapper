#ifndef __GEMALTO_MODULE_EXPORT__

#include <chrono>
#include <type_traits>
#include <utility>

#include <dog_api.h>

#endif

#include "gemalto/superdog.hpp"

namespace
{

#include "superdog.vc"

template<typename F, typename... Args>
[[nodiscard]]
static inline bool _wrap(F&& f, Args&&... args) noexcept
	requires std::is_invocable_r_v<dog_status_t, F, Args...>
{
	return f(std::forward<Args>(args)...) == DOG_STATUS_OK;
}

}

namespace gemalto
{

[[nodiscard]]
bool superdog::get_version(version& v) noexcept
{
	return _wrap(dog_get_version, &v.major, &v.minor, &v.server, &v.number, _SUPERDOG_VENDOR_CODE);
}

superdog::superdog() noexcept : _handle(DOG_INVALID_HANDLE_VALUE) {}

superdog::~superdog() noexcept
{
	if (not close())
		std::terminate();
}

superdog::superdog(superdog&& other) noexcept : _handle(other._handle)
{
	other._handle = DOG_INVALID_HANDLE_VALUE;
}

[[nodiscard]]
superdog::operator bool() const noexcept
{
	return _handle != DOG_INVALID_HANDLE_VALUE;
}

[[nodiscard]]
bool superdog::open() noexcept
{
	return open(DOG_DEFAULT_FID);
}

[[nodiscard]]
bool superdog::open(feature_type feature) noexcept
{
	return _handle == DOG_INVALID_HANDLE_VALUE and _wrap(dog_login, feature, _SUPERDOG_VENDOR_CODE, &_handle);
}

[[nodiscard]]
bool superdog::close() noexcept
{
	if (_handle == DOG_INVALID_HANDLE_VALUE)
		return true;
	auto ret = _wrap(dog_logout, _handle);
	_handle = DOG_INVALID_HANDLE_VALUE;
	return ret;
}

[[nodiscard]]
bool superdog::encrypt(char *data, size_t size) const noexcept
{
	return _wrap(dog_encrypt, _handle, data, size);
}

[[nodiscard]]
bool superdog::decrypt(char *data, size_t size) const noexcept
{
	return _wrap(dog_decrypt, _handle, data, size);
}

[[nodiscard]]
bool superdog::get_size(file_id_type file_id, size_t& size) const noexcept
{
	if (dog_size_t s; _wrap(dog_get_size, _handle, file_id, &s))
	{
		size = s;
		return true;
	}
	return false;
}

[[nodiscard]]
bool superdog::read(file_id_type file_id, char *data, size_t size, size_t offset) const noexcept
{
	return _wrap(dog_read, _handle, file_id, offset, size, data);
}

[[nodiscard]]
bool superdog::write(file_id_type file_id, const char *data, size_t size, size_t offset) const noexcept
{
	return _wrap(dog_write, _handle, file_id, offset, size, data);
}

[[nodiscard]]
bool superdog::get_time(std::chrono::utc_seconds& time) const noexcept
{
	if (dog_time_t dog_time; _wrap(dog_get_time, _handle, &dog_time))
	{
		time = std::chrono::utc_seconds(std::chrono::seconds(dog_time));
		return true;
	}
	return false;
}

}
