module;

#include <cstddef>

#include <chrono>
#include <concepts>
#include <exception>
#include <ranges>
#include <type_traits>
#include <utility>

#include <dog_api.h>

export module gemalto;

namespace gemalto
{

export
struct superdog final
{
	using feature_type = typename dog_feature_t;
	using file_id_type = typename dog_fileid_t;

	struct version final
	{
		unsigned int major, minor, server, number;
	};
private:
	#include "./superdog.vc"

	template<typename F, typename... Args>
		requires std::is_invocable_r_v<dog_status_t, F, Args...>
	[[nodiscard]]
	static inline bool _wrap(F&& f, Args&&... args) noexcept
	{
		return f(std::forward<Args>(args)...) == DOG_STATUS_OK;
	}

	dog_handle_t _handle;
public:
	[[nodiscard]]
	static bool get_version(version& v) noexcept
	{
		return _wrap(dog_get_version, &v.major, &v.minor, &v.server, &v.number, _VENDOR_CODE);
	}

	superdog() noexcept : _handle(DOG_INVALID_HANDLE_VALUE) {}

	~superdog() noexcept
	{
		if (not close())
			std::terminate();
	}

	superdog(const superdog&) = delete;

	superdog(superdog&& other) noexcept : _handle(other._handle)
	{
		other._handle = DOG_INVALID_HANDLE_VALUE;
	}

	superdog& operator=(const superdog&) = delete;

	superdog& operator=(superdog&& other) noexcept
	{
		if (not close())
			std::terminate();
		_handle = other._handle;
		other._handle = DOG_INVALID_HANDLE_VALUE;
	}

	operator bool() const noexcept
	{
		return _handle != DOG_INVALID_HANDLE_VALUE;
	}

	[[nodiscard]]
	bool open(feature_type feature) noexcept
	{
		return _handle == DOG_INVALID_HANDLE_VALUE and _wrap(dog_login, feature, _VENDOR_CODE, &_handle);
	}

	[[nodiscard]]
	bool close() noexcept
	{
		if (_handle == DOG_INVALID_HANDLE_VALUE)
			return true;
		if (_wrap(dog_logout, _handle))
		{
			_handle = DOG_INVALID_HANDLE_VALUE;
			return true;
		}
		return false;
	}

	template<std::ranges::contiguous_range R>
	[[nodiscard]]
	bool encrypt(R& data) const noexcept
	{
		return _wrap(
			dog_encrypt,
			_handle,
			std::ranges::data(data),
			std::ranges::size(data) * sizeof(std::ranges::range_value_t<R>)
		);
	}

	template<std::ranges::contiguous_range R>
	[[nodiscard]]
	bool decrypt(R& data) const noexcept
	{
		return _wrap(
			dog_decrypt,
			_handle,
			std::ranges::data(data),
			std::ranges::size(data) * sizeof(std::ranges::range_value_t<R>)
		);
	}

	[[nodiscard]]
	bool get_size(file_id_type file_id, size_t& size) const noexcept
	{
		if (dog_size_t s; _wrap(dog_get_size, _handle, file_id, &s))
		{
			size = s;
			return true;
		}
		return false;
	}

	template<std::ranges::contiguous_range R>
	[[nodiscard]]
	bool read(R& data, file_id_type file_id, size_t offset = 0) const noexcept
	{
		return _wrap(
			dog_read,
			_handle,
			file_id,
			offset,
			std::ranges::size(data) * sizeof(std::ranges::range_value_t<R>),
			std::ranges::data(data)
		);
	}

	template<std::ranges::contiguous_range R>
	[[nodiscard]]
	bool write(const R& data, file_id_type file_id, size_t offset = 0) const noexcept
	{
		return _wrap(
			dog_write,
			_handle,
			file_id,
			offset,
			std::ranges::size(data) * sizeof(std::ranges::range_value_t<R>),
			std::ranges::cdata(data)
		);
	}

	[[nodiscard]]
	bool get_time(std::chrono::utc_seconds& time) const noexcept
	{
		if (dog_time_t t; _wrap(dog_get_time, _handle, &t))
		{
			time = std::chrono::utc_seconds(std::chrono::seconds(t));
			return true;
		}
		return false;
	}
};

}
