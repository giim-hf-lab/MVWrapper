module;

#include <cstddef>

#include <chrono>
#include <ranges>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include <fmt/compile.h>
#include <fmt/core.h>
#include <fmt/format.h>

#include <dog_api.h>

export module gemalto;

namespace
{

#include "./superdog.vc"

}

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
	template<typename S, typename F, typename... Args>
		requires std::is_invocable_r_v<dog_status_t, F, Args...>
	inline static void _raise_if_fail(S&& what, F&& f, Args&&... args)
	{
		if (auto ret = f(std::forward<Args>(args)...); ret != dog_error_codes::DOG_STATUS_OK)
		[[unlikely]]
			throw std::runtime_error(fmt::format(FMT_COMPILE("({:#X}) {}"), ret, std::forward<S>(what)));
	}

	dog_handle_t _handle;
public:
	[[nodiscard]]
	static version get_version() noexcept
	{
		version ret;
		_raise_if_fail(
			"failed to get version",
			dog_get_version,
			&ret.major,
			&ret.minor,
			&ret.server,
			&ret.number,
			_SUPERDOG_VENDOR_CODE
		);
		return ret;
	}

	superdog(feature_type feature) : _handle(DOG_INVALID_HANDLE_VALUE)
	{
		_raise_if_fail("failed to find superdog", dog_login, feature, _SUPERDOG_VENDOR_CODE, &_handle);
	}

	~superdog() noexcept
	{
		if (_handle != DOG_INVALID_HANDLE_VALUE)
		{
			dog_logout(_handle);
			_handle = DOG_INVALID_HANDLE_VALUE;
		}
	}

	superdog(const superdog&) = delete;

	superdog(superdog&& other) noexcept : _handle(other._handle)
	{
		other._handle = DOG_INVALID_HANDLE_VALUE;
	}

	superdog& operator=(const superdog&) = delete;

	superdog& operator=(superdog&&) = delete;

	template<std::ranges::contiguous_range R>
	void encrypt(R& data)
	{
		_raise_if_fail(
			"failed to encrypt data",
			dog_encrypt,
			_handle,
			std::ranges::data(data),
			std::ranges::size(data) * sizeof(std::ranges::range_value_t<R>)
		);
	}

	template<std::ranges::contiguous_range R>
	void decrypt(R& data)
	{
		_raise_if_fail(
			"failed to decrypt data",
			dog_decrypt,
			_handle,
			std::ranges::data(data),
			std::ranges::size(data) * sizeof(std::ranges::range_value_t<R>)
		);
	}

	size_t get_size(file_id_type file_id)
	{
		dog_size_t size;
		_raise_if_fail("failed to get file size", dog_get_size, _handle, file_id, &size);
		return size;
	}

	template<std::ranges::contiguous_range R>
	void read(R& data, file_id_type file_id, size_t offset = 0)
	{
		_raise_if_fail(
			"failed to read data",
			dog_read,
			_handle,
			file_id,
			offset,
			std::ranges::size(data) * sizeof(std::ranges::range_value_t<R>),
			std::ranges::data(data)
		);
	}

	template<std::ranges::contiguous_range R>
	void write(const R& data, file_id_type file_id, size_t offset = 0)
	{
		_raise_if_fail(
			"failed to write data",
			dog_write,
			_handle,
			file_id,
			offset,
			std::ranges::size(data) * sizeof(std::ranges::range_value_t<R>),
			std::ranges::data(data)
		);
	}

	std::chrono::utc_seconds get_time()
	{
		dog_time_t time;
		_raise_if_fail("failed to get time", dog_get_time, _handle, &time);
		return std::chrono::utc_seconds(std::chrono::seconds(time));
	}
};

}
