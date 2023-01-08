module;

#include <cstddef>

#include <chrono>
#include <ranges>
#include <type_traits>
#include <utility>

#include <dog_api.h>

export module gemalto;

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
	dog_handle_t _handle;
public:
	[[nodiscard]]
	static bool get_version(version& v) noexcept
	{
		return _wrap(dog_get_version, &v.major, &v.minor, &v.server, &v.number, _SUPERDOG_VENDOR_CODE);
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

	superdog& operator=(superdog&&) = delete;

	[[nodiscard]]
	operator bool() const noexcept
	{
		return _handle != DOG_INVALID_HANDLE_VALUE;
	}

	[[nodiscard]]
	bool open(feature_type feature) noexcept
	{
		return _handle == DOG_INVALID_HANDLE_VALUE and _wrap(dog_login, feature, _SUPERDOG_VENDOR_CODE, &_handle);
	}

	[[nodiscard]]
	inline bool open() noexcept
	{
		return open(DOG_DEFAULT_FID);
	}

	[[nodiscard]]
	bool close() noexcept
	{
		if (_handle == DOG_INVALID_HANDLE_VALUE)
			return true;
		auto ret = _wrap(dog_logout, _handle);
		_handle = DOG_INVALID_HANDLE_VALUE;
		return ret;
	}

	[[nodiscard]]
	bool encrypt(char *data, size_t size) const noexcept
	{
		return _wrap(dog_encrypt, _handle, data, size);
	}

	[[nodiscard]]
	inline bool encrypt(std::ranges::contiguous_range auto& data) const noexcept
	{
		return encrypt(std::ranges::data(data), std::ranges::size(data));
	}

	[[nodiscard]]
	bool decrypt(char *data, size_t size) const noexcept
	{
		return _wrap(dog_decrypt, _handle, data, size);
	}

	[[nodiscard]]
	inline bool decrypt(std::ranges::contiguous_range auto& data) const noexcept
	{
		return decrypt(std::ranges::data(data), std::ranges::size(data));
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

	[[nodiscard]]
	bool read(file_id_type file_id, char *data, size_t size, size_t offset = 0) const noexcept
	{
		return _wrap(dog_read, _handle, file_id, offset, size, data);
	}

	[[nodiscard]]
	inline bool read(
		file_id_type file_id,
		std::ranges::contiguous_range auto& data,
		size_t offset = 0
	) const noexcept
	{
		return read(file_id, std::ranges::data(data), std::ranges::size(data), offset);
	}

	[[nodiscard]]
	bool write(file_id_type file_id, const char *data, size_t size, size_t offset = 0) const noexcept
	{
		return _wrap(dog_write, _handle, file_id, offset, size, data);
	}

	[[nodiscard]]
	inline bool write(
		file_id_type file_id,
		const std::ranges::contiguous_range auto& data,
		size_t offset = 0
	) const noexcept
	{
		return write(file_id, std::ranges::cdata(data), std::ranges::size(data), offset);
	}

	[[nodiscard]]
	bool get_time(std::chrono::utc_seconds& time) const noexcept
	{
		if (dog_time_t dog_time; _wrap(dog_get_time, _handle, &dog_time))
		{
			time = std::chrono::utc_seconds(std::chrono::seconds(dog_time));
			return true;
		}
		return false;
	}

	template<typename Duration, typename Rep, typename Period>
	[[nodiscard]]
	inline bool check_expiry(
		const std::chrono::utc_time<Duration>& expiry,
		const std::chrono::duration<Rep, Period>& tolerance
	) const noexcept
	{
		std::chrono::utc_seconds time;
		return get_time(time) and time <= expiry + tolerance;
	}

	template<typename Duration, typename Rep, typename Period>
	[[nodiscard]]
	inline bool check_expiry(
		const std::chrono::sys_time<Duration>& expiry,
		const std::chrono::duration<Rep, Period>& tolerance
	) const noexcept
	{
		return check_expiry(std::chrono::utc_clock::from_sys(expiry), tolerance);
	}

	template<typename Duration>
	[[nodiscard]]
	inline bool check_expiry(const std::chrono::utc_time<Duration>& expiry) const noexcept
	{
		return check_expiry(expiry, std::chrono::nanoseconds::zero());
	}

	template<typename Duration>
	[[nodiscard]]
	inline bool check_expiry(const std::chrono::sys_time<Duration>& expiry) const noexcept
	{
		return check_expiry(std::chrono::utc_clock::from_sys(expiry));
	}

	template<typename Rep, typename Period>
	[[nodiscard]]
	inline bool check_time(const std::chrono::duration<Rep, Period>& tolerance) const noexcept
	{
		return check_expiry(std::chrono::utc_clock::now(), tolerance);
	}

	[[nodiscard]]
	inline bool check_time() const noexcept
	{
		return check_expiry(std::chrono::utc_clock::now());
	}
};

}
