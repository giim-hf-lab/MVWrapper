#ifndef __GEMALTO_SUPERDOG_HPP__
#define __GEMALTO_SUPERDOG_HPP__

#ifndef __GEMALTO_MODULE_EXPORT__

#include <cstddef>

#include <chrono>
#include <ranges>

#include <dog_api.h>

#endif

namespace gemalto
{

#ifdef __GEMALTO_MODULE_EXPORT__
__GEMALTO_MODULE_EXPORT__
#endif
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
	static bool get_version(version& v) noexcept;

	superdog() noexcept;

	~superdog() noexcept;

	superdog(const superdog&) = delete;

	superdog(superdog&& other) noexcept;

	superdog& operator=(const superdog&) = delete;

	superdog& operator=(superdog&&) = delete;

	[[nodiscard]]
	operator bool() const noexcept;

	[[nodiscard]]
	bool open() noexcept;

	[[nodiscard]]
	bool open(feature_type feature) noexcept;

	[[nodiscard]]
	bool close() noexcept;

	[[nodiscard]]
	bool encrypt(char *data, size_t size) const noexcept;

	[[nodiscard]]
	inline bool encrypt(std::ranges::contiguous_range auto& data) const noexcept
	{
		return encrypt(std::ranges::data(data), std::ranges::size(data));
	}

	[[nodiscard]]
	bool decrypt(char *data, size_t size) const noexcept;

	[[nodiscard]]
	inline bool decrypt(std::ranges::contiguous_range auto& data) const noexcept
	{
		return decrypt(std::ranges::data(data), std::ranges::size(data));
	}

	[[nodiscard]]
	bool get_size(file_id_type file_id, size_t& size) const noexcept;

	[[nodiscard]]
	bool read(file_id_type file_id, char *data, size_t size, size_t offset = 0) const noexcept;

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
	bool write(file_id_type file_id, const char *data, size_t size, size_t offset = 0) const noexcept;

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
	bool get_time(std::chrono::utc_seconds& time) const noexcept;

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

#endif
