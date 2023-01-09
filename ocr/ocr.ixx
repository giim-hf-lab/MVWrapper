module;

#include <cstddef>
#include <cstdint>

#include <concepts>
#include <memory>
#include <ranges>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <clipper2/clipper.core.h>
#include <clipper2/clipper.offset.h>
#include <mio/mmap.hpp>
#include <onnxruntime/core/session/onnxruntime_c_api.h>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

export module ocr;

namespace std
{

template<typename T>
concept arithmetic = is_arithmetic_v<T>;

}

namespace cv
{

template<std::arithmetic PointType, size_t... Indices, std::arithmetic... PointTypes>
[[nodiscard]]
static inline Mat _from_points(std::index_sequence<Indices...>, const Point_<PointTypes>&... points)
{
	Mat output(sizeof...(PointTypes), 1, CV_MAKETYPE(traits::Depth<PointType>::value, 2));
	((output.at<Point_<PointType>>(Indices, 0) = points), ...);
	return output;
}

template<std::arithmetic PointType, std::arithmetic... PointTypes>
[[nodiscard]]
inline Mat from_points(const Point_<PointTypes>&... points)
{
	return _from_points<PointType>(std::index_sequence_for<PointTypes...> {}, points...);
}

inline void crop(const Mat& image, const RotatedRect& box, Mat& vertices, Mat& cropped)
{
	vertices.create(4, 1, CV_32FC2);
	box.points(vertices.ptr<Point2f>());
	vertices.convertTo(vertices, CV_32SC2);

	Size dest_size = box.size;
	auto normalised = from_points<int>(
		Point { 0, dest_size.height },
		Point { 0, 0 },
		Point { dest_size.width, 0 },
		Point { dest_size.width, dest_size.height }
	);

	warpPerspective(
		image,
		cropped,
		getPerspectiveTransform(vertices, normalised),
		dest_size,
		InterpolationFlags::INTER_CUBIC,
		BorderTypes::BORDER_REPLICATE
	);
}

}

namespace
{

class model final
{
	static Ort::AllocatorWithDefaultOptions _allocator;

	[[nodiscard]]
	static inline Ort::Session _create_session(
		const std::basic_string<ORTCHAR_T>& model_path,
		Ort::Env& env,
		const Ort::SessionOptions& common_options,
		GraphOptimizationLevel graph_opt_level
	)
	{
		if (graph_opt_level > GraphOptimizationLevel::ORT_DISABLE_ALL)
		[[likely]]
		{
			auto optimised_model_path = model_path + ORT_TSTR(".opt");
			auto session_options = common_options.Clone();
			session_options
				.SetOptimizedModelFilePath(optimised_model_path.c_str())
				.SetGraphOptimizationLevel(graph_opt_level);
			return { env, model_path.c_str(), session_options };
		}
		return { env, model_path.c_str(), common_options };
	}

	Ort::Env _env;
	Ort::Session _session;
	std::vector<Ort::AllocatedStringPtr> _names;
	std::vector<const char *> _input_names, _output_names;

	inline void _run(
		const Ort::Value *inputs,
		size_t input_num,
		Ort::Value *outputs,
		size_t output_num,
		const Ort::RunOptions& run_options = {}
	)
	{
		if (input_num != _input_names.size())
			throw std::runtime_error("model input number mismatch");
		if (output_num != _output_names.size())
			throw std::runtime_error("model output number mismatch");
		_session.Run(run_options, _input_names.data(), inputs, input_num, _output_names.data(), outputs, output_num);
	}
public:
	model(
		const std::basic_string<ORTCHAR_T>& model_path,
		const Ort::SessionOptions& common_options,
		GraphOptimizationLevel graph_opt_level = GraphOptimizationLevel::ORT_ENABLE_EXTENDED
	) :
		_env(),
		_session(_create_session(model_path, _env, common_options, graph_opt_level)),
		_names(),
		_input_names(),
		_output_names()
	{
		auto input_num = _session.GetInputCount(), output_num = _session.GetOutputCount();

		_names.reserve(input_num + output_num);
		_input_names.reserve(input_num);
		for (size_t i = 0; i < input_num; ++i)
			_input_names.emplace_back(_names.emplace_back(_session.GetInputNameAllocated(i, _allocator)).get());
		_output_names.reserve(output_num);
		for (size_t i = 0; i < output_num; ++i)
			_output_names.emplace_back(_names.emplace_back(_session.GetOutputNameAllocated(i, _allocator)).get());
	}

	void run(
		const Ort::Value& input,
		Ort::Value& output,
		const Ort::RunOptions& run_options = {}
	)
	{
		_run(&input, 1, &output, 1, run_options);
	}

	void run(
		const std::ranges::contiguous_range auto& inputs,
		std::ranges::contiguous_range auto& outputs,
		const Ort::RunOptions& run_options = {}
	)
	{
		_run(
			std::ranges::data(inputs),
			std::ranges::size(inputs),
			std::ranges::data(outputs),
			std::ranges::size(outputs),
			run_options
		);
	}
};

Ort::AllocatorWithDefaultOptions model::_allocator {};

}

namespace ocr
{

namespace
{

static const auto _GLOBAL_DEFAULT_OPTIONS = []
{
	Ort::SessionOptions session_options;
	OrtCUDAProviderOptions cuda_options;
	session_options
		.EnableCpuMemArena()
		.EnableMemPattern()
		.DisableProfiling()
		.AppendExecutionProvider_CUDA(cuda_options);
	return session_options;
}();

static const auto _OPENCV_MEM_INFO = Ort::MemoryInfo::CreateCpu(
	OrtAllocatorType::OrtInvalidAllocator,
	OrtMemType::OrtMemTypeCPU
);

static const auto _BLACK = cv::Scalar::all(0), _WHITE = cv::Scalar::all(255);

static const auto _kernel_2x2 = cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, { 2, 2 });

}

namespace scalers
{

export
struct base
{
	virtual ~base() noexcept = default;

	[[nodiscard]]
	virtual bool run(const cv::Mat& src, cv::Mat& dest) = 0;
};

export
struct trivial final : public base
{
	[[nodiscard]]
	virtual bool run(const cv::Mat&, cv::Mat&) noexcept override
	{
		return false;
	}
};

export
class resize final : public base
{
	cv::Size _size;
public:
	resize(cv::Size size) noexcept : _size(std::move(size)) {}

	[[nodiscard]]
	virtual bool run(const cv::Mat& src, cv::Mat& dest) override
	{
		if (src.size() == _size)
			return false;
		cv::resize(src, dest, _size);
		return true;
	}
};

export
class zoom final : public base
{
	size_t _side_len;
	bool _min_side;
public:
	zoom(size_t side_len, bool min_side) noexcept : _side_len(side_len), _min_side(min_side) {}

	[[nodiscard]]
	virtual bool run(const cv::Mat& src, cv::Mat& dest) override
	{
		// https://github.com/PaddlePaddle/PaddleOCR/blob/v2.6.0/ppocr/data/imaug/operators.py#L254-L301

		auto height = src.rows, width = src.cols;
		bool scaled;
		if (
			_min_side and height <= width and height < _side_len or
			not _min_side and height >= width and height > _side_len
		)
		{
			width *= double(_side_len) / height;
			height = _side_len;
			scaled = true;
		}
		else if (
			_min_side and height > width and width < _side_len or
			not _min_side and height < width and width > _side_len
		)
		{
			height *= double(_side_len) / width;
			width = _side_len;
			scaled = true;
		}
		else
			scaled = false;
		if (auto r = height % 32)
		{
			height += 32 - r;
			scaled = true;
		}
		if (auto r = width % 32)
		{
			width += 32 - r;
			scaled = true;
		}

		if (scaled)
			cv::resize(src, dest, { width, height });
		return scaled;
	}
};

}

export
template<typename T>
concept scaler = std::derived_from<T, scalers::base>;

namespace detectors
{

export
struct base
{
	virtual ~base() noexcept = default;

	[[nodiscard]]
	virtual bool run(const cv::Mat& src, std::vector<cv::RotatedRect>& boxes) = 0;
};

export
struct trivial final : public base
{
	[[nodiscard]]
	virtual bool run(const cv::Mat&, std::vector<cv::RotatedRect>&) noexcept override
	{
		return false;
	}
};

export
struct db final : public base
{
	enum class scoring_methods
	{
		BOX,
		APPROXIMATE,
		ORIGINAL
	};
private:
	template<scoring_methods scoring_method>
	[[nodiscard]]
	static inline bool _transform(
		const cv::Mat& scores,
		cv::Mat& src,
		cv::Mat& dest,
		double box_threshold,
		double unclip_ratio
	)
	{
		if constexpr (scoring_method == scoring_methods::APPROXIMATE)
		{
			cv::approxPolyDP(src, src, 0.002 * cv::arcLength(src, true), true);
			if (dest.rows < 4)
				return false;
		}

		// https://github.com/PaddlePaddle/PaddleOCR/blob/v2.6.0/ppocr/postprocess/db_postprocess.py#L182-L218

		auto bounding = cv::boundingRect(src);
		cv::Mat mask(bounding.height, bounding.width, CV_8UC1, _BLACK);
		if constexpr (scoring_method == scoring_methods::BOX)
		{
			auto enclosing = cv::minAreaRect(src);
			cv::Mat vertices(4, 1, CV_32FC2);
			enclosing.points(vertices.ptr<cv::Point2f>());
			cv::fillPoly(mask, vertices, _WHITE, cv::LineTypes::LINE_AA, 0, -bounding.tl());
		}
		else
			cv::fillPoly(mask, src, _WHITE, cv::LineTypes::LINE_AA, 0, -bounding.tl());

		if (cv::mean(scores(bounding), mask)[0] < box_threshold)
			return false;

		// https://github.com/PaddlePaddle/PaddleOCR/blob/v2.6.0/ppocr/postprocess/db_postprocess.py#L151-L157

		Clipper2Lib::Path64 contour_path;
		contour_path.reserve(src.rows);
		for (int i = 0; i < src.rows; ++i)
			contour_path.emplace_back(src.at<Clipper2Lib::Point<int>>(i, 0));
		Clipper2Lib::ClipperOffset clipper_offset;
		clipper_offset.AddPaths({ contour_path }, Clipper2Lib::JoinType::Round, Clipper2Lib::EndType::Polygon);
		auto result = clipper_offset.Execute(cv::contourArea(src) * unclip_ratio / cv::arcLength(src, true));
		if (result.size() != 1)
			return false;

		const auto& unclipped = result.front();
		dest.create(unclipped.size(), 1, CV_32SC2);
		for (size_t i = 0; i < unclipped.size(); ++i)
		{
			const auto& point = unclipped[i];
			dest.at<Clipper2Lib::Point<int>>(i, 0).Init(point.x, point.y);
		}
		return true;
	}

	cv::Scalar _mean, _stddev;
	model _model;

	double _threshold;
	bool _use_dilation;
	scoring_methods _scoring_method;
	size_t _max_candidates;
	double _box_threshold;
	double _unclip_ratio;
public:
	db(
		cv::Scalar mean,
		cv::Scalar stddev,
		double threshold,
		bool use_dilation,
		scoring_methods scoring_method,
		size_t max_candidates,
		double box_threshold,
		double unclip_ratio,
		const std::basic_string<ORTCHAR_T>& model_path,
		const Ort::SessionOptions& options,
		GraphOptimizationLevel graph_opt_level = GraphOptimizationLevel::ORT_ENABLE_EXTENDED
	) :
		_mean(std::move(mean)), 
		_stddev(std::move(stddev)),
		_threshold(threshold),
		_use_dilation(use_dilation),
		_scoring_method(scoring_method),
		_max_candidates(max_candidates),
		_box_threshold(box_threshold),
		_unclip_ratio(unclip_ratio),
		_model(model_path, options, graph_opt_level)
	{}

	[[nodiscard]]
	virtual bool run(const cv::Mat& src, std::vector<cv::RotatedRect>& boxes) noexcept override
	{
		cv::Mat input;
		cv::subtract(src, _mean, input, cv::noArray(), CV_32FC3);
		cv::divide(input, _stddev, input, 1, CV_32FC3);

		auto stride = src.rows * src.cols;
		input = input.reshape(1, stride).t();
		cv::Mat output(src.rows, src.cols, CV_32FC1);

		int64_t shapes[] { 1, 3, src.rows, src.cols };
		auto input_tensor = Ort::Value::CreateTensor<float>(
			_OPENCV_MEM_INFO,
			input.ptr<float>(),
			3 * stride,
			shapes,
			4
		);

		shapes[1] = 1;
		auto output_tensor = Ort::Value::CreateTensor<float>(
			_OPENCV_MEM_INFO,
			output.ptr<float>(),
			stride,
			shapes,
			4
		);

		_model.run(input_tensor, output_tensor);

		// https://github.com/PaddlePaddle/PaddleOCR/blob/v2.6.0/ppocr/postprocess/db_postprocess.py#L230-L235

		cv::Mat mask;
		cv::threshold(output, mask, _threshold, 255, cv::ThresholdTypes::THRESH_BINARY);
		if (_use_dilation)
			cv::dilate(mask, mask, _kernel_2x2);

		// https://github.com/PaddlePaddle/PaddleOCR/blob/v2.6.0/ppocr/postprocess/db_postprocess.py#L57-L149

		std::vector<cv::Mat> contours;
		cv::findContours(
			mask,
			contours,
			cv::RetrievalModes::RETR_LIST,
			cv::ContourApproximationModes::CHAIN_APPROX_SIMPLE
		);

		size_t kept = 0;

#define _OCR_DB_POSTPROCESS_TRANSFORM(LABEL, METHOD) \
			LABEL: \
				for (size_t i = 0; i < contours.size() and kept < _max_candidates; ++i) \
					if (_transform<scoring_methods::METHOD>( \
						output, \
						contours[i], \
						contours[kept], \
						_box_threshold, \
						_unclip_ratio \
					)) \
						++kept;

#define _OCR_DB_POSTPROCESS_TRANSFORM_DIRECT(METHOD) \
			_OCR_DB_POSTPROCESS_TRANSFORM(case scoring_methods::METHOD, METHOD) \
				break;

		switch (_scoring_method)
		{
			_OCR_DB_POSTPROCESS_TRANSFORM_DIRECT(BOX)
			_OCR_DB_POSTPROCESS_TRANSFORM_DIRECT(APPROXIMATE)
			_OCR_DB_POSTPROCESS_TRANSFORM(default, ORIGINAL)
		}

		if (not kept)
			return false;
		contours.resize(kept);

		boxes.reserve(boxes.size() + kept);
		for (const auto& contour : contours)
			boxes.push_back(cv::minAreaRect(contour));
		return true;
	}
};

}

export
template<typename T>
concept detector = std::derived_from<T, detectors::base>;

namespace classifiers
{

export
struct base
{
	virtual ~base() noexcept = default;

	[[nodiscard]]
	virtual bool run(const std::vector<cv::Mat>& fragments, std::vector<size_t>& indices) = 0;
};

export
struct trivial final : public base
{
	[[nodiscard]]
	virtual bool run(const std::vector<cv::Mat>&, std::vector<size_t>&) noexcept override
	{
		return false;
	}
};

export
class concrete final : public base
{
	static const cv::Scalar _mean, _stddev;

	size_t _batch_size;
	cv::Size _shape;
	double _threshold;
	model _model;
public:
	concrete(
		size_t batch_size,
		cv::Size shape,
		double threshold,
		const std::basic_string<ORTCHAR_T>& model_path,
		const Ort::SessionOptions& options,
		GraphOptimizationLevel graph_opt_level = GraphOptimizationLevel::ORT_ENABLE_EXTENDED
	) :
		_batch_size(batch_size),
		_shape(std::move(shape)),
		_threshold(threshold),
		_model(model_path, options, graph_opt_level)
	{}

	[[nodiscard]]
	virtual bool run(const std::vector<cv::Mat>& fragments, std::vector<size_t>& indices) override
	{
		auto stride = _shape.area();
		cv::Mat input({ int(_batch_size), 3, stride }, CV_32FC1);
		cv::Mat output(_batch_size, 2, CV_32FC1);

		int64_t shapes[] { _batch_size, 3, _shape.height, _shape.width };
		auto input_tensor = Ort::Value::CreateTensor<float>(
			_OPENCV_MEM_INFO,
			input.ptr<float>(),
			_batch_size * 3 * stride,
			shapes,
			4
		);

		shapes[1] = 2;
		auto output_tensor = Ort::Value::CreateTensor<float>(
			_OPENCV_MEM_INFO,
			output.ptr<float>(),
			_batch_size * 2,
			shapes,
			2
		);

		indices.reserve(indices.size() + fragments.size());
		for (size_t i = 0; i < fragments.size(); i += _batch_size)
		{
			size_t current_batch = std::min(_batch_size, fragments.size() - i);
			for (size_t c = 0, j = i; c < current_batch; ++c, ++j)
			{
				cv::Mat tmp;
				if (const auto& fragment = fragments[j]; fragment.size() == _shape)
					cv::subtract(fragment, _mean, tmp, cv::noArray(), CV_32FC3);
				else
				{
					cv::resize(fragment, tmp, _shape);
					cv::subtract(tmp, _mean, tmp, cv::noArray(), CV_32FC3);
				}
				cv::divide(tmp, _stddev, tmp, 1, CV_32FC3);

				cv::Mat store(3, stride, CV_32FC1, input.ptr<float>(c, 0, 0));
				cv::copyTo(tmp.reshape(1, stride).t(), store, cv::noArray());
			}

			_model.run(input_tensor, output_tensor);

			for (size_t c = 0, j = i; c < current_batch; ++c, ++j)
			{
				double score;
				int index;
				cv::minMaxIdx(output.row(c), nullptr, &score, nullptr, &index);
				if (index and score >= _threshold)
					indices.push_back(j);
			}
		}
		return true;
	}
};

const cv::Scalar concrete::_mean = cv::Scalar::all(0.5), concrete::_stddev = cv::Scalar::all(0.5);

}

export
template<typename T>
concept classifier = std::derived_from<T, classifiers::base>;

namespace recognisers
{

export
struct base
{
	virtual ~base() noexcept = default;

	[[nodiscard]]
	virtual bool run(
		const std::vector<cv::Mat>& fragments,
		std::vector<std::tuple<size_t, std::string, double>>& results
	) = 0;
};

export
struct trivial final : public base
{
	[[nodiscard]]
	virtual bool run(
		const std::vector<cv::Mat>&,
		std::vector<std::tuple<size_t, std::string, double>>&
	) noexcept override
	{
		return false;
	}
};

export
class ctc final : public base
{
	static const cv::Scalar _mean, _stddev;

	size_t _batch_size;
	cv::Size _shape;
	double _threshold;

	model _model;
	std::vector<std::vector<char>> _dictionary;
public:
	ctc(
		size_t batch_size,
		cv::Size shape,
		double threshold,
		const std::string& dictionary_path,
		const std::basic_string<ORTCHAR_T>& model_path,
		const Ort::SessionOptions& options,
		GraphOptimizationLevel graph_opt_level = GraphOptimizationLevel::ORT_ENABLE_EXTENDED
	) :
		_batch_size(batch_size),
		_shape(std::move(shape)),
		_threshold(threshold),
		_model(model_path, options, graph_opt_level),
		_dictionary()
	{
		mio::mmap_source dict(dictionary_path);
		std::vector<char> buffer;
		buffer.reserve(4);
		// Unicode replacement character 0xFFFD in UTF-8
		buffer.push_back(0xef);
		buffer.push_back(0xbf);
		buffer.push_back(0xbd);
		_dictionary.emplace_back(buffer);
		buffer.clear();
		for (size_t i = 0; i < dict.size(); ++i)
		{
			char c = dict[i];
			switch (c)
			{
				case '\r':
				case '\n':
					if (buffer.size())
					{
						_dictionary.push_back(buffer);
						buffer.clear();
					}
					break;
				default:
					buffer.push_back(c);
			}
			if (buffer.size())
				_dictionary.emplace_back(std::move(buffer)).shrink_to_fit();
			_dictionary.emplace_back(1, ' ');
			_dictionary.shrink_to_fit();
		}
	}

	[[nodiscard]]
	virtual inline bool run(
		const std::vector<cv::Mat>& fragments,
		std::vector<std::tuple<size_t, std::string, double>>& results
	) override
	{
		auto stride = _shape.area();
		cv::Mat input({ int(_batch_size), 3, stride }, CV_32FC1);
		cv::Mat output({ int(_batch_size), 40, int(_dictionary.size()) }, CV_32FC1);

		int64_t shapes[] { _batch_size, 3, _shape.height, _shape.width };
		auto input_tensor = Ort::Value::CreateTensor<float>(
			_OPENCV_MEM_INFO,
			input.ptr<float>(),
			_batch_size * 3 * stride,
			shapes,
			4
		);

		shapes[1] = 40;
		shapes[2] = _dictionary.size();
		auto output_tensor = Ort::Value::CreateTensor<float>(
			_OPENCV_MEM_INFO,
			output.ptr<float>(),
			_batch_size * 40 * _dictionary.size(),
			shapes,
			3
		);

		results.reserve(results.size() + fragments.size());
		std::string buffer;
		buffer.reserve(160);
		for (size_t i = 0; i < fragments.size(); i += _batch_size)
		{
			size_t current_batch = std::min(_batch_size, fragments.size() - i);
			for (size_t c = 0, j = i; c < current_batch; ++c, ++j)
			{
				cv::Mat tmp;
				if (const auto& fragment = fragments[j]; fragment.size() == _shape)
					cv::subtract(fragment, _mean, tmp, cv::noArray(), CV_32FC3);
				else
				{
					cv::resize(fragment, tmp, _shape);
					cv::subtract(tmp, _mean, tmp, cv::noArray(), CV_32FC3);
				}
				cv::divide(tmp, _stddev, tmp, 1, CV_32FC3);

				cv::Mat store(3, stride, CV_32FC1, input.ptr<float>(c, 0, 0));
				cv::copyTo(tmp.reshape(1, stride).t(), store, cv::noArray());
			}

			_model.run(input_tensor, output_tensor);

			for (size_t c = 0, j = i; c < current_batch; ++c, ++j)
			{
				cv::Mat scores(40, _dictionary.size(), CV_32FC1, output.ptr<float>(c, 0, 0));
				int last_index = _dictionary.size();
				size_t count = 0;
				double total_score = 0;
				for (size_t i = 0; i < 40; ++i)
				{
					double score;
					int index;
					cv::minMaxIdx(scores.row(i), nullptr, &score, nullptr, &index);
					if (index != last_index)
					{
						++count;
						total_score += score;
						const auto& word = _dictionary[index];
						buffer.append(word.data(), word.size());
						last_index = index;
					}
				}
				total_score /= count;
				if (total_score >= _threshold)
					results.emplace_back(j, buffer, total_score);
				buffer.clear();
			}
		}
		return true;
	}
};

}

export
template<typename T>
concept recogniser = std::derived_from<T, recognisers::base>;

namespace pmr
{

export
class system final
{
	std::unique_ptr<scalers::base> _scaler;
	std::unique_ptr<detectors::base> _detector;
	std::unique_ptr<classifiers::base> _classifier;
	std::unique_ptr<recognisers::base> _recogniser;
public:
	system(
		std::unique_ptr<scalers::base> scaler,
		std::unique_ptr<detectors::base> detector,
		std::unique_ptr<classifiers::base> classifier,
		std::unique_ptr<recognisers::base> recogniser
	) :
		_scaler(std::move(scaler)),
		_detector(std::move(detector)),
		_classifier(std::move(classifier)),
		_recogniser(std::move(recogniser))
	{}

	[[nodiscard]]
	std::vector<std::tuple<cv::Mat, std::string, double>> ocr(const cv::Mat& image)
	{
		std::vector<cv::Mat> fragments, indices;
		if (cv::Mat scaled; _scaler and _scaler->run(image, scaled))
			if (std::vector<cv::RotatedRect> boxes; _detector and _detector->run(scaled, boxes))
			{
				double hr = double(image.rows) / scaled.rows, wr = double(image.cols) / scaled.cols;

				fragments.reserve(boxes.size());
				indices.reserve(boxes.size());
				for (auto& box : boxes)
				{
					box.center.x *= wr;
					box.center.y *= hr;
					box.size.width *= wr;
					box.size.height *= hr;
					cv::crop(image, box, indices.emplace_back(), fragments.emplace_back());
				}
			}
			else
			{
				fragments.emplace_back(image);
				indices.emplace_back();
			}
		else
			if (std::vector<cv::RotatedRect> boxes; _detector and _detector->run(image, boxes))
			{
				fragments.reserve(boxes.size());
				indices.reserve(boxes.size());
				for (const auto& box : boxes)
					cv::crop(image, box, indices.emplace_back(), fragments.emplace_back());
			}
			else
			{
				fragments.emplace_back(image);
				indices.emplace_back();
			}

		if (std::vector<size_t> rotations; _classifier and _classifier->run(fragments, rotations))
			for (const auto& index : rotations)
			{
				auto& fragment = fragments[index];
				cv::flip(fragment, fragment, -1);
			}

		std::vector<std::tuple<cv::Mat, std::string, double>> ret;
		if (
			std::vector<std::tuple<size_t, std::string, double>> results;
			_recogniser and _recogniser->run(fragments, results)
		)
		{
			ret.reserve(results.size());
			for (auto& [index, text, score] : results)
				ret.emplace_back(std::move(indices[index]), std::move(text), score);
		}
		else
		{
			ret.reserve(indices.size());
			for (auto& index : indices)
				ret.emplace_back(std::move(index), std::string {}, 0.0);
		}
		return ret;
	}
};

}

export
template<
	scaler Scaler,
	detector Detector,
	classifier Classifier,
	recogniser Recogniser
>
class system final
{
	Scaler _scaler;
	Detector _detector;
	Classifier _classifier;
	Recogniser _recogniser;
public:
	system(
		Scaler scaler,
		Detector detector,
		Classifier classifier,
		Recogniser recogniser
	) :
		_scaler(std::move(scaler)),
		_detector(std::move(detector)),
		_classifier(std::move(classifier)),
		_recogniser(std::move(recogniser))
	{}

	[[nodiscard]]
	std::vector<std::tuple<cv::Mat, std::string, double>> ocr(const cv::Mat& image)
	{
		std::vector<cv::Mat> fragments, indices;
		if (cv::Mat scaled; _scaler.run(image, scaled))
			if (std::vector<cv::RotatedRect> boxes; _detector.run(scaled, boxes))
			{
				double hr = double(image.rows) / scaled.rows, wr = double(image.cols) / scaled.cols;

				fragments.reserve(boxes.size());
				indices.reserve(boxes.size());
				for (auto& box : boxes)
				{
					box.center.x *= wr;
					box.center.y *= hr;
					box.size.width *= wr;
					box.size.height *= hr;
					cv::crop(image, box, indices.emplace_back(), fragments.emplace_back());
				}
			}
			else
			{
				fragments.emplace_back(image);
				indices.emplace_back();
			}
		else
			if (std::vector<cv::RotatedRect> boxes; _detector.run(image, boxes))
			{
				fragments.reserve(boxes.size());
				indices.reserve(boxes.size());
				for (const auto& box : boxes)
					cv::crop(image, box, indices.emplace_back(), fragments.emplace_back());
			}
			else
			{
				fragments.emplace_back(image);
				indices.emplace_back();
			}

		if (std::vector<size_t> rotations; _classifier.run(fragments, rotations))
			for (const auto& index : rotations)
			{
				auto& fragment = fragments[index];
				cv::flip(fragment, fragment, -1);
			}

		std::vector<std::tuple<cv::Mat, std::string, double>> ret;
		if (
			std::vector<std::tuple<size_t, std::string, double>> results;
			_recogniser.run(fragments, results)
		)
		{
			ret.reserve(results.size());
			for (auto& [index, text, score] : results)
				ret.emplace_back(std::move(indices[index]), std::move(text), score);
		}
		else
		{
			ret.reserve(indices.size());
			for (auto& index : indices)
				ret.emplace_back(std::move(index), std::string {}, 0.0);
		}
		return ret;
	}
};

}
