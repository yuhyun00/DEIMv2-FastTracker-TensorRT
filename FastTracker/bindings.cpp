// pybind11 bindings for the FastTracker C++ multi-object tracker.
//
// Exposes a single class `FastTracker` to Python. Detections go in and tracks
// come out as plain numpy arrays so the Python side never touches C++ types:
//
//   update(dets: np.ndarray[N, 6]) -> np.ndarray[M, 7]
//     input  columns: [x1, y1, x2, y2, score, class_id]   (absolute image coords)
//     output columns: [x1, y1, x2, y2, score, class_id, track_id]
//
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <vector>

#include "FastTracker.h"

namespace py = pybind11;

// Thin Python-facing wrapper around the C++ FastTracker so we can accept and
// return numpy arrays instead of std::vector<Object>/std::vector<STrack>.
class PyFastTracker
{
public:
	PyFastTracker(int frame_rate = 30, int track_buffer = 30)
		: tracker_(frame_rate, track_buffer) {}

	py::array_t<float> update(py::array_t<float, py::array::c_style | py::array::forcecast> dets)
	{
		auto buf = dets.request();

		// Accept an empty array (no detections this frame).
		std::vector<Object> objects;
		if (buf.size != 0)
		{
			if (buf.ndim != 2 || buf.shape[1] != 6)
				throw std::runtime_error(
					"FastTracker.update expects an (N, 6) array: [x1, y1, x2, y2, score, class_id]");

			const ssize_t n = buf.shape[0];
			const float* ptr = static_cast<const float*>(buf.ptr);
			objects.reserve(n);
			for (ssize_t i = 0; i < n; i++)
			{
				const float* row = ptr + i * 6;
				Object obj;
				obj.x1 = row[0];
				obj.y1 = row[1];
				obj.x2 = row[2];
				obj.y2 = row[3];
				obj.prob = row[4];
				obj.label = static_cast<int>(row[5]);
				objects.push_back(obj);
			}
		}

		std::vector<STrack> tracks = tracker_.update(objects);

		// Build the (M, 7) output array.
		const ssize_t m = static_cast<ssize_t>(tracks.size());
		py::array_t<float> out({m, static_cast<ssize_t>(7)});
		auto out_buf = out.request();
		float* out_ptr = static_cast<float*>(out_buf.ptr);
		for (ssize_t i = 0; i < m; i++)
		{
			STrack& t = tracks[i];
			float* row = out_ptr + i * 7;
			row[0] = t.tlbr[0];
			row[1] = t.tlbr[1];
			row[2] = t.tlbr[2];
			row[3] = t.tlbr[3];
			row[4] = t.score;
			row[5] = static_cast<float>(t.label);
			row[6] = static_cast<float>(t.track_id);
		}
		return out;
	}

private:
	FastTracker tracker_;
};

PYBIND11_MODULE(fasttracker, m)
{
	m.doc() = "FastTracker C++ multi-object tracker (pybind11 binding)";

	py::class_<PyFastTracker>(m, "FastTracker")
		.def(py::init<int, int>(),
			py::arg("frame_rate") = 30,
			py::arg("track_buffer") = 30,
			"Create a tracker. frame_rate/track_buffer control how long lost tracks survive.")
		.def("update", &PyFastTracker::update, py::arg("dets"),
			"Update with this frame's detections.\n"
			"dets: (N, 6) float array [x1, y1, x2, y2, score, class_id].\n"
			"returns: (M, 7) float array [x1, y1, x2, y2, score, class_id, track_id].");
}
