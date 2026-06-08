#pragma once

#include <vector>
#include <deque>
#include "kalmanFilter.h"

using namespace std;

enum TrackState { New = 0, Tracked, Lost, Removed };

class STrack
{
public:
	STrack(vector<float> tlwh_, float score, int label = -1);
	~STrack();

	vector<float> static tlbr_to_tlwh(vector<float> &tlbr);
	void static multi_predict(vector<STrack*> &stracks, fast_kalman::KalmanFilter &kalman_filter);
	void static_tlwh();
	void static_tlbr();
	vector<float> tlwh_to_xyah(vector<float> tlwh_tmp);
	vector<float> to_xyah();
	void mark_lost();
	void mark_removed();
	int next_id();
	int end_frame();

	void activate(fast_kalman::KalmanFilter &kalman_filter, int frame_id);
	void re_activate(STrack &new_track, int frame_id, bool new_id = false);
	void update(STrack &new_track, int frame_id);

public:
	bool is_activated;

	int track_id;
	int state;

	// Class label carried through tracking so the caller keeps class info
	int label;

	// Occlusion handling
	bool is_occluded = false;
	bool was_recently_occluded = false;
	int last_occluded_frame = -1;
	int occluded_len = 0;
	int not_matched = 0;

	vector<float> _tlwh;
	vector<float> tlwh;
	vector<float> tlbr;
	int frame_id;
	int tracklet_len;
	int start_frame;

	KAL_MEAN mean;
	KAL_COVA covariance;
	float score;

	// History buffer
	int max_history_len = 10;  // or more
	std::deque<KAL_MEAN> mean_history;
	std::deque<KAL_COVA> cov_history;


private:
	fast_kalman::KalmanFilter kalman_filter;
};
