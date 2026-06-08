#pragma once

#include <vector>
#include <map>
#include <climits>
#include "STrack.h"

// Detection passed into the tracker. Box is in absolute image coordinates (xyxy).
struct Object
{
	float x1;
	float y1;
	float x2;
	float y2;
	int label;
	float prob;
};

class FastTracker
{
public:
	FastTracker(int frame_rate = 30, int track_buffer = 30);
	~FastTracker();

	vector<STrack> update(const vector<Object>& objects);

private:
	vector<STrack*> joint_stracks(vector<STrack*> &tlista, vector<STrack> &tlistb);
	vector<STrack> joint_stracks(vector<STrack> &tlista, vector<STrack> &tlistb);

	vector<STrack> sub_stracks(vector<STrack> &tlista, vector<STrack> &tlistb);
	void remove_duplicate_stracks(vector<STrack> &resa, vector<STrack> &resb, vector<STrack> &stracksa, vector<STrack> &stracksb);

	void linear_assignment(vector<vector<float> > &cost_matrix, int cost_matrix_size, int cost_matrix_size_size, float thresh,
		vector<vector<int> > &matches, vector<int> &unmatched_a, vector<int> &unmatched_b);
	vector<vector<float> > iou_distance(vector<STrack*> &atracks, vector<STrack> &btracks, int &dist_size, int &dist_size_size);
	vector<vector<float> > iou_distance(vector<STrack> &atracks, vector<STrack> &btracks);
	vector<vector<float> > ious(vector<vector<float> > &atlbrs, vector<vector<float> > &btlbrs);

	bool is_occluded_by(const vector<float>& box_a, const vector<float>& box_b, float iou_thresh = 0.70);
	double lapjv(const vector<vector<float> > &cost, vector<int> &rowsol, vector<int> &colsol,
		bool extend_cost = false, float cost_limit = LONG_MAX, bool return_cost = true);

private:

	float track_thresh;
	float high_thresh;
	float match_thresh;
	float low_match_thresh;
	float unconfirmed_threshold;
	float Beta_enlarge;
	float Dampen_factor;
	int T_occ;
	int reset_vel_offset;
	int reset_pos_offset;
	int T_recent_occ;

	int frame_id;
	int max_time_lost;

	vector<STrack> tracked_stracks;
	vector<STrack> lost_stracks;
	vector<STrack> removed_stracks;
	fast_kalman::KalmanFilter kalman_filter;
};
