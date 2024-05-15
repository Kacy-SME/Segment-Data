from metaseg import SegAutoMaskPredictor, SegManualMaskPredictor

# if gpu memory is not enough, reduct the points_per_side and points_per_batch

# For image

# For video

results = SegAutoMaskPredictor().video_predict(
	source="wasp-driving.mp4",
	model_type="vit_l", # vit_l, vit_h, vit_b
	points_per_side=16,
	points_per_batch=64,
	min_area=1000,
	output_path="Output/segmented.mp4"
)

