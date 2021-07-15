# VOC_semantic_evaluation
calculating the mIOU of each class with given labels and model's result
# Run
> python test_IOU.py --check_file check_list.txt --label_dir label_data --check_dir generated_data
# Note
the param 'label_dir' is a directory saving the colorful imgs(.png) as ground truth.

the param 'check_dir' is a directory saving the unit imgs(.png) which will be generated by model, the pixel value of unit img is your predicted class_id.

the param 'check_file' is a text list recording the imgs-name for evaluation, here I rename the 'trainval.txt' in PASCAL VOC12 dataset as 'check_list.txt'. 

more details is shown in the code ~

# Accuracy
I have checked the eavaluation result of VOC12_val with VOC server result, and the comparation demonstrates its correction.
