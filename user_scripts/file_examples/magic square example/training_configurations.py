training_configurations = {
	"N": 3,
	"cell_1_index": "0,0",
	"cell_2_index": "0,1",
	"cell_3_index": "0,2",
	"cell_ukn_index": "2,2",
	"mnist_dataset_portion": 1.0,
	"add_typed_digits_to_dataset": False,

	"epochs": 10,
	"batch_size": 12,
	"validation_split": 0.2,
	"dropout_rate": 0.5,
	"learning_rate": 1e-3,
	"decay": 1e-5,
	
	"neurasp_conf": {
		"list_of_possible_output_classes": [1,2,3,4,5,6,7,8,9],
		"output_type" : "cell_22_digit",
		"classifiers_conf": {
			"1": {
				"objects": [0],
				"list_of_object_classes": [1,2,3,4,5,6,7,8,9],
				"object_type": "image",
				"classes_type" : "digit"
			},
			"2": {
				"objects": [1],
				"list_of_object_classes": [1,2,3,4,5,6,7,8,9],
				"object_type": "image",
				"classes_type" : "digit"
			},
			"3": {
				"objects": [2],
				"list_of_object_classes": [1,2,3,4,5,6,7,8,9],
				"object_type": "image",
				"classes_type" : "digit"
			}
		}
	}
}
