training_configurations = {
	"mnist_dataset_portion": 1.0,

	"epochs": 15,
	"batch_size": 12,
	"validation_split": 0.1,
	"dropout_rate": 0.5,
	"learning_rate": 1e-3,
	"decay": 1e-5,
	
	"neurasp_conf": {
		"list_of_possible_output_classes": [0,1,2],
		"output_type" : "num_of_zeros",
		"classifiers_conf": {
			"1": {
				"objects": [0,1],
				"list_of_object_classes": [0,1],
				"object_type": "image",
				"classes_type" : "digit"
			}
		}
	}
}
