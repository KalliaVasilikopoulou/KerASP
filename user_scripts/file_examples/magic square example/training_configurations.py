training_configurations = {	
	"N": 3,
	"cell_1_index": "0,0",
	"cell_2_index": "0,1",
	"cell_3_index": "0,2",
	"cell_ukn_index": "2,2",
	"mnist_dataset_portion": 1.0,

	"epochs": 5,
	"batch_size": 12,
	"validation_split": 0.2,
	"dropout_rate": 0.5,
	"learning_rate": 1e-3,
	"decay": 1e-5,
	"add_typed_digits_to_dataset": False,

	"program_specs": {
		"num_of_objects": 3,
		"list_of_object_classes": [1,2,3,4,5,6,7,8,9],
		"list_of_possible_output_classes": [1,2,3,4,5,6,7,8,9],
		"object_type": "cell",
		"output_type" : "cell_22_digit",
		"classes_type" : "digit"
	}
}
