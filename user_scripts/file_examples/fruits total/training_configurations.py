min_price = 0.50
max_price = 10.00

training_configurations = {
	"mnist_dataset_portion": 1.0,

	"epochs": 15,
	"batch_size": 12,
	"validation_split": 0.1,
	"dropout_rate": 0.5,
	"learning_rate": 1e-3,
	"decay": 1e-5,
	
	"neurasp_conf": {
		"list_of_possible_output_classes": list(range(int(min_price*100)*3,int(max_price*100)*3+1,50)),
		"output_type" : "total",
		"classifiers_conf": {
			"1": {
				"objects": [0,1,2],
				"list_of_object_classes": list(range(1000)),
				"object_type": "image",
				"classes_type" : "fruit"
			}
		}
	}
}
