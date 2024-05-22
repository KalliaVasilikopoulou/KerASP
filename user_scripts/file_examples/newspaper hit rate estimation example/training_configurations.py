training_configurations = {	"epochs": 10,
	"batch_size": 12,
	"validation_split": 0.2,
	"dropout_rate": 0.5,
	"learning_rate": 1e-3,
	"decay": 1e-5,

	"embedding_dim": 32,
	"max_words_in_dict": 50000,
	"max_words_per_sample" : 20,
	"tokenize_data": true,

	"program_specs": {
		"num_of_objects": 3,
		"list_of_object_classes": [0,1,2,3,4,5,6,7,8,9],
		"list_of_possible_output_classes": [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30],
		"object_type": "article",
		"output_type" : "newspaper_rate",
		"classes_type" : "topic"
	}
}
