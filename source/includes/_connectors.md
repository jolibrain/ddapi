# Connectors

The DeepDetect API supports the control of input and output connectors.

* `input connectors` are parametrized with the `input` JSON object

> input connector:

```json
"parameters":{
   "input":{

  }
}
```

* `output connectors` are parametrized with the `output` JSON object

> output connector:

```json
"parameters":{
   "output":{

  }
}
```

<aside class="notice">
Connectors are defined at service creation but their options can be modified in `train` and `predict` calls as needed.
</aside>

## Input connectors
The `connector` field defines its type:

* `image` instantiates the image input connector
* `csv` instantiates the input connector for CSV files
* `txt` instantiates the input connector for text files

Input connectors work almost the same during both the training and prediction phases. But the training phase usually deals with large masses of data, and therefore the connectors above are optimized to automate some tasks, typically building and preprocessing the dataset at training time.

Below is a summary of input connectors options, though they are all already defined in each API resource and call documentation.

- Image

Parameter | Type | Optional | Default | Description
--------- | ---- | -------- | ------- | -----------
width | int | yes | 227 | Resize images to width ("image" only)
height | int | yes | 227 | Resize images to height ("image" only)
bw | bool | yes | false | Treat images as black & white
test_split | real | yes | 0 | Test split part of the dataset
shuffle | bool | yes | false | Whether to shuffle the training set (prior to splitting)

- CSV

Parameter | Type | Optional | Default | Description
--------- | ---- | -------- | ------- | -----------
label | string | no | N/A | Label column name
ignore | array of string | yes | empty | Array of column names to ignore
label_offset | int | yes | 0 | Negative offset (e.g. -1) s othat labels range from 0 onward
separator | string | yes | ',' | Column separator character
id | string | yes | empty | Column name of the training examples identifier field, if any
scale | bool | yes | false | Whether to scale all values into [0,1]
min_vals,max_vals | array | yes | empty | Instead of `scale`, provide the scaling parameters, as returned from a training call
categoricals | array | yes | empty | List of categorical variables
categoricals_mapping | object | yes | empty | Categorical mappings, as returned from a training call
test_split | real | yes | 0 | Test split part of the dataset
shuffle | bool | yes | false | Whether to shuffle the training set (prior to splitting)

- Text

Parameter | Type | Optional | Default | Description
--------- | ---- | -------- | ------- | -----------
count | int | yes | true | whether to count words and report counters
min_count | int | yes | 5 | min word count occurences for a word to be taken into account
min_word_length | int | yes | 5 | min word length for a word to be taken into account
tfidf | bool | yes | false | whether to compute TF/IDF for every word
test_split | real | yes | 0 | Test split part of the dataset
shuffle | bool | yes | false | Whether to shuffle the training set (prior to splitting)

## Output connectors

The output connector is at this stage very simple, and dedicated to supervised machine learning output.

Its two main features are the control of the number of predictions per URI, and the output templating, which allows for custom output and seamless integration in external applications.

Parameter | Type | Optional | Default | Description
--------- | ---- | -------- | ------- | -----------
best | int | yes | 1 | Number of top predictions returned by data URI (supervised)
measure | array | yes | empty | Output measures requested, from `acc`: accuracy, `f1`: f1, precision and recall, `mcll`: multi-class log loss, `auc`: area under the curve, `cmdiag`: diagonal of confusion matrix (requires `f1`), `cmfull`: full confusion matrix (requires `f1`)
template | string | yes | empty | Output template in Mustache format

The variables that are usable in the output template format are those from the standard JSON output. See the [output template](#output-templates) dedicated section for more details and examples.