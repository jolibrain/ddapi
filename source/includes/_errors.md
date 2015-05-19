# Errors

The DeepDetect API uses the following error HTTP and associated custom error codes when applicable:


HTTP Status Code | Meaning
---------------- | -------
400 | Bad Request -- Malformed syntax in request or JSON body
403 | Forbidden -- The requested resource or method cannot be accessed
404 | Not Found -- The requested resource, service or model does not exist
409 | Conflict -- The requested method cannot be processed due to a conflict
500 | Internal Server Error -- Other errors, including internal Machine Learning libraries errors

DeepDetect Error Code | Meaning
--------------------- | -------
1000 | Unknown Library -- The requested Machine Learning library is unknown
1001 | No Data -- Empty data provided
1002 | Service Not Found -- Machine Learning service not found
1003 | Job Not Found -- Training job not found
1004 | Input Connector Not Found -- Unknown or incompatible input connector and service
1005 | Service Input Bad Request -- Any service error from input connector
1006 | Service Bad Request -- Any bad parameter request error from Machine Learning library
1007 | Internal ML Library Error -- Internal Machine Learning library error
1008 | Train Predict Conflict -- Algorithm does not support prediction while training