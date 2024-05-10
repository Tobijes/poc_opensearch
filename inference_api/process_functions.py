from time import perf_counter

########################################################
### Functions that will be run in the worker process ###
########################################################
def worker_create_model(model_type):
    global model
    model = model_type()
 
 
def worker_model_predict(args):
    start_time = perf_counter()
    result = model.predict(*args) 
    inference_time = perf_counter() - start_time
    return inference_time, result
########################################################