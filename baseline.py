import numpy as np
def relu_backward(dA, Z):
    dZ = np.array(dA, copy = True)
    #dZ=np.clip(dZ,-1000,1000)
    #Z=np.clip(Z,-1000,1000)
    #print(Z)
    dZ[Z <= 0] = 0
    return dZ
def single_layer_backward_propagation(dA_curr, W_curr, b_curr, Z_curr, A_prev, activation="relu"):
    #m = A_prev.shape[1]
    
    if activation == "relu":
        backward_activation_func = relu_backward
    elif activation == "sigmoid":
        backward_activation_func = sigmoid_backward
    else:
        raise Exception('Non-supported activation function')
    
    dZ_curr = backward_activation_func(dA_curr, Z_curr)
    dW_curr = np.dot(dZ_curr, A_prev.T) 
    dW_curr=np.clip(dW_curr,-10**5,100000)
    db_curr = np.sum(dZ_curr, axis=1, keepdims=True)
    db_curr=np.clip(db_curr,-10**5,100000)
    dA_prev = np.dot(W_curr.T, dZ_curr)
    #dA_prev=np.clip(dA_prev,-1000,1000)
    return dA_prev, dW_curr, db_curr
def x_full_backward_propagation(Y_hat,reward,memory, params_values, nn_architecture):
    grads_values = {}
    #m = Y.shape[1]
    #Y = Y.reshape(Y_hat.shape)
   
    dA_prev = 2*(Y_hat-reward)
    
    for layer_idx_prev, layer in reversed(list(enumerate(nn_architecture))):
        layer_idx_curr = layer_idx_prev + 1
        activ_function_curr = layer["activation"]
        
        dA_curr = dA_prev
        
        A_prev = memory["A" + str(layer_idx_prev)]
        Z_curr = memory["Z" + str(layer_idx_curr)]
        #Z_curr=np.clip(Z_curr,-1000,1000)
        W_curr = params_values["W" + str(layer_idx_curr)]
        b_curr = params_values["b" + str(layer_idx_curr)]
        
        dA_prev, dW_curr, db_curr = single_layer_backward_propagation(
            dA_curr, W_curr, b_curr, Z_curr, A_prev, activ_function_curr)
        #dW_curr=np.clip(dW_curr,-1000,1000)
        #db_curr=np.clip(db_curr,-1000,1000)
        grads_values["dW" + str(layer_idx_curr)] = dW_curr
        grads_values["db" + str(layer_idx_curr)] = db_curr
    
    return grads_values

def update_b(params_values, grads_values, nn_architecture, learning_rate):
    for layer_idx, layer in enumerate(nn_architecture):
        params_values["W" + str(layer_idx+1)] -= learning_rate * grads_values["dW" + str(layer_idx+1)]        
        params_values["b" + str(layer_idx+1)] -= learning_rate * grads_values["db" + str(layer_idx+1)]
    

    return params_values
def b_full_backward_propagation(Y_hat,reward,memory, params_values, nn_architecture):
    grads_values = {}
    #m = Y.shape[1]
    #Y = Y.reshape(Y_hat.shape)
   
    dA_prev = 2*(reward-Y_hat)
    
    for layer_idx_prev, layer in reversed(list(enumerate(nn_architecture))):
        layer_idx_curr = layer_idx_prev + 1
        activ_function_curr = layer["activation"]
        
        dA_curr = dA_prev
        
        A_prev = memory["A" + str(layer_idx_prev)]
        Z_curr = memory["Z" + str(layer_idx_curr)]
        #Z_curr=np.clip(Z_curr,-1000,1000)
        W_curr = params_values["W" + str(layer_idx_curr)]
        b_curr = params_values["b" + str(layer_idx_curr)]
        
        dA_prev, dW_curr, db_curr = single_layer_backward_propagation(
            dA_curr, W_curr, b_curr, Z_curr, A_prev, activ_function_curr)
        #dW_curr=np.clip(dW_curr,-1000,1000)
        #db_curr=np.clip(db_curr,-1000,1000)
        grads_values["dW" + str(layer_idx_curr)] = dW_curr
        grads_values["db" + str(layer_idx_curr)] = db_curr
    
    return grads_values
