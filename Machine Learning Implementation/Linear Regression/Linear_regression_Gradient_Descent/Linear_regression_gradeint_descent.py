from re import L


class Linear_regression():
    def __init__(self , slope , intercept):
        self.slope = slope
        self.intercept = intercept

    def predict(self , x_value):
        prediction = (self.slope * x_value) + self.intercept
        return prediction

def calculate_mse(x_values , y_values , model):
    mse = 0
    for x_value , y_value in zip(x_values , y_values):
        y_predict = model.predict(x_value)
        error = (y_predict - y_value)
        mse += abs(error)**2

    mse = mse / len(x_values)
    return mse

def train_model(x_values , y_values , model, BestModel_parameters):

    if not BestModel_parameters:
        model_cost = calculate_mse(x_values , y_values , model)
        BestModel_parameters["model"] = [model_cost , model.slope , model.intercept]

    for cycle in range(200):
        grad_descent(x_values, y_values ,  model)
        model_cost = calculate_mse(x_values , y_values , model)
        BestModel_parameters["model"] = [model_cost , model.slope , model.intercept]
        print(f"{cycle} , model : {BestModel_parameters['model']}")
        

    return BestModel_parameters

def grad_descent(x_values , y_values, model):
    learning_rate = 0.1
    dldm = dldc = 0
    


    # calculus 
    # i. partial derivative of cost fucntion w.r.t. slope .
    for x_value, y_value in zip(x_values,y_values):
        y_predict = model.predict(x_value)
        dldm += -2*(y_value - y_predict)*(x_value)
    # ii. partial derivative of cost fucntion w.r.t. intercept .
        dldc += -2*(y_value - y_predict)

    # nw we have change of cost fucntion with both and on each sample and sum of it . so now avearge tehm will give me a starting point for both parameters to make chnages as per the learning rate as well

    dldm = dldm / len(x_values)
    dldc = dldc/ len(x_values)

    model.slope -= learning_rate*(dldm)
    model.intercept -= learning_rate*(dldc)



def main():

    hours = [1,2,3,4]
    marks = [42,51,61,69]

    model = Linear_regression(0,0)

    BestModel_parameters = dict()

    user_value = int(input("Enter ther value"))

    BestModel_parameters = train_model(hours , marks , model ,BestModel_parameters)

    print(f" prediction : {model.predict(user_value)}")
    

if __name__ == "__main__":
    main()

