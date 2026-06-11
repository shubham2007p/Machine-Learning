
class Linear_regression():
    def __init__(self , slope_cgpa , slope_project , slope_comm , intercept):
        self.slope_cgpa = slope_cgpa
        self.slope_project = slope_project
        self.slope_comm = slope_comm
        self.intercept = intercept

    def predict(self , cgpa_value , project_value , comm_value):
        prediction = (self.slope_cgpa * cgpa_value)+ (self.slope_project * project_value)+ (self.slope_comm * comm_value)+ self.intercept
        return prediction

def calculate_mse(x_values , y_values , model):
    mse = 0
    for x_value , y_value in zip(x_values , y_values):
        y_predict = model.predict(x_value[0] , x_value[1] , x_value[2]  )
        error = (y_value - y_predict)
        mse += abs(error)**2

    mse = mse / len(x_values)
    return mse

def train_model(x_values , y_values , model, BestModel_parameters):

    if not BestModel_parameters:
        model_cost = calculate_mse(x_values , y_values , model)
        BestModel_parameters["model"] = [model_cost , model.slope_cgpa , model.slope_project , model.slope_comm , model.intercept]

    for cycle in range(200):
        grad_descent(x_values, y_values ,  model)
        model_cost = calculate_mse(x_values , y_values , model)
        BestModel_parameters["model"] = [model_cost , model.slope_cgpa , model.slope_project , model.slope_comm , model.intercept]
        print(f"{cycle} , model : {BestModel_parameters['model']}")
        

    return BestModel_parameters

def grad_descent(x_values , y_values, model):
    learning_rate = 0.001
    dldm_1 = dldm_2 = dldm_3 = dldc = 0
    


    # calculus 
    # now we have more then one feature ofr one label therefore so many partial derivates with each slop
    # i. partial derivative of cost fucntion w.r.t. slope .
    for x_value, y_value in zip(x_values,y_values):
        y_predict = model.predict(x_value[0] , x_value[1] , x_value[2]  )
        # with all slopes
        dldm_1 += -2*(y_value - y_predict)*(x_value[0])
        dldm_2 += -2*(y_value - y_predict)*(x_value[1])
        dldm_3 += -2*(y_value - y_predict)*(x_value[2])
        
    # ii. partial derivative of cost fucntion w.r.t. intercept .
        dldc += -2*(y_value - y_predict)

    # nw we have change of cost fucntion with both and on each sample and sum of it . so now avearge tehm will give me a starting point for both parameters to make chnages as per the learning rate as well


    dldm_1  = dldm_1 / len(x_values)
    dldm_2  = dldm_2 / len(x_values)
    dldm_3  = dldm_3 / len(x_values)
    
    dldc = dldc/ len(x_values)

    model.slope_cgpa -= learning_rate*(dldm_1)
    model.slope_project -= learning_rate*(dldm_2)
    model.slope_comm -= learning_rate*(dldm_3)
    model.intercept -= learning_rate*(dldc)



def main():

    students = [
    [8.5, 4, 7],
    [7.2, 2, 5],
    [9.1, 6, 9],
]

    placement_score = [85,60,95,]

    model = Linear_regression(0,0,0,0)

    BestModel_parameters = dict()

    user_cgpa = float(input("Enter ther cgpa : "))
    user_project = int(input("Enter ther project : "))
    user_comm = int(input("Enter ther comm : "))

    BestModel_parameters = train_model(students , placement_score , model ,BestModel_parameters)

    print(f" prediction : {model.predict(user_cgpa , user_project , user_comm)}")
    

if __name__ == "__main__":
    main()

