
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
    learning_rate = 0.001 # part of learning
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

    placement_score = [85,60,95]

    model = Linear_regression(0,0,0,0)

    BestModel_parameters = dict()

    user_cgpa = float(input("Enter ther cgpa : "))
    user_project = int(input("Enter ther project : "))
    user_comm = int(input("Enter ther comm : "))

    BestModel_parameters = train_model(students , placement_score , model ,BestModel_parameters)

    print(f" prediction : {model.predict(user_cgpa , user_project , user_comm)}")
    

if __name__ == "__main__":
    main()


# =====================================================================================
# 💡 LEARNING LOG: THE GRADIENT EXPLOSION PHENOMENON
# =====================================================================================
# PROBLEM IDENTIFIED: 
# Initially, using a learning rate (alpha) of 0.1 caused the Mean Squared Error (MSE) 
# to explode exponentially (e.g., 6,000 -> 150,000 -> Millions) instead of decreasing.
#
# WHY THIS HAPPENED (The Calculus & Update Step Mechanics):
# 1. LARGE DATA MULTIPLIERS: Our target variables (Placement Scores ~95) and feature 
#    inputs (CGPA ~9, Comm ~9) produce large errors on early iterations.
# 2. MASSIVE GRADIENTS: Because the partial derivative formula multiplies the error 
#    by the raw feature values (e.g., dldm = -2 * error * x_value), the accumulated 
#    gradients (dldm_1, dldm_2, etc.) naturally scale up into thousands (e.g., ~ -1300).
# 3. THE OVERSHOOT EFFECT: In the parameter update step:
#       `model.slope_cgpa -= learning_rate * dldm`
#    Multiplying a massive gradient (~ -1300) by a high learning rate (0.1) results 
#    in a massive step size (+130). 
# 4. WALKING OFF THE CLIFF: Instead of taking a gentle step down to the minimum of 
#    the cost bowl, the parameters completely overshot the optimal valley bottom, 
#    landing much higher up on the opposite wall. This widened the error on the 
#    next cycle, triggering a compounding loop of infinite divergence.
#
# THE SOLUTION IMPLEMENTED:
# Tuned the learning rate down to 0.005. This aggressively forces the parameter 
# steps to remain small, keeping the optimization adjustments bound tightly inside 
# the convex cost bowl, allowing the slopes to smoothly converge toward zero error.
# =====================================================================================

