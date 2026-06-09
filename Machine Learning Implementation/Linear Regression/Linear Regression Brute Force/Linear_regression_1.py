


class Linear_regression():
    def __init__(self , slope , intercept):
        self.slope = self.m = slope
        self.intercept = self.b = intercept


    def predict(self, value):
        prediction = (self.slope*value) + self.intercept
        return prediction


    



def calculate_mse(x_value , y_value ,model):
    mse = 0
    
    for x,y in zip(x_value,y_value):
        prediction = model.predict(x)
        error = prediction - y
        
        if error != 0:
            mse += (abs(error)**2)
        else:
            continue
    mse = mse / len(x_value)

    return mse
    
   
def train_model(x_value , y_value ,model):
    for pair in possible:
        slope,intercept = pair[0],pair[1]
        model.slope = slope 
        model.intercept = intercept
        current_mse = calculate_mse(x_value, y_value , model)

        if not best_values:
                best_values["model"] = [current_mse , model.slope , model.intercept]
                continue
        
        if current_mse < best_values["model"][0]:
            best_values["model"] = [current_mse ,model.slope , model.intercept]
            continue
    return best_values
    
        

    
                
    

    
        




def main():
    global best_values , possible
    best_values = dict()

    hours = [1, 2, 3, 4, 5, 6]
    marks = [42, 51, 61, 69, 81, 88]

    x_value = int(input("Enter your Hours studied : "))

    model = Linear_regression(0,0)
    best_model_values = train_model(hours, marks , model)
    model.slope = best_model_values["model"][1]
    model.intercept= best_model_values["model"][2]
    prediction = model.predict(x_value)
    print(f"best values : {best_model_values}")
    print(f"prediction is : {prediction}")


if __name__ == "__main__":

    possible = list()
    for m in range(0,21):
        for b in range(0,51):
            possible.append([m,b])
    
    
    main()


"""
DEVELOPER NOTES
1. lets think desgin or lets just go as er chat for now
2. i made class made init fucntion but made 4 var in group of 2 to chekc my cursioty of same varible ,, maybe not a good practise but experiment for now . 
3. i made a method in class linear regression called prediction and putted y = mx + c equation by the model slop and intercept
4. i making a fucntion of calulatin mse first heache of the code is i i wann call predict via linear regresiion but to call it normally in code we need 2 positional argument . \
okay searched and found two solutions . either make it inside class or make a instance like model and pass it to function
5. second headache saying mode =  linear_regression is not suffice i have to give calue but when i do idk why red underline appear



"""